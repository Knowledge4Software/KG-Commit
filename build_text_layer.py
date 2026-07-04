"""
V3 B1 -- Commit-text as a first-class SEMANTIC GRAPH LAYER.

In V2 the commit message was only a per-Commit TF-IDF *feature vector* (the
weakest single online signal). V3 lifts text into the graph STRUCTURE so it can
propagate relationally and be GROUNDED in the code entities a commit touches:

    (Commit)-[:MENTIONS {tf, df}]->(Term)              # bipartite text layer
    (Term)-[:GROUNDS_IN]->(ASTNode)                    # term == identifier leaf value
    (Term)-[:REFERS_TO]->(File)                        # term appears in a changed path
    (Commit)-[:FIXES_ISSUE]->(Issue)                   # already exists (hard anchor)

Two Term kinds are distinguished:
  * kind="nl"   -- lemmatised natural-language words (stopword-pruned, min_df).
  * kind="code" -- code identifiers / API names lifted from the message by light
                   NER (camelCase, PascalCase, snake_case, CONST, a.b() calls).

This module has a PURE, dependency-light extraction core (`extract_terms`,
testable with no Neo4j) and a Neo4j ingestion pass (`build_layer`) that reuses
the project's driver/label conventions. No transformer, no GPU.

  python build_text_layer.py --selftest      # extractor unit test, no DB
  python build_text_layer.py --build         # build the layer in Neo4j
  python build_text_layer.py --build --ground # also add GROUNDS_IN / REFERS_TO
"""
import re, sys, math
from collections import defaultdict, Counter

# sklearn ships a curated English stoplist; avoids an nltk dependency.
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SW
    STOP = set(_SW)
except Exception:                                    # minimal fallback
    STOP = set("the a an and or of to in is are be this that for with on it as at "
               "by from we our i you he she they them was were has have had not no "
               "if then else when which who what how".split())

NEO4J_URI = "bolt://localhost:7687"; NEO4J_AUTH = ("neo4j", "password1234")

# ── light NER / tokenisation (pure, no deps) ─────────────────────────────────
ISSUE_RE  = re.compile(r"\b([A-Z]+-\d+|#\d+|GH-\d+)\b")
CALL_RE   = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*\(")     # a.b(  -> a, b
CAMEL_RE  = re.compile(r"\b[a-z]+(?:[A-Z]\w*)+\b")                    # fooBarBaz
PASCAL_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+){2,}\b")                # FooBarBaz
SNAKE_RE  = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")                 # foo_bar
CONST_RE  = re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)*\b")     # MAX_SIZE
WORD_RE   = re.compile(r"[A-Za-z][A-Za-z]+")                          # nl words (>=2 alpha)


def _split_identifier(idn):
    """Split camelCase / snake_case / Pascal into constituent lowercase words."""
    parts = re.split(r"[_\s]+", idn)
    out = []
    for p in parts:
        out += re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", p)
    return [w.lower() for w in out if w]


def _lemmatise(w):
    """Cheap, deterministic suffix normaliser (no nltk): plurals & common verb
    endings. Good enough to merge tokens; not linguistically perfect."""
    for suf in ("izes", "ized", "izing", "ies", "ing", "ers", "er", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            if suf == "ies":
                return w[:-3] + "y"
            return w[:-len(suf)]
    return w


def extract_terms(message):
    """Pure extractor. Returns dict with:
        nl:     Counter{lemma: tf}          natural-language content words
        code:   Counter{identifier: tf}     raw code identifiers (NER)
        code_words: Counter{word: tf}       identifier sub-words (for grounding)
        issues: [issue_key,...]
    """
    msg = message or ""
    issues = sorted(set(ISSUE_RE.findall(msg)))

    code = Counter()
    for a, b in CALL_RE.findall(msg):
        code[a] += 1; code[b] += 1
    for rx in (CAMEL_RE, PASCAL_RE, SNAKE_RE, CONST_RE):
        for m in rx.findall(msg):
            code[m] += 1

    # natural-language words: drop stopwords, issue tokens, pure identifiers
    code_surface = set(code)
    nl = Counter()
    code_words = Counter()
    for m in code:                                    # index identifier sub-words
        for w in _split_identifier(m):
            if w not in STOP and len(w) > 1:
                code_words[w] += code[m]
    lowered = ISSUE_RE.sub(" ", msg)
    for w in WORD_RE.findall(lowered):
        lw = w.lower()
        if lw in STOP or w in code_surface:
            continue
        nl[_lemmatise(lw)] += 1
    return dict(nl=nl, code=code, code_words=code_words, issues=issues)


# ── Neo4j ingestion (needs a live DB) ────────────────────────────────────────
def _load_commits(session):
    return session.run("""
        MATCH (c:Commit {in_jit:true})
        RETURN c.id AS id, c.message AS msg
    """).data()


def _ensure_indexes(session):
    """Indexes are essential on the containerised DB: without them MERGE (:Term)
    and the value==text grounding join do full scans over millions of nodes."""
    for stmt in (
        "CREATE INDEX term_id   IF NOT EXISTS FOR (t:Term)    ON (t.id)",
        "CREATE INDEX term_text IF NOT EXISTS FOR (t:Term)    ON (t.text)",
        "CREATE INDEX astnode_value IF NOT EXISTS FOR (a:ASTNode) ON (a.value)",
    ):
        session.run(stmt)
    session.run("CALL db.awaitIndexes(300)")


def build_layer(driver, ground=False, min_df=3):
    """Create Term nodes + MENTIONS (tf/df weighted); optionally ground terms to
    ASTNode identifier leaves and Files. Idempotent (MERGE)."""
    with driver.session() as s:
        _ensure_indexes(s)
        commits = _load_commits(s)
        print(f"{len(commits)} labelled commits")

        # first pass: document frequencies for pruning + idf on MENTIONS
        df = Counter()
        per_commit = {}
        for r in commits:
            t = extract_terms(r["msg"])
            terms = Counter()
            for w, k in t["nl"].items():       terms[("nl", w)]   += k
            for w, k in t["code"].items():     terms[("code", w)] += k
            per_commit[r["id"]] = terms
            for key in terms:
                df[key] += 1
        N = len(commits)
        kept = {k for k, c in df.items() if c >= min_df}
        print(f"terms: {len(df)} raw -> {len(kept)} kept (min_df={min_df})")

        # write Term nodes + MENTIONS in batches
        rows = []
        for cid, terms in per_commit.items():
            for (kind, w), tf in terms.items():
                if (kind, w) not in kept:
                    continue
                rows.append(dict(cid=cid, tid=f"{kind}:{w}", text=w, kind=kind,
                                 tf=int(tf), idf=math.log(1.0 + N / df[(kind, w)])))
        print(f"MENTIONS edges: {len(rows)}")
        for i in range(0, len(rows), 5000):
            batch = rows[i:i+5000]
            s.execute_write(lambda tx: tx.run("""
                UNWIND $rows AS r
                MERGE (t:Term {id:r.tid}) ON CREATE SET t.text=r.text, t.kind=r.kind
                WITH t, r MATCH (c:Commit {id:r.cid})
                MERGE (c)-[m:MENTIONS]->(t) SET m.tf=r.tf, m.tfidf=r.tf*r.idf
            """, rows=batch))

        if ground:
            _ground_terms(s)


def _ground_terms(session):
    """GROUNDS_IN: a code Term whose text equals an alive identifier-leaf value.
    REFERS_TO: a Term whose text is a token of a changed File path."""
    print("grounding code Terms to AST identifier leaves...")
    session.execute_write(lambda tx: tx.run("""
        MATCH (t:Term {kind:'code'})
        MATCH (a:ASTNode {is_leaf:true}) WHERE coalesce(a.alive,true) AND a.value = t.text
        MERGE (t)-[:GROUNDS_IN]->(a)
    """))
    print("grounding Terms to Files (path token match)...")
    session.execute_write(lambda tx: tx.run("""
        MATCH (t:Term)
        MATCH (:Commit)-[:MODIFIED|ADDED]->(f:File)
        WHERE toLower(f.id) CONTAINS toLower(t.text) AND size(t.text) >= 4
        MERGE (t)-[:REFERS_TO]->(f)
    """))


def _selftest():
    samples = [
        "GROOVY-1234: Fix NullPointerException in AstBuilder.visitClass when parsing generics",
        "Refactor the getMetaClass() lookup; rename FOO_BAR constant and update call_site cache",
        "minor: tidy imports and whitespace",
    ]
    for m in samples:
        t = extract_terms(m)
        print("\nMSG:", m)
        print("  issues:", t["issues"])
        print("  code  :", dict(t["code"]))
        print("  nl    :", dict(t["nl"]))
        print("  cwords:", dict(t["code_words"]))


def main():
    if "--selftest" in sys.argv or not any(a.startswith("--build") for a in sys.argv):
        _selftest(); return
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    build_layer(driver, ground=("--ground" in sys.argv))
    driver.close()
    print("done. Re-run v2_audit.py / inference to use the text layer.")


if __name__ == "__main__":
    main()
