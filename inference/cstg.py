"""
CSTG -- Commit Semantic-Text Graph.

A third first-class layer for the commit KG, alongside the AST and delta layers.
Where the delta layer models the STRUCTURE of a code change, the CSTG models the
STRUCTURE of its textual description (commit message + diff), grounded in code.

Theory (each citeable):
  * Graph-of-Words / TW-IDF (Rousseau & Vazirgiannis, CIKM'13): each commit's text
    is a word co-occurrence graph; a term's weight is its TextRank centrality in
    that graph, x IDF. TW-IDF beats TF-IDF because it captures term dependence and
    position, not just frequency.
  * Corpus term graph via NPMI (cf. TextGCN, Yao et al. AAAI'19): a global
    Term-[COOCCURS {npmi}]-Term graph enables higher-order semantic propagation and
    mitigates sparsity -- the basis for graph-based text classification.
  * Defect-semantic typing: terms are typed {code, bug, action, error, nl} so the
    graph encodes defect semantics, not just tokens.
  * Term-risk propagation: a commit's textual risk = TW-IDF-weighted aggregate of
    its terms' bug-rates, SMOOTHED over the NPMI graph -- a relational signal a flat
    bag-of-words cannot express. All statistics are fit on TRAIN commits only.

This module is pure-Python/CPU (no Neo4j needed for the features); a companion
ingestion (build_cstg_neo4j) writes the same graph into Neo4j for querying/viz.

Public:
  parse_commit_text(diff_text) -> dict(message, added, removed, paths, seq)
  CSTG().fit(texts, y, train_idx) ; .transform(texts) -> feature bundle
"""
import re, math
import numpy as np, scipy.sparse as sp
from collections import defaultdict, Counter

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SW
    STOP = set(_SW)
except Exception:
    STOP = set("the a an and or of to in is are be this that for with on it as at by from".split())

# defect-semantic lexicons (small, high-precision)
BUG_LEX = {"bug", "fix", "fixed", "fixes", "fixing", "error", "errors", "fail", "failure",
           "failing", "crash", "npe", "null", "leak", "leaks", "race", "deadlock", "hang",
           "regression", "exception", "corrupt", "overflow", "underflow", "infinite",
           "wrong", "incorrect", "broken", "issue", "defect", "fault", "invalid",
           "unexpected", "missing", "mismatch", "unsafe", "concurrency", "concurrent",
           "synchronization", "thread", "threading", "memory", "boundary"}
ACTION_LEX = {"add", "added", "remove", "removed", "delete", "deleted", "refactor",
              "refactored", "rename", "renamed", "move", "moved", "update", "updated",
              "revert", "reverted", "cleanup", "optimize", "simplify", "replace",
              "introduce", "implement", "support", "handle", "improve", "change"}

CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]{2,}|[A-Z]{2,}")
IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
ERRTYPE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:Exception|Error))\b")
CODE_IDENT = re.compile(r"[a-z]+[A-Z]\w*|[A-Z][a-z]+[A-Z]\w*|\w+_\w+|\w+\.\w+")


def _split_ident(tok):
    return [w.lower() for w in CAMEL.findall(tok) if len(w) > 1]


def parse_commit_text(diff_text):
    """Split a git-show blob into message header, added/removed code lines, paths."""
    t = str(diff_text)
    # header = everything before the first diff marker
    m = re.search(r"^(diff --git|---\s|\+\+\+\s|@@ )", t, re.M)
    header = t[:m.start()] if m else t[:400]
    body = t[m.start():] if m else ""
    added, removed, paths, msg = [], [], [], []
    # message: header lines that are not metadata
    for ln in header.splitlines():
        s = ln.strip()
        if not s or s.startswith(("commit ", "Author:", "Date:", "git-svn-id", "Merge:", "index ")):
            continue
        msg.append(s)
    for ln in body.splitlines():
        if ln.startswith("+++") or ln.startswith("---"):
            mp = re.search(r"[ab]/(.+)$", ln)
            if mp: paths.append(mp.group(1))
        elif ln.startswith("+"):
            added.append(ln[1:])
        elif ln.startswith("-"):
            removed.append(ln[1:])
    return dict(message=" ".join(msg), added="\n".join(added),
                removed="\n".join(removed), paths=paths)


_NOISE = re.compile(r"tokenset|^mk_|^_+|_?\d{2,}$|^[a-z]$|\d{3,}|^[0-9a-f]{6,}$")

def _noise(term):
    """Drop generated-parser / non-semantic tokens (ANTLR tokensets, long hex,
    numeric-suffixed constants) that pollute the vocabulary without meaning."""
    return len(term) > 30 or bool(_NOISE.search(term))


def _tokens(text, code=False):
    """Tokenise. For code, also split identifiers into sub-words but KEEP the
    original identifier and any *Exception/*Error type as high-value tokens."""
    out = []
    if code:
        for et in ERRTYPE.findall(text):
            if not _noise(et.lower()): out.append(("error", et.lower()))
        for idt in CODE_IDENT.findall(text):
            lo = idt.lower()
            if not _noise(lo): out.append(("code", lo))
            out += [("nl", w) for w in _split_ident(idt) if not _noise(w)]
    for w in re.findall(r"[A-Za-z]{2,}", text.lower()):
        if w in STOP or _noise(w):
            continue
        if w in BUG_LEX:      out.append(("bug", w))
        elif w in ACTION_LEX: out.append(("action", w))
        else:                 out.append(("nl", w))
    return out


def commit_terms(parsed, max_tokens=4000):
    """Ordered (type, term) token stream for one commit (message + diff added/removed).
    Removed-line tokens are suffixed so 'X removed' differs from 'X added' where it
    matters? -- we keep them merged but track type; positional order feeds Graph-of-Words."""
    toks = []
    toks += _tokens(parsed["message"], code=True)
    toks += _tokens(parsed["added"], code=True)
    toks += _tokens(parsed["removed"], code=True)
    return toks[:max_tokens]


def graph_of_words_weights(term_seq, window=4):
    """TextRank centrality of each term within one commit's word-graph.
    term_seq: list of terms (strings) in order. Returns {term: centrality}."""
    terms = [t for t in term_seq]
    uniq = list(dict.fromkeys(terms))
    if len(uniq) == 1:
        return {uniq[0]: 1.0}
    idx = {t: i for i, t in enumerate(uniq)}
    n = len(uniq)
    W = np.zeros((n, n))
    for i in range(len(terms)):
        for j in range(i + 1, min(i + window, len(terms))):
            a, b = idx[terms[i]], idx[terms[j]]
            if a != b:
                W[a, b] += 1.0; W[b, a] += 1.0
    deg = W.sum(1); deg[deg == 0] = 1.0
    P = W / deg[:, None]
    r = np.ones(n) / n
    for _ in range(30):
        r = 0.15 / n + 0.85 * (P.T @ r)
    return {t: float(r[idx[t]]) for t in uniq}


class CSTG:
    """Fit corpus statistics on TRAIN commits; transform any commits to features.

    Feature bundle (all leakage-safe):
      X_twidf   : sparse commit x term, TW-IDF (TextRank centrality x idf)
      prior     : dense (n,), NPMI-propagated term-risk aggregated per commit
      typed     : dense (n, 5), TW-IDF mass per term-type {code,bug,action,error,nl}
    """
    def __init__(self, min_df=3, npmi_thresh=0.15, prop_iters=8, prop_damp=0.6):
        self.min_df = min_df; self.npmi_thresh = npmi_thresh
        self.prop_iters = prop_iters; self.prop_damp = prop_damp

    def _prep(self, texts):
        parsed = [parse_commit_text(t) for t in texts]
        toks = [commit_terms(p) for p in parsed]                 # [(type,term)]
        return parsed, toks

    def fit(self, texts, y, train_idx):
        self._parsed, self._toks = self._prep(texts)
        tr = set(int(i) for i in train_idx)
        # vocabulary + idf + term type (majority) from TRAIN only
        df = Counter(); ttype = {}
        for i in train_idx:
            seen = set()
            for typ, term in self._toks[i]:
                ttype.setdefault(term, typ)
                if term not in seen:
                    df[term] += 1; seen.add(term)
        self.vocab = {t: j for j, t in enumerate(sorted(k for k, c in df.items() if c >= self.min_df))}
        self.ttype = {t: ttype[t] for t in self.vocab}
        Ntr = len(train_idx)
        self.idf = {t: math.log((1 + Ntr) / (1 + df[t])) + 1 for t in self.vocab}

        # NPMI term-term graph from TRAIN commit co-occurrence
        pair = Counter(); occ = Counter()
        for i in train_idx:
            terms = sorted({t for _, t in self._toks[i] if t in self.vocab})
            for a in terms: occ[a] += 1
            for a in range(len(terms)):
                for b in range(a + 1, len(terms)):
                    pair[(terms[a], terms[b])] += 1
        self.npmi = {}
        for (a, b), c in pair.items():
            if c < 2: continue
            pa, pb, pab = occ[a] / Ntr, occ[b] / Ntr, c / Ntr
            pmi = math.log(pab / (pa * pb))
            npmi = pmi / (-math.log(pab))
            if npmi >= self.npmi_thresh:
                self.npmi.setdefault(a, []).append((b, npmi))
                self.npmi.setdefault(b, []).append((a, npmi))

        # base term risk (train bug-rate) then propagate over NPMI graph
        g = float(np.mean([y[i] for i in train_idx]))
        tb = Counter(); tt = Counter()
        for i in train_idx:
            for t in {t for _, t in self._toks[i] if t in self.vocab}:
                tt[t] += 1; tb[t] += int(y[i])
        base = {t: (tb[t] + g * 5) / (tt[t] + 5) for t in self.vocab}
        risk = dict(base)
        for _ in range(self.prop_iters):
            new = {}
            for t in self.vocab:
                nb = self.npmi.get(t, [])
                if nb:
                    wsum = sum(w for _, w in nb)
                    agg = sum(w * risk.get(u, g) for u, w in nb) / wsum
                    new[t] = (1 - self.prop_damp) * base[t] + self.prop_damp * agg
                else:
                    new[t] = base[t]
            risk = new
        self.term_risk = risk
        self.global_rate = g
        return self

    def transform(self, texts=None):
        toks = self._toks if texts is None else self._prep(texts)[1]
        n = len(toks); V = len(self.vocab)
        rows, cols, vals = [], [], []
        prior = np.zeros(n); typed = np.zeros((n, 5))
        TYPES = {"code": 0, "bug": 1, "action": 2, "error": 3, "nl": 4}
        for i, tk in enumerate(toks):
            seq = [t for _, t in tk if t in self.vocab]
            if not seq:
                prior[i] = self.global_rate; continue
            cen = graph_of_words_weights(seq)                    # TextRank per term
            wsum = 0.0; rsum = 0.0
            for t, c in cen.items():
                w = c * self.idf[t]
                rows.append(i); cols.append(self.vocab[t]); vals.append(w)
                typed[i, TYPES[self.ttype[t]]] += w
                rsum += w * self.term_risk.get(t, self.global_rate); wsum += w
            prior[i] = rsum / wsum if wsum else self.global_rate
        X = sp.csr_matrix((vals, (rows, cols)), shape=(n, V))
        return dict(X_twidf=X, prior=prior, typed=typed)
