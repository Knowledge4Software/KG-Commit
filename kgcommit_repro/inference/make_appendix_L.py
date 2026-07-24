"""
Appendix L -- Knowledge-graph SCHEMA and illustrative CYPHER SUBGRAPH EXPORTS.

Produces self-contained, academically-styled HTML pages (no external assets, so
they open offline and embed cleanly in a supplementary bundle):

  1. appL_schema.html        -- the meta-schema of the deployed KG-Commit graph:
                                node-label inventory (with live counts), relationship
                                inventory, the (src)-[rel]->(dst) meta-triples that
                                define the three tiers, and a rendered schema diagram.
  2. appL_export_core.html   -- Core process tier around one labelled commit
                                (Commit -> File / Developer / Issue / parent Commit).
  3. appL_export_ast.html    -- AST structural-change tier for that commit
                                (Commit -[ADDS/UPDATES/REMOVES/MOVES]-> ASTNode).
  4. appL_export_cstg.html   -- CSTG semantic-text tier for that commit
                                (Commit -[MENTIONS]-> Term -[REFERS_TO]-> File,
                                 Term -[COOCCURS]- Term).

Each export page shows the exact Cypher query used and a node-link rendering of the
returned subgraph, drawn as inline SVG (deterministic radial layout; no JS libs).

C3: needs the target project's graph RESIDENT in Neo4j. Run while a dump is restored:
    KGC_PROJECT=zookeeper python snapshot_neo4j.py restore
    KGC_PROJECT=zookeeper python inference/make_appendix_L.py [--commit <id>]

The paper's FINAL design has no method-scoped AST layer, so ASTMethodNode /
HAS_AST_METHOD are excluded from the schema and all exports.
"""
import argparse
import html
import math
from pathlib import Path

from neo4j import GraphDatabase

import _kgc_paths  # noqa: F401
from config.project_config import PROJECT, NEO4J_URI, NEO4J_AUTH

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "Paper" / "paper_material" / "appendices" / "L_schema_cypher"
OUT.mkdir(parents=True, exist_ok=True)

# Excluded from the paper's final design (method-scoped AST was dropped).
EXCLUDE_LABELS = {"ASTMethodNode"}
EXCLUDE_RELS = {"HAS_AST_METHOD"}

# Three-tier grouping + a consistent colour per tier (colour-blind-safe, print-friendly).
TIER = {
    "Core":  (["Project", "Branch", "Commit", "File", "Developer", "Issue", "Intent"], "#2166AC"),
    "AST":   (["ASTNode"], "#B2182B"),
    "Subgraph": (["CFGNode", "DFGNode", "PDGNode", "SEQNode"], "#7B3294"),
    "CSTG":  (["Term"], "#1B7837"),
}
LABEL_TIER = {lab: t for t, (labs, _) in TIER.items() for lab in labs}
LABEL_COLOR = {lab: c for t, (labs, c) in TIER.items() for lab in labs}
TIER_COLOR = {t: c for t, (_, c) in TIER.items()}


# --------------------------------------------------------------------------- #
#  HTML scaffold (self-contained; light/dark aware; academic serif headings)   #
# --------------------------------------------------------------------------- #
def _page(title, body):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
 :root{{--fg:#1a1a1a;--bg:#ffffff;--mut:#666;--line:#ddd;--card:#f7f7f9;}}
 @media(prefers-color-scheme:dark){{:root{{--fg:#e8e8e8;--bg:#161719;--mut:#9aa;--line:#333;--card:#1e2023;}}}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);
   font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
 .wrap{{max-width:1000px;margin:0 auto;padding:2.2rem 1.4rem 4rem}}
 h1,h2,h3{{font-family:Georgia,"Times New Roman",serif;font-weight:600;line-height:1.25}}
 h1{{font-size:1.7rem;margin:.2rem 0 .3rem}} h2{{font-size:1.2rem;margin:2rem 0 .6rem;
   border-bottom:1px solid var(--line);padding-bottom:.3rem}}
 .sub{{color:var(--mut);margin:0 0 1.4rem}}
 table{{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:13.5px}}
 th,td{{text-align:left;padding:.4rem .6rem;border-bottom:1px solid var(--line)}}
 th{{color:var(--mut);font-weight:600}} td.n{{text-align:right;font-variant-numeric:tabular-nums}}
 code,pre{{font-family:"SF Mono",Consolas,Menlo,monospace}}
 pre.cy{{background:var(--card);border:1px solid var(--line);border-left:3px solid #2166AC;
   border-radius:6px;padding:.9rem 1rem;overflow-x:auto;font-size:12.5px;line-height:1.5}}
 .k{{color:#2166AC;font-weight:600}} .s{{color:#1B7837}}
 .chips{{display:flex;flex-wrap:wrap;gap:.4rem;margin:.5rem 0 0}}
 .chip{{display:inline-flex;align-items:center;gap:.35rem;font-size:12px;color:var(--mut)}}
 .dot{{width:10px;height:10px;border-radius:50%;display:inline-block}}
 .fig{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:1rem;margin:.8rem 0}}
 .cap{{color:var(--mut);font-size:12.5px;margin-top:.5rem}}
 svg{{max-width:100%;height:auto;display:block;margin:auto}}
 .foot{{color:var(--mut);font-size:12px;margin-top:2.5rem;border-top:1px solid var(--line);padding-top:.8rem}}
</style></head><body><div class="wrap">{body}
<p class="foot">KG-Commit &mdash; Appendix L. Generated from the resident
<b>{html.escape(PROJECT)}</b> graph in Neo4j. Self-contained; no external assets.</p>
</div></body></html>"""


def _legend():
    chips = "".join(
        f'<span class="chip"><span class="dot" style="background:{c}"></span>{t} tier</span>'
        for t, c in TIER_COLOR.items())
    return f'<div class="chips">{chips}</div>'


# --------------------------------------------------------------------------- #
#  Deterministic radial SVG node-link drawing of a small subgraph              #
# --------------------------------------------------------------------------- #
def _svg_subgraph(center, nodes, edges, w=880, h=520):
    """center=(id,label,caption); nodes=[(id,label,caption)]; edges=[(src,dst,rel)]."""
    # Hard dedup: one circle per node id, one line per (src,dst,rel).
    _nseen, _nd = set(), []
    for n in nodes:
        if n[0] not in _nseen:
            _nseen.add(n[0]); _nd.append(n)
    nodes = _nd
    _eseen, _ed = set(), []
    for e in edges:
        if e not in _eseen:
            _eseen.add(e); _ed.append(e)
    edges = _ed
    cx, cy = w / 2, h / 2
    pos = {center[0]: (cx, cy)}
    others = [n for n in nodes if n[0] != center[0]]
    R = min(w, h) * 0.38
    for i, n in enumerate(others):
        ang = 2 * math.pi * i / max(1, len(others)) - math.pi / 2
        pos[n[0]] = (cx + R * math.cos(ang), cy + R * math.sin(ang))
    byid = {n[0]: n for n in nodes}
    byid[center[0]] = center

    def esc(s): return html.escape(str(s))
    svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
           f'font-family="-apple-system,Segoe UI,Roboto,sans-serif">']
    svg.append('<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="8" refY="3" '
               'orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#9aa"/></marker></defs>')
    # edges
    for s, d, rel in edges:
        if s not in pos or d not in pos:
            continue
        (x1, y1), (x2, y2) = pos[s], pos[d]
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy) or 1
        ux, uy = dx / L, dy / L
        x1b, y1b = x1 + ux * 26, y1 + uy * 26
        x2b, y2b = x2 - ux * 30, y2 - uy * 30
        mx, my = (x1b + x2b) / 2, (y1b + y2b) / 2
        svg.append(f'<line x1="{x1b:.0f}" y1="{y1b:.0f}" x2="{x2b:.0f}" y2="{y2b:.0f}" '
                   f'stroke="#9aa" stroke-width="1.3" marker-end="url(#ah)"/>')
        svg.append(f'<text x="{mx:.0f}" y="{my-3:.0f}" font-size="10.5" fill="#8a8a8a" '
                   f'text-anchor="middle">{esc(rel)}</text>')
    # nodes
    for nid, (x, y) in pos.items():
        lab, cap = byid[nid][1], byid[nid][2]
        col = LABEL_COLOR.get(lab, "#666")
        r = 26 if nid == center[0] else 22
        svg.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r}" fill="{col}" '
                   f'fill-opacity="0.16" stroke="{col}" stroke-width="2"/>')
        svg.append(f'<text x="{x:.0f}" y="{y-r-6:.0f}" font-size="11.5" font-weight="600" '
                   f'fill="{col}" text-anchor="middle">{esc(lab)}</text>')
        svg.append(f'<text x="{x:.0f}" y="{y+4:.0f}" font-size="10" fill="var(--fg)" '
                   f'text-anchor="middle">{esc(cap)}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


# --------------------------------------------------------------------------- #
#  Queries                                                                     #
# --------------------------------------------------------------------------- #
def schema_page(sess):
    labs = {r["l"]: r["c"] for r in sess.run(
        "MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c ORDER BY c DESC")
        if r["l"] not in EXCLUDE_LABELS}
    rels = {r["t"]: r["c"] for r in sess.run(
        "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC")
        if r["t"] not in EXCLUDE_RELS}
    triples = [(r["src"], r["rel"], r["dst"], r["c"]) for r in sess.run(
        "MATCH (a)-[r]->(b) WITH labels(a)[0] AS src, type(r) AS rel, labels(b)[0] AS dst, "
        "count(*) AS c RETURN src, rel, dst, c ORDER BY c DESC")
        if r["src"] not in EXCLUDE_LABELS and r["dst"] not in EXCLUDE_LABELS
        and r["rel"] not in EXCLUDE_RELS][:24]

    def tier_of(lab): return LABEL_TIER.get(lab, "&mdash;")
    lab_rows = "".join(
        f'<tr><td><span class="dot" style="background:{LABEL_COLOR.get(l,"#666")}"></span> '
        f'<code>{html.escape(l)}</code></td><td>{tier_of(l)}</td>'
        f'<td class="n">{c:,}</td></tr>' for l, c in labs.items())
    rel_rows = "".join(
        f'<tr><td><code>{html.escape(t)}</code></td><td class="n">{c:,}</td></tr>'
        for t, c in rels.items())
    tri_rows = "".join(
        f'<tr><td><code>{html.escape(s)}</code></td><td><code>{html.escape(r)}</code></td>'
        f'<td><code>{html.escape(d)}</code></td><td class="n">{c:,}</td></tr>'
        for s, r, d, c in triples)

    # schema meta-diagram: one representative node per tier + defining edges
    meta_nodes = [("Commit", "Commit", "process node"),
                  ("File", "File", "source file"),
                  ("Developer", "Developer", "author"),
                  ("Issue", "Issue", "tracker issue"),
                  ("ASTNode", "ASTNode", "AST change"),
                  ("CFGNode", "CFGNode", "CFG/DFG/PDG/SEQ"),
                  ("Term", "Term", "vocabulary")]
    meta_edges = [("Commit", "File", "MODIFIED"), ("Commit", "Developer", "AUTHORED_BY"),
                  ("Commit", "Issue", "FIXES_ISSUE"), ("Commit", "ASTNode", "ADDS/UPDATES/…"),
                  ("Commit", "CFGNode", "ADDS/…"), ("Commit", "Term", "MENTIONS"),
                  ("Term", "File", "REFERS_TO"), ("File", "CFGNode", "HAS_CFG/…")]
    diagram = _svg_subgraph(("Commit", "Commit", "process node"), meta_nodes, meta_edges)

    body = f"""<h1>Appendix L&mdash;Knowledge-Graph Schema</h1>
<p class="sub">Meta-schema of the deployed three-tier KG-Commit graph, with live node
and relationship counts for the resident project. Method-scoped AST
(<code>ASTMethodNode</code>) is excluded, per the final design.</p>
{_legend()}
<h2>L.1&nbsp;&nbsp;Node-label inventory</h2>
<table><tr><th>Label</th><th>Tier</th><th>Count</th></tr>{lab_rows}</table>
<h2>L.2&nbsp;&nbsp;Relationship inventory</h2>
<table><tr><th>Relationship type</th><th>Count</th></tr>{rel_rows}</table>
<h2>L.3&nbsp;&nbsp;Meta-schema triples&nbsp;<span class="sub" style="font-size:12px">(top by frequency)</span></h2>
<table><tr><th>Source</th><th>Relationship</th><th>Target</th><th>Count</th></tr>{tri_rows}</table>
<h2>L.4&nbsp;&nbsp;Schema diagram</h2>
<div class="fig">{diagram}
<p class="cap">Figure L.1&nbsp;The <b>Commit</b> node is the hub connecting the Core
process tier (File, Developer, Issue) to the AST structural-change tier, the
per-method subgraph tier (CFG/DFG/PDG/SEQ), and the CSTG semantic-text tier (Term).</p></div>"""
    (OUT / "appL_schema.html").write_text(_page("Appendix L — KG Schema", body), encoding="utf-8")
    return list(labs), list(rels)


def _fmt_cypher(q):
    kw = ["MATCH", "WHERE", "RETURN", "LIMIT", "OPTIONAL", "WITH", "AS", "ORDER BY"]
    out = html.escape(q)
    for k in kw:
        out = out.replace(k, f'<span class="k">{k}</span>')
    return out


def export_page(sess, fname, title, intro, cypher, center, nodes, edges, capt):
    diagram = _svg_subgraph(center, nodes, edges)
    body = f"""<h1>{html.escape(title)}</h1>
<p class="sub">{intro}</p>
{_legend()}
<h2>Cypher query</h2>
<pre class="cy">{_fmt_cypher(cypher)}</pre>
<h2>Returned subgraph</h2>
<div class="fig">{diagram}<p class="cap">{capt}</p></div>"""
    (OUT / fname).write_text(_page(title, body), encoding="utf-8")


def core_export(sess, cid):
    q = ("MATCH (c:Commit {id:$id})\n"
         "OPTIONAL MATCH (c)-[:MODIFIED]->(f:File)\n"
         "OPTIONAL MATCH (c)-[:AUTHORED_BY]->(d:Developer)\n"
         "OPTIONAL MATCH (c)-[:FIXES_ISSUE]->(i:Issue)\n"
         "OPTIONAL MATCH (p:Commit)-[:PARENT_OF]->(c)\n"
         "RETURN c, f, d, i, p LIMIT 25")
    center = (cid, "Commit", cid[:8])
    nodes, edges, seen = [center], [], {cid}
    for rec in sess.run(q, id=cid):
        for key, lab, capfn, rel, out in [
            ("f", "File", lambda x: (x.get("path") or x.get("name") or "file").split("/")[-1], "MODIFIED", True),
            ("d", "Developer", lambda x: x.get("name") or x.get("email") or "dev", "AUTHORED_BY", True),
            ("i", "Issue", lambda x: x.get("key") or x.get("id") or "issue", "FIXES_ISSUE", True),
            ("p", "Commit", lambda x: (x.get("id") or "")[:8], "PARENT_OF", False)]:
            n = rec.get(key)
            if n is None:
                continue
            nid = n.element_id
            if nid not in seen:
                seen.add(nid)
                nodes.append((nid, lab, capfn(n)))
            edges.append((cid, nid, rel) if out else (nid, cid, rel))
    export_page(sess, "appL_export_core.html", "Appendix L&mdash;Core Process Tier Export",
                f"The Core process-tier neighbourhood of one labelled commit "
                f"(<code>{html.escape(cid[:12])}</code>): its modified files, author, "
                f"fixed issue, and parent commit.", q, center, nodes, edges,
                "Figure L.2&nbsp;Core tier: a commit links to files, developer, issue, and its parent.")


def ast_export(sess, cid):
    q = ("MATCH (c:Commit {id:$id})-[r:ADDS|UPDATES|REMOVES|MOVES]->(a:ASTNode)\n"
         "RETURN type(r) AS rel, a LIMIT 8")
    center = (cid, "Commit", cid[:8])
    nodes, edges, seen = [center], [], {cid}
    for rec in sess.run(q, id=cid):
        a = rec["a"]; nid = a.element_id
        cap = (a.get("type") or a.get("node_type") or a.get("label") or "ASTNode")
        if nid not in seen:
            seen.add(nid); nodes.append((nid, "ASTNode", str(cap)[:14]))
        edges.append((cid, nid, rec["rel"]))
    export_page(sess, "appL_export_ast.html", "Appendix L&mdash;AST Structural-Change Tier Export",
                f"The AST structural-change edges emitted by commit "
                f"<code>{html.escape(cid[:12])}</code> (ADDS / UPDATES / REMOVES / MOVES "
                f"to individual AST nodes).", q, center, nodes, edges,
                "Figure L.3&nbsp;AST tier: typed change edges from the commit to affected AST nodes.")


def cstg_export(sess, cid):
    # Distinct mentioned terms (up to 6), then the COOCCURS edges AMONG those terms
    # (the interesting CSTG structure), plus any REFERS_TO file when present.
    # Terms are stored one node per mention, so dedup by term TEXT: keep one
    # representative Term node per distinct text (up to 6), then COOCCURS among them.
    # dedup terms by text (one node per mention in the store); keep <=6, and at most
    # ONE representative REFERS_TO file per term (terms can refer to hundreds of files).
    q = ("MATCH (c:Commit {id:$id})-[:MENTIONS]->(t:Term)\n"
         "WITH t.text AS txt, head(collect(t)) AS t\n"
         "WITH collect(t)[..6] AS ts\n"
         "UNWIND ts AS t\n"
         "OPTIONAL MATCH (t)-[:REFERS_TO]->(f:File)\n"
         "WITH ts, t, head(collect(f)) AS f\n"
         "OPTIONAL MATCH (t)-[:COOCCURS]-(t2:Term) WHERE t2 IN ts\n"
         "RETURN t, f, collect(DISTINCT t2) AS co")
    center = (cid, "Commit", cid[:8])
    nodes, edges, seen = [center], [], {cid}
    termids = {}

    def term_cap(t):
        return str(t.get("text") or t.get("term") or t.get("name") or "term")[:14]
    recs = list(sess.run(q, id=cid))
    for rec in recs:
        t = rec["t"]
        if t is None:
            continue
        tid = t.element_id
        termids[t.get("text") or tid] = tid
        if tid not in seen:
            seen.add(tid); nodes.append((tid, "Term", term_cap(t)))
        edges.append((cid, tid, "MENTIONS"))
        f = rec.get("f")
        if f is not None:
            fid = f.element_id
            fcap = (f.get("path") or f.get("name") or "file").split("/")[-1]
            if fid not in seen:
                seen.add(fid); nodes.append((fid, "File", str(fcap)[:14]))
            edges.append((tid, fid, "REFERS_TO"))
    # add COOCCURS edges among the drawn terms (dedup undirected)
    coset = set()
    for rec in recs:
        t = rec["t"]
        if t is None:
            continue
        tid = t.element_id
        for t2 in rec.get("co") or []:
            t2id = t2.element_id
            if t2id in seen and tid != t2id:
                key = tuple(sorted((tid, t2id)))
                if key not in coset:
                    coset.add(key); edges.append((tid, t2id, "COOCCURS"))
    export_page(sess, "appL_export_cstg.html", "Appendix L&mdash;CSTG Semantic-Text Tier Export",
                f"The CSTG semantic-text neighbourhood of commit "
                f"<code>{html.escape(cid[:12])}</code>: the vocabulary terms it mentions, "
                f"their co-occurrence links, and any files they refer to.", q, center, nodes, edges,
                "Figure L.4&nbsp;CSTG tier: commit&rarr;Term (MENTIONS), Term&mdash;Term "
                "(COOCCURS), Term&rarr;File (REFERS_TO).")


def pick_commit(sess):
    rec = sess.run(
        "MATCH (c:Commit {in_jit:true}) WHERE c.buggy=true "
        "WITH c, size([(c)-[:ADDS|UPDATES|REMOVES|MOVES]->(:ASTNode) | 1]) AS ast, "
        "size([(c)-[:MENTIONS]->(:Term) | 1]) AS tm "
        "WHERE ast>=3 AND tm>=4 RETURN c.id AS id ORDER BY ast LIMIT 1").single()
    if rec:
        return rec["id"]
    rec = sess.run("MATCH (c:Commit {in_jit:true}) RETURN c.id AS id LIMIT 1").single()
    return rec["id"] if rec else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default=None, help="commit id for the example exports")
    args = ap.parse_args()
    drv = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with drv.session() as sess:
        schema_page(sess)
        cid = args.commit or pick_commit(sess)
        if cid is None:
            print("No commit found; schema page written, exports skipped.")
            return
        core_export(sess, cid)
        ast_export(sess, cid)
        cstg_export(sess, cid)
    drv.close()
    n = len(list(OUT.glob("*.html")))
    print(f"[{PROJECT}] Appendix L: wrote {n} HTML pages -> {OUT}  (example commit {cid[:8]})")


if __name__ == "__main__":
    main()
