"""
RQ3 comprehensive matrix (Table 10).

Layout requested for the paper:

  supersupercolumns : Layer 1 | Layer 2 (Subgraph Ablation) | Layer 3
  supercolumns      : Core    | AST  CFG  DFG  PDG          | CSTG
  columns           : Macro-F1, G-Mean, AUC
  superrows         : the 7 active projects, then Macro-Avg and Micro-Avg
  rows              : RN, PPR, LP, DW, KGE, and F (the project's chosen fusion)

Every cell is re-scored from the cached per-commit method scores at the verification
gap G = 50, on the warm-up window the fusion was selected under (stored in meta["W"]).
The F row uses each project's own selected fusion, and its label names that fusion,
e.g. "F (RN+PPR)".

The window matters: F is a prequential LR stack, so re-deriving a shorter warm-up
starves its head while the single methods, which have none, are barely affected. Doing
so drove F below its own members on 6 of 7 projects -- a window artefact, not a result.
F is selected on the deployed CSTG graph only; on the ablated columns the same fixed
combination is reported without re-optimisation, so a single method may beat it there.

Cache-only: reads raw_method_scores.pkl and final_fusion_results.pkl. No Neo4j.

Run:  python inference/make_rq3_big_matrix.py
Out:  Paper/paper_material/RQ3_subgraphs/table10_big_matrix.tex
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _kgc_paths  # noqa: E402,F401
from paper_projects import ACTIVE as PROJECTS  # noqa: E402
from online_jit import final_metrics  # noqa: E402
import run_final_fusion as rff  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUTP = ROOT / "outputs"
PM = ROOT / "Paper" / "paper_material" / "RQ3_subgraphs"

WARMUP = 0.20
GAP = 50
INIT = 300

GRAPHS = [("core", "Core"), ("ast", "AST"), ("cfg", "CFG"),
          ("dfg", "DFG"), ("pdg", "PDG"), ("final", "CSTG")]
METHODS = ["RN", "PPR", "LP", "DW", "KGE"]
METS = ["Macro_F1", "G_Mean", "AUC"]
MET_HDR = [r"$F_1$", "G", "AUC"]


def project_cells(folder):
    """{graph: {row_label: {metric: value}}} for one project, at (WARMUP, GAP)."""
    rp = OUTP / folder / "raw_method_scores.pkl"
    fp = OUTP / folder / "final_fusion_results.pkl"
    if not (rp.exists() and fp.exists()):
        return None, None
    raw = pickle.load(open(rp, "rb"))
    fus = pickle.load(open(fp, "rb"))
    F = fus["chosen"].split("+")

    y = np.asarray(raw["y"], int)
    N = int(raw["N"])
    # Use the warm-up the fusion was SELECTED under, not a re-derived one. F is a
    # prequential LR stack: shrinking the warm-up starves its head while leaving the
    # single methods (which have no head) almost untouched, so a re-derived window
    # made F score below its own members -- an artefact of the window, not a result.
    W = int(fus.get("meta", {}).get("W") or N * WARMUP)
    ev0 = W + INIT
    if ev0 >= N - 10:
        return None, None
    ev = np.arange(ev0, N)

    out = {}
    for g, _ in GRAPHS:
        if g not in raw["scores"]:
            continue
        col = {}
        scores = {m: np.asarray(raw["scores"][g][m], float) for m in METHODS}
        for m in METHODS:
            p = np.clip(np.nan_to_num(scores[m][ev], nan=float(y[ev].mean())), 0, 1)
            r = final_metrics(y[ev], p, gap=GAP)
            col[m] = {k: float(r.get(k, np.nan)) for k in METS}
        # the project's chosen fusion, evaluated on THIS graph
        mm, _, _, _ = rff.eval_subset(scores, F, y, W, N, init=INIT, gap=GAP)
        col["F"] = {k: float(mm.get(k, np.nan)) for k in METS}
        out[g] = col
    return out, fus["chosen"]


def _fmt(v):
    if v is None or v != v:
        return "--"
    return f"{v:.3f}".lstrip("0")          # .634 style, saves width


def build():
    per_proj, chosen, weights = {}, {}, {}
    for disp, folder in PROJECTS:
        cells, ch = project_cells(folder)
        if cells:
            per_proj[disp] = cells
            chosen[disp] = ch
            bx = OUTP / folder / "baseline_extra_results.pkl"
            weights[disp] = (pickle.load(open(bx, "rb"))["n_eval"]
                             if bx.exists() else 1)
    if not per_proj:
        return None

    rows = METHODS + ["F"]
    ngraph = len(GRAPHS)
    ncols = 2 + ngraph * len(METS)

    L = [r"\begin{table*}[p!]", r"\centering",
         r"\caption{RQ3: cumulative predictive performance across the seven active "
         r"projects under the real-time online protocol, for five inference methods and "
         r"their per-project fusion $F$. Columns follow the hierarchical construction of "
         r"KG-Commit: the Core process layer (Layer~1), each candidate structural "
         r"subgraph attached to it (Layer~2), and the full deployed graph with the "
         r"semantic tier (Layer~3). The $F$ row of each project names that project's "
         r"selected fusion. Reading down the AST column against CFG/DFG/PDG isolates the "
         r"effect of the structural choice; reading across from Core to CSTG isolates the "
         r"marginal contribution of each layer. Each fusion is selected once, on the "
         r"deployed CSTG graph, and the same combination is then reported unchanged on "
         r"every column; on the ablated columns it is therefore not re-optimised, and a "
         r"single method may exceed it there.}",
         r"\label{tab:massive_evaluation_matrix}",
         r"\tiny", r"\setlength{\tabcolsep}{2.6pt}",
         r"\renewcommand{\arraystretch}{0.92}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{ll " + " ".join(["ccc"] * ngraph) + "}",
         r"\toprule"]

    # supersupercolumn header
    h1 = [r"\textbf{Dataset}", r"\textbf{Inf.}",
          r"\multicolumn{3}{c}{\textbf{Layer 1}}",
          r"\multicolumn{12}{c}{\textbf{Layer 2 (Subgraph Ablation)}}",
          r"\multicolumn{3}{c}{\textbf{Layer 3}}"]
    L.append(" & ".join(h1) + r" \\")
    L.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-17} \cmidrule(lr){18-20}")
    # supercolumn header
    h2 = ["", ""] + [r"\multicolumn{3}{c}{%s}" % lab for _, lab in GRAPHS]
    L.append(" & ".join(h2) + r" \\")
    L.append(" ".join(r"\cmidrule(lr){%d-%d}" % (3 + 3 * i, 5 + 3 * i)
                      for i in range(ngraph)))
    # metric header
    L.append(" & ".join(["", ""] + MET_HDR * ngraph) + r" \\")
    L.append(r"\midrule")

    def emit_block(label, cells, fusion_name):
        # Exactly one bold per metric per project block: the single best cell over all
        # (graph, method) pairs. Ties at the printed precision are broken by the exact
        # value, then by column then row order, so the block never shows two winners.
        win = {}
        for m in METS:
            cand = []
            for gi, (g, _) in enumerate(GRAPHS):
                for ri, r in enumerate(rows):
                    v = cells.get(g, {}).get(r, {}).get(m)
                    if v is not None and v == v:
                        cand.append((v, -gi, -ri, g, r))
            if cand:
                _, _, _, g, r = max(cand)
                win[m] = (g, r)
        first = True
        for r in rows:
            rlab = (rf"F ({fusion_name})" if r == "F" else r)
            lead = (rf"\multirow{{6}}{{*}}{{\texttt{{{label}}}}}" if first else "")
            first = False
            vals = []
            for g, _ in GRAPHS:
                c = cells.get(g, {}).get(r, {})
                for m in METS:
                    v = c.get(m)
                    s = _fmt(v)
                    if v is not None and v == v and win.get(m) == (g, r):
                        s = rf"\textbf{{{s}}}"
                    vals.append(s)
            if r == "F":
                rlab = rf"\textbf{{{rlab}}}"
            L.append(" & ".join([lead, rlab] + vals) + r" \\")
        L.append(r"\midrule")

    for disp, _ in PROJECTS:
        if disp in per_proj:
            emit_block(disp, per_proj[disp], chosen[disp])

    # aggregates
    for agg_label, use_w in [("Macro-Avg", False), ("Micro-Avg", True)]:
        agg = {}
        for g, _ in GRAPHS:
            agg[g] = {}
            for r in rows:
                acc, ws = [], []
                for disp in per_proj:
                    c = per_proj[disp].get(g, {}).get(r)
                    if c and all(c[m] == c[m] for m in METS):
                        acc.append([c[m] for m in METS])
                        ws.append(weights[disp] if use_w else 1.0)
                if acc:
                    A = np.array(acc); W_ = np.array(ws, float)
                    agg[g][r] = {m: float(np.average(A[:, j], weights=W_))
                                 for j, m in enumerate(METS)}
        emit_block(agg_label, agg, "per project")

    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"}", r"\end{table*}"]
    return "\n".join(L)


def main():
    PM.mkdir(parents=True, exist_ok=True)
    s = build()
    if not s:
        print("no data")
        return
    (PM / "table10_big_matrix.tex").write_text(s + "\n", encoding="utf-8")
    print(f"  wrote {PM / 'table10_big_matrix.tex'}")


if __name__ == "__main__":
    main()
