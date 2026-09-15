#!/usr/bin/env python3
"""
Wave 2 --- solution families the pilot never tested.
====================================================

The 95 arms so far explore ONE idea in many variants: summarise a bounded
neighbourhood of the file-dependency graph, then stack. Six families were never
touched at all. Each is CPU-only, inference-level, and reaches project context
outside the commit.

  W1  HETEROGENEOUS CO-CHANGE / OWNERSHIP GRAPH
      The dependency graph is not the only global structure. Developers connect
      files that never import each other; issues connect commits across the
      repository. Risk propagated over Commit-Developer-Commit and
      Commit-Issue-Commit reaches context that IMPORTS cannot express.

  W2  ASYMMETRIC / DIRECTED DIFFUSION
      All diffusion so far symmetrised the graph. Defect risk is not symmetric:
      breaking a widely-imported file endangers its dependants, not its
      dependencies. Forward-only and backward-only diffusion, separately.

  W3  TEMPORAL DECAY
      Every history statistic weighted all past equally. A defect 5 years ago
      says less than one last month. Exponentially-decayed defect density over
      each tier.

  W4  RELATIVE / NORMALISED CONTEXT
      Absolute tier statistics conflate "risky region" with "large region".
      Normalising each commit's context against the project-wide distribution
      at that time makes the feature a z-score, comparable across projects --
      which is what generalisation needs.

  W5  STRUCTURAL HOLES / BROKERAGE
      A commit touching files that bridge otherwise-disconnected parts of the
      dependency graph is architecturally risky. Burt's constraint, computed on
      the induced neighbourhood.

  W6  INTERACTION TERMS
      Global context may matter CONDITIONALLY: a large change in a fragile
      region is worse than either alone. Explicit products of the strongest
      context features with change size.

Out: outputs/<project>/global_context/wave2_results.csv
Run: KGC_PROJECT=kafka python inference/run_wave2_arms.py
"""
from __future__ import annotations

import csv
import pickle
import sys
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import _kgc_paths  # noqa: E402,F401
from config.project_config import OUT, PROJECT, NEO4J_URI, NEO4J_AUTH  # noqa: E402
from online_jit import final_metrics  # noqa: E402
from protocol import BLOCK  # noqa: E402
from run_tgc_pilot import build_features, omega_ids  # noqa: E402
from kg_topology_methods import build_file_graph  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

DEST = OUT / "global_context"
ROOT = HERE.parent.parent
N_SNAP = 8
CTX, ROWS = {}, []


def q(cypher):
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with drv.session(default_access_mode="READ") as s:
        out = s.execute_read(lambda tx: [r.data() for r in tx.run(cypher)])
    drv.close()
    return out


def stack(cols, y, W, C=0.2, penalty="l1"):
    Z = np.nan_to_num(np.column_stack(cols), nan=0.0, posinf=0.0, neginf=0.0)
    N = len(y)
    p = np.full(N, np.nan)
    i = W
    while i < N:
        j = min(N, i + BLOCK)
        past = np.arange(i)
        if len(np.unique(y[past])) < 2:
            p[i:j] = float(y[past].mean()) if len(past) else 0.5
            i = j
            continue
        sc = StandardScaler().fit(Z[past])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=C,
                                 penalty=penalty,
                                 solver="liblinear" if penalty == "l1" else "lbfgs")
        clf.fit(sc.transform(Z[past]), y[past])
        p[i:j] = clf.predict_proba(sc.transform(Z[i:j]))[:, 1]
        i = j
    return np.nan_to_num(p[np.arange(W, N)], nan=float(y[np.arange(W, N)].mean()))


def boot(y, pa, pb, metric="Macro_F1", n=250, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        if len(np.unique(y[s])) < 2:
            continue
        a = final_metrics(y[s], np.clip(pa[s], 0, 1)).get(metric)
        b = final_metrics(y[s], np.clip(pb[s], 0, 1)).get(metric)
        if a is not None and b is not None:
            d.append(float(a) - float(b))
    if len(d) < 30:
        return None
    d = np.asarray(d)
    return {"diff": float(d.mean()), "p": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


def record(stage, label, p, nfeat=""):
    c = CTX
    yc, mask, base_p = c["yc"], c["mask"], c["base_p"]
    m = final_metrics(yc, np.clip(p, 0, 1))
    mo = final_metrics(yc[mask], np.clip(p[mask], 0, 1))
    sa = boot(yc, p, base_p)
    so = boot(yc[mask], p[mask], base_p[mask])
    both = (m["Macro_F1"] > c["bm"]["Macro_F1"]) and (mo["Macro_F1"] > c["bo"]["Macro_F1"])
    ROWS.append([stage, label, nfeat,
                 round(float(m["Macro_F1"]), 4), round(float(m["MCC"]), 4),
                 round(float(m["AUC"]), 4), round(float(mo["Macro_F1"]), 4),
                 round(float(mo["MCC"]), 4),
                 round(sa["diff"], 4) if sa else "", round(sa["p"], 4) if sa else "",
                 round(so["diff"], 4) if so else "", round(so["p"], 4) if so else "",
                 "YES" if both else ""])
    print(f"  [{stage}] {label:<34} ALL F1={m['Macro_F1']:.4f} MCC={m['MCC']:.4f} | "
          f"OM F1={mo['Macro_F1']:.4f} MCC={mo['MCC']:.4f}"
          f"{'  <<< BOTH UP' if both else ''}", flush=True)


def flush():
    with (DEST / "wave2_results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "arm", "n_features", "all_MacroF1", "all_MCC",
                    "all_AUC", "omega_MacroF1", "omega_MCC", "d_all", "p_all",
                    "d_omega", "p_omega", "both_up"])
        w.writerows(ROWS)


# ------------------------------------------------------------------- W1
def stage_W1():
    """Ownership / issue co-change graphs -- context IMPORTS cannot express."""
    dev = q("MATCH (c:Commit {in_jit:true})-[:AUTHORED_BY]->(d:Developer) "
            "RETURN c.id AS cid, d.id AS did")
    iss = q("MATCH (c:Commit {in_jit:true})-[:FIXES_ISSUE]->(i:Issue) "
            "RETURN c.id AS cid, i.id AS iid")
    order = CTX["order_ids"]
    cpos = {c: k for k, c in enumerate(order)}
    ybuggy = np.array([c["buggy"] for c in CTX["blob"]["commits"]], float)

    dmap = {r["cid"]: r["did"] for r in dev}
    imap_ = defaultdict(list)
    for r in iss:
        imap_[r["cid"]].append(r["iid"])
    print(f"     devs={len(set(dmap.values()))} issue-links={len(iss)}", flush=True)

    n = len(order)
    F = np.zeros((n, 4))
    dstat = defaultdict(lambda: [0.0, 0.0])     # dev -> [changes, buggy]
    istat = defaultdict(lambda: [0.0, 0.0])
    for k, cid in enumerate(order):
        d = dmap.get(cid)
        if d and dstat[d][0] > 0:
            F[k, 0] = dstat[d][1] / dstat[d][0]
            F[k, 1] = np.log1p(dstat[d][0])
        vals = [istat[i][1] / istat[i][0] for i in imap_.get(cid, ()) if istat[i][0] > 0]
        if vals:
            F[k, 2] = float(np.mean(vals))
            F[k, 3] = float(np.max(vals))
        # fold in AFTER scoring
        b = ybuggy[k]
        if d:
            dstat[d][0] += 1
            dstat[d][1] += b
        for i in imap_.get(cid, ()):
            istat[i][0] += 1
            istat[i][1] += b
    W1 = F[np.array([cpos[x] for x in CTX["fids"]])]
    CTX["W1"] = W1
    record("W1", "F+G + ownership/issue", stack([CTX["dep"], W1], CTX["y"], CTX["W"]), 4)
    record("W1", "TGC+M1 + ownership/issue",
           stack([CTX["dep"], CTX["Xtgc"], CTX["M1"], W1], CTX["y"], CTX["W"]),
           CTX["Xtgc"].shape[1] + CTX["M1"].shape[1] + 4)


# ------------------------------------------------------------------- W2
def stage_W2():
    """Directed diffusion: risk flows to dependants, not from them."""
    blob, order, fids = CTX["blob"], CTX["order_ids"], CTX["fids"]
    tiers = blob["tiers"]
    files = CTX["all_files"]
    fmap = {f: i for i, f in enumerate(files)}
    n = len(order)
    F = np.zeros((n, 4))
    dep_e, dry_e, seen = [], [], set()
    risk = np.zeros(len(files))
    bounds = [int(n * k / N_SNAP) for k in range(N_SNAP + 1)]
    for s in range(N_SNAP):
        lo, hi = bounds[s], bounds[s + 1]
        props = {}
        for name, es in (("fwd", dep_e), ("bwd", dry_e)):
            if es:
                A, _ = build_file_graph(es, files)
                d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1.0
                P = sp.diags(1.0 / d) @ A          # DIRECTED, not symmetrised
                x = risk.copy()
                for _ in range(8):
                    x = 0.15 * risk + 0.85 * (P.T @ x)
                props[name] = x
        for k in range(lo, hi):
            t = tiers.get(order[k])
            ids = [fmap[f] for f in (t["T0"] if t else []) if f in fmap]
            if ids and props:
                fw = props.get("fwd")
                bw = props.get("bwd")
                F[k] = [float(np.log1p(fw[ids].mean())) if fw is not None else 0.0,
                        float(np.log1p(fw[ids].max())) if fw is not None else 0.0,
                        float(np.log1p(bw[ids].mean())) if bw is not None else 0.0,
                        float(np.log1p(bw[ids].max())) if bw is not None else 0.0]
        for k in range(lo, hi):
            t = tiers.get(order[k])
            if not t:
                continue
            for f in t["T0"]:
                for g in t.get("T1_dep", ()):
                    if (g, f) not in seen:
                        seen.add((g, f)); dep_e.append((g, f))
                for g in t.get("T1_dry", ()):
                    if (f, g) not in seen:
                        seen.add((f, g)); dry_e.append((f, g))
            if blob["commits"][k]["buggy"]:
                for f in t["T0"]:
                    if f in fmap:
                        risk[fmap[f]] += 1.0
    W2 = F[np.array([{c: k for k, c in enumerate(order)}[x] for x in fids])]
    CTX["W2"] = W2
    record("W2", "F+G + directed diffusion", stack([CTX["dep"], W2], CTX["y"], CTX["W"]), 4)
    record("W2", "TGC+M1 + directed diffusion",
           stack([CTX["dep"], CTX["Xtgc"], CTX["M1"], W2], CTX["y"], CTX["W"]),
           CTX["Xtgc"].shape[1] + CTX["M1"].shape[1] + 4)


# ------------------------------------------------------------------- W3/W4/W5/W6
def stage_W3():
    """Temporal decay: recent defects in the context weigh more."""
    blob, order, fids = CTX["blob"], CTX["order_ids"], CTX["fids"]
    tiers = blob["tiers"]
    n = len(order)
    F = np.zeros((n, 6))
    for hl_i, HL in enumerate((50, 250)):
        lam = np.log(2) / HL
        dec = defaultdict(float)     # file -> decayed defect mass
        last = defaultdict(int)
        for k, cid in enumerate(order):
            t = tiers.get(cid)
            if t:
                for j, key in enumerate(("T0", "T1_dep", "T1_dry")):
                    fs = t.get(key, ())
                    if fs:
                        v = 0.0
                        for f in fs:
                            if dec[f]:
                                v += dec[f] * np.exp(-lam * (k - last[f]))
                        F[k, hl_i * 3 + j] = np.log1p(v / len(fs))
            if t and blob["commits"][k]["buggy"]:
                for f in t["T0"]:
                    if dec[f]:
                        dec[f] *= np.exp(-lam * (k - last[f]))
                    dec[f] += 1.0
                    last[f] = k
    W3 = F[np.array([{c: k for k, c in enumerate(order)}[x] for x in fids])]
    CTX["W3"] = W3
    record("W3", "F+G + temporal decay", stack([CTX["dep"], W3], CTX["y"], CTX["W"]), 6)
    record("W3", "TGC+M1 + temporal decay",
           stack([CTX["dep"], CTX["Xtgc"], CTX["M1"], W3], CTX["y"], CTX["W"]),
           CTX["Xtgc"].shape[1] + CTX["M1"].shape[1] + 6)


def stage_W4():
    """Relative context: z-score each feature against the project-so-far."""
    X = CTX["Xtgc"]
    Z = np.zeros_like(X)
    mu = np.zeros(X.shape[1]); s2 = np.ones(X.shape[1]); cnt = 0
    for i in range(len(X)):
        if cnt > 30:
            Z[i] = (X[i] - mu) / np.sqrt(np.maximum(s2, 1e-9))
        cnt += 1
        d = X[i] - mu
        mu = mu + d / cnt
        s2 = s2 + (d * (X[i] - mu) - s2) / cnt
    CTX["W4"] = Z
    record("W4", "F+G + relative(z) context", stack([CTX["dep"], Z], CTX["y"], CTX["W"]),
           Z.shape[1])
    record("W4", "TGC+M1 + relative(z)",
           stack([CTX["dep"], CTX["Xtgc"], CTX["M1"], Z], CTX["y"], CTX["W"]),
           CTX["Xtgc"].shape[1] + CTX["M1"].shape[1] + Z.shape[1])


def stage_W5():
    """Structural holes: does the change bridge disconnected regions?"""
    blob, order, fids = CTX["blob"], CTX["order_ids"], CTX["fids"]
    tiers = blob["tiers"]
    files = CTX["all_files"]
    n = len(order)
    F = np.zeros((n, 3))
    edges, seen = [], set()
    bounds = [int(n * k / N_SNAP) for k in range(N_SNAP + 1)]
    for s in range(N_SNAP):
        lo, hi = bounds[s], bounds[s + 1]
        if edges:
            A, idx = build_file_graph(edges, files)
            S = ((A + A.T) > 0).astype(float).tocsr()
            deg = np.asarray(S.sum(1)).ravel()
            tri = np.asarray((S @ S).multiply(S).sum(1)).ravel()
            # Burt constraint proxy: low clustering + high degree = brokerage
            clus = tri / np.maximum(deg * np.maximum(deg - 1, 1), 1)
            broker = deg * (1.0 - clus)
        else:
            idx = {f: i for i, f in enumerate(files)}
            deg = clus = broker = np.zeros(len(files))
        for k in range(lo, hi):
            t = tiers.get(order[k])
            ids = [idx[f] for f in (t["T0"] if t else []) if f in idx]
            if ids:
                F[k] = [float(np.log1p(broker[ids].mean())),
                        float(np.log1p(broker[ids].max())),
                        float(1.0 - clus[ids].mean())]
            if t:
                for f in t["T0"]:
                    for g in t.get("T1_dep", ()):
                        if (g, f) not in seen:
                            seen.add((g, f)); edges.append((g, f))
                    for g in t.get("T1_dry", ()):
                        if (f, g) not in seen:
                            seen.add((f, g)); edges.append((f, g))
    W5 = F[np.array([{c: k for k, c in enumerate(order)}[x] for x in fids])]
    CTX["W5"] = W5
    record("W5", "F+G + structural holes", stack([CTX["dep"], W5], CTX["y"], CTX["W"]), 3)
    record("W5", "TGC+M1 + structural holes",
           stack([CTX["dep"], CTX["Xtgc"], CTX["M1"], W5], CTX["y"], CTX["W"]),
           CTX["Xtgc"].shape[1] + CTX["M1"].shape[1] + 3)


def stage_W6():
    """Interaction: context matters conditionally on change size."""
    X, M1 = CTX["Xtgc"], CTX["M1"]
    size = X[:, 0:1]                       # log tier size of T0
    ctx = X[:, [1, 2]]                     # T0 defect density / max
    inter = np.hstack([ctx * size, M1[:, :2] * size])
    CTX["W6"] = inter
    record("W6", "TGC+M1 + interactions",
           stack([CTX["dep"], X, M1, inter], CTX["y"], CTX["W"]),
           X.shape[1] + M1.shape[1] + inter.shape[1])


def main():
    base = ROOT / "outputs" / PROJECT
    blob = pickle.load(open(DEST / "tiers.pkl", "rb"))
    fu = pickle.load(open(base / "raw_fusion_scores.pkl", "rb"))
    mp = pd.read_csv(base / "raw_method_scores.csv",
                     usecols=["commit_index", "commit_id"]).drop_duplicates("commit_index")
    imap = dict(zip(mp.commit_index.astype(int), mp.commit_id.astype(str)))
    om = omega_ids()

    fidx = np.asarray(fu["commit_index"], int)
    y = np.asarray(fu["y"], int)
    dep = np.asarray(fu["scores"]["F+G"], float)
    fids = [imap.get(int(i), "") for i in fidx]
    W = max(int(len(y) * 0.2), 50)
    ev = np.arange(W, len(y))
    yc = y[ev]
    mask = np.array([fids[i] in om for i in ev])
    base_p = dep[ev]

    X, _, ids_all = build_features(blob, ["T0", "T1_dep", "T1_dry"])
    p2 = {c: k for k, c in enumerate(ids_all)}
    Xtgc = X[np.array([p2[c] for c in fids])]

    from run_kg_topology import snapshot_features
    order_ids = [c["id"] for c in blob["commits"]]
    T = snapshot_features(blob, order_ids)
    tpos = {cid: k for k, cid in enumerate(order_ids)}
    M1 = T["M1_position"][np.array([tpos[c] for c in fids])]

    all_files = sorted({f for t in blob["tiers"].values() if t
                        for k in ("T0", "T1_dep", "T1_dry") for f in t.get(k, ())})

    CTX.update(dict(blob=blob, order_ids=order_ids, fids=fids, y=y, W=W,
                    yc=yc, mask=mask, dep=dep, base_p=base_p, Xtgc=Xtgc, M1=M1,
                    all_files=all_files,
                    bm=final_metrics(yc, np.clip(base_p, 0, 1)),
                    bo=final_metrics(yc[mask], np.clip(base_p[mask], 0, 1))))

    print(f"[{PROJECT}] WAVE 2 | rows={len(ev)} omega={mask.sum()}", flush=True)
    print(f"  baseline ALL F1={CTX['bm']['Macro_F1']:.4f} | "
          f"OM F1={CTX['bo']['Macro_F1']:.4f}", flush=True)

    for name, fn in (("W1", stage_W1), ("W2", stage_W2), ("W3", stage_W3),
                     ("W4", stage_W4), ("W5", stage_W5), ("W6", stage_W6)):
        t0 = time.time()
        try:
            fn()
            print(f"  == {name} ok ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  !! {name} FAILED: {e}", flush=True)
            traceback.print_exc()
            ROWS.append([name, f"FAILED: {e}", "", "", "", "", "", "", "", "", "", "", ""])
        flush()
    print(f"\n{len(ROWS)} rows -> {DEST/'wave2_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
