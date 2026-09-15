#!/usr/bin/env python3
"""
The remaining untested candidates, in one protected chained run.
================================================================

Each stage is independent and wrapped: a failure is recorded and the chain
continues, so one broken arm cannot lose the whole run. Every stage writes its
own CSV immediately on completion.

  R0  REPLACEMENT LADDER   -- does global context REPLACE commit-hub, or only
                              augment it? This decides the paper's claim.
  A4  BETWEENNESS/ARTIC    -- extends M1, the topology block that works
  B1  PARENT-CONTEXT TOKENS-- new_parent_type is stored and never read
  C1  METAPATH PROPAGATION -- Commit-Term-COOCCURS-Term-Commit (built, untested)
  C2  TYPED MULTI-REL PPR  -- per-relation transitions replacing the flat
                              bipartite projection
  D2  SELECTIVE ABSTENTION -- retest under the new features

Out: outputs/<project>/global_context/untested_<stage>.csv  (+ a combined file)
Run: KGC_PROJECT=kafka python inference/run_untested_arms.py
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

CTX = {}          # shared state across stages
ROWS = []         # combined results


# ------------------------------------------------------------------ utilities
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
    row = [stage, label, nfeat,
           round(float(m["Macro_F1"]), 4), round(float(m["MCC"]), 4),
           round(float(m["AUC"]), 4), round(float(mo["Macro_F1"]), 4),
           round(float(mo["MCC"]), 4),
           round(sa["diff"], 4) if sa else "", round(sa["p"], 4) if sa else "",
           round(so["diff"], 4) if so else "", round(so["p"], 4) if so else "",
           "YES" if both else ""]
    ROWS.append(row)
    print(f"  [{stage}] {label:<32} ALL F1={m['Macro_F1']:.4f} MCC={m['MCC']:.4f} | "
          f"OM F1={mo['Macro_F1']:.4f} MCC={mo['MCC']:.4f}"
          f"{'  <<< BOTH UP' if both else ''}", flush=True)
    return row


def flush(name):
    hdr = ["stage", "arm", "n_features", "all_MacroF1", "all_MCC", "all_AUC",
           "omega_MacroF1", "omega_MCC", "d_all", "p_all", "d_omega", "p_omega",
           "both_up"]
    with (DEST / name).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(ROWS)


def q(cypher):
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with drv.session(default_access_mode="READ") as s:
        out = s.execute_read(lambda tx: [r.data() for r in tx.run(cypher)])
    drv.close()
    return out


# ------------------------------------------------------------------- stage R0
def stage_R0():
    """Replacement ladder: can global context REPLACE the commit-hub score?"""
    c = CTX
    dep, F_only = c["dep"], c["F_only"]
    y, W = c["y"], c["W"]
    X, M1 = c["Xtgc"], c["M1"]

    record("R0", "commit-hub F+G (baseline)", c["base_p"], 0)
    record("R0", "TGC+M1 ONLY (no commit-hub)", stack([X, M1], y, W), X.shape[1] + M1.shape[1])
    record("R0", "F + TGC+M1 (drop CSTG G)", stack([F_only, X, M1], y, W),
           1 + X.shape[1] + M1.shape[1])
    record("R0", "F+G + TGC+M1 (augment)", stack([dep, X, M1], y, W),
           1 + X.shape[1] + M1.shape[1])
    record("R0", "TGC ONLY (no topology, no hub)", stack([X], y, W), X.shape[1])


# ------------------------------------------------------------------- stage A4
def stage_A4():
    """Betweenness / articulation proxies over the file dependency graph."""
    c = CTX
    blob, order_ids, fids = c["blob"], c["order_ids"], c["fids"]
    tiers = blob["tiers"]
    all_files = c["all_files"]
    fidx = {f: i for i, f in enumerate(all_files)}
    n = len(order_ids)
    F = np.zeros((n, 4))
    edges, seen = [], set()
    bounds = [int(n * k / N_SNAP) for k in range(N_SNAP + 1)]
    for s in range(N_SNAP):
        lo, hi = bounds[s], bounds[s + 1]
        if edges:
            A, idx = build_file_graph(edges, all_files)
            S = ((A + A.T) > 0).astype(float).tocsr()
            deg = np.asarray(S.sum(1)).ravel()
            # articulation proxy: a node whose neighbours are mutually
            # unconnected is a cut point in its local structure
            tri = np.asarray((S @ S).multiply(S).sum(1)).ravel()
            poss = deg * np.maximum(deg - 1, 1)
            clus = tri / np.maximum(poss, 1)
            artic = deg * (1.0 - clus)          # high degree, low clustering
            # 2-step reach: how much of the graph is within two hops
            reach = np.asarray((S @ S).astype(bool).sum(1)).ravel()
        else:
            idx = fidx
            deg = clus = artic = reach = np.zeros(len(all_files))
        for k in range(lo, hi):
            t = tiers.get(order_ids[k])
            ids = [idx[f] for f in (t["T0"] if t else []) if f in idx]
            if ids:
                F[k] = [float(np.log1p(artic[ids].mean())),
                        float(np.log1p(artic[ids].max())),
                        float(clus[ids].mean()),
                        float(np.log1p(reach[ids].mean()))]
            if t:
                for f in t["T0"]:
                    for g in t.get("T1_dep", ()):
                        if (g, f) not in seen:
                            seen.add((g, f)); edges.append((g, f))
                    for g in t.get("T1_dry", ()):
                        if (f, g) not in seen:
                            seen.add((f, g)); edges.append((f, g))
    tpos = {cid: k for k, cid in enumerate(order_ids)}
    A4 = F[np.array([tpos[x] for x in fids])]
    CTX["A4"] = A4
    record("A4", "M1 + articulation", stack([CTX["dep"], CTX["M1"], A4], CTX["y"], CTX["W"]),
           CTX["M1"].shape[1] + 4)
    record("A4", "TGC + M1 + articulation",
           stack([CTX["dep"], CTX["Xtgc"], CTX["M1"], A4], CTX["y"], CTX["W"]),
           CTX["Xtgc"].shape[1] + CTX["M1"].shape[1] + 4)


# ------------------------------------------------------------------- stage B1
def stage_B1():
    """Parent-context tokens: new_parent_type is stored and never read."""
    rows = q("""
        MATCH (c:Commit {in_jit:true})-[r:ADDS|MOVES]->(a:ASTNode)
        WHERE r.new_parent_type IS NOT NULL
        RETURN c.id AS cid, a.ast_type + '@' + r.new_parent_type AS tok,
               count(*) AS n
    """)
    per = defaultdict(dict)
    for r in rows:
        per[r["cid"]][r["tok"]] = float(r["n"])
    vocab = sorted({t for d in per.values() for t in d})
    vidx = {t: i for i, t in enumerate(vocab)}
    fids = CTX["fids"]
    Xp = np.zeros((len(fids), len(vocab)))
    for i, cid in enumerate(fids):
        for t, v in per.get(cid, {}).items():
            Xp[i, vidx[t]] = np.log1p(v)
    print(f"     parent-context vocabulary: {len(vocab)} types", flush=True)
    CTX["B1"] = Xp
    record("B1", "F+G + parent-context tokens",
           stack([CTX["dep"], Xp], CTX["y"], CTX["W"]), Xp.shape[1])
    record("B1", "TGC+M1 + parent tokens",
           stack([CTX["dep"], CTX["Xtgc"], CTX["M1"], Xp], CTX["y"], CTX["W"]),
           CTX["Xtgc"].shape[1] + CTX["M1"].shape[1] + Xp.shape[1])


# ------------------------------------------------------------------- stage C1
def stage_C1():
    """Metapath propagation: Commit-Term-COOCCURS-Term-Commit."""
    ment = q("MATCH (c:Commit {in_jit:true})-[:MENTIONS]->(t:Term) "
             "RETURN c.id AS cid, t.id AS tid")
    cooc = q("MATCH (a:Term)-[r:COOCCURS]->(b:Term) "
             "RETURN a.id AS s, b.id AS d, coalesce(r.npmi,1.0) AS w")
    order_ids = CTX["order_ids"]
    cpos = {cid: k for k, cid in enumerate(order_ids)}
    terms = sorted({r["tid"] for r in ment} | {r["s"] for r in cooc} | {r["d"] for r in cooc})
    tidx = {t: i for i, t in enumerate(terms)}
    nC, nT = len(order_ids), len(terms)
    r_, c_ = [], []
    for r in ment:
        if r["cid"] in cpos and r["tid"] in tidx:
            r_.append(cpos[r["cid"]]); c_.append(tidx[r["tid"]])
    CT = sp.csr_matrix((np.ones(len(r_)), (r_, c_)), shape=(nC, nT))
    rs, cs, ws = [], [], []
    for r in cooc:
        if r["s"] in tidx and r["d"] in tidx:
            rs.append(tidx[r["s"]]); cs.append(tidx[r["d"]]); ws.append(float(r["w"]))
    TT = sp.csr_matrix((ws, (rs, cs)), shape=(nT, nT))
    TT = TT + TT.T
    print(f"     metapath: {nT} terms, {TT.nnz} cooccurs, {CT.nnz} mentions", flush=True)

    # causal label propagation along Commit-Term-(COOCCURS)-Term-Commit
    y_all = np.array([c["buggy"] for c in CTX["blob"]["commits"]], float)
    lab = np.zeros(nC)
    out = np.zeros((nC, 2))
    bounds = [int(nC * k / N_SNAP) for k in range(N_SNAP + 1)]
    for s in range(N_SNAP):
        lo, hi = bounds[s], bounds[s + 1]
        tmass = CT.T @ lab                      # term-level past defect mass
        spread = TT @ tmass                      # one COOCCURS hop
        sc1 = CT[lo:hi] @ tmass
        sc2 = CT[lo:hi] @ spread
        deg = np.asarray(CT[lo:hi].sum(1)).ravel()
        deg[deg == 0] = 1.0
        out[lo:hi, 0] = np.log1p(sc1 / deg)
        out[lo:hi, 1] = np.log1p(sc2 / deg)
        lab[lo:hi] = y_all[lo:hi]                # reveal only after scoring
    fids = CTX["fids"]
    C1 = out[np.array([cpos[x] for x in fids])]
    CTX["C1"] = C1
    record("C1", "F+G + metapath(term)", stack([CTX["dep"], C1], CTX["y"], CTX["W"]), 2)
    record("C1", "TGC+M1 + metapath",
           stack([CTX["dep"], CTX["Xtgc"], CTX["M1"], C1], CTX["y"], CTX["W"]),
           CTX["Xtgc"].shape[1] + CTX["M1"].shape[1] + 2)


# ------------------------------------------------------------------- stage C2
def stage_C2():
    """Typed multi-relational PPR: per-relation transitions, learned weights."""
    c = CTX
    blob, order_ids, fids = c["blob"], c["order_ids"], c["fids"]
    tiers = blob["tiers"]
    all_files = c["all_files"]
    n = len(order_ids)
    OUT_ = np.zeros((n, 3))
    dep_e, dry_e, seen = [], [], set()
    risk = defaultdict(float)
    bounds = [int(n * k / N_SNAP) for k in range(N_SNAP + 1)]
    for s in range(N_SNAP):
        lo, hi = bounds[s], bounds[s + 1]
        mats = {}
        for name, es in (("dep", dep_e), ("dry", dry_e)):
            if es:
                A, idx = build_file_graph(es, all_files)
                d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1.0
                mats[name] = (sp.diags(1.0 / d) @ A, idx)
        if mats:
            rv = np.zeros(len(all_files))
            fmap = {f: i for i, f in enumerate(all_files)}
            for f, v in risk.items():
                if f in fmap:
                    rv[fmap[f]] = v
            props = {}
            for name, (P, idx) in mats.items():
                x = rv.copy()
                for _ in range(6):
                    x = 0.15 * rv + 0.85 * (P.T @ x)
                props[name] = x
            for k in range(lo, hi):
                t = tiers.get(order_ids[k])
                ids = [fmap[f] for f in (t["T0"] if t else []) if f in fmap]
                if ids:
                    a = props.get("dep", np.zeros(1))
                    b = props.get("dry", np.zeros(1))
                    OUT_[k] = [float(np.log1p(a[ids].mean())) if len(a) > 1 else 0.0,
                               float(np.log1p(b[ids].mean())) if len(b) > 1 else 0.0,
                               float(np.log1p(max(a[ids].max() if len(a) > 1 else 0,
                                                  b[ids].max() if len(b) > 1 else 0)))]
        for k in range(lo, hi):
            t = tiers.get(order_ids[k])
            if not t:
                continue
            for f in t["T0"]:
                for g in t.get("T1_dep", ()):
                    if (g, f) not in seen:
                        seen.add((g, f)); dep_e.append((g, f))
                for g in t.get("T1_dry", ()):
                    if (f, g) not in seen:
                        seen.add((f, g)); dry_e.append((f, g))
            if CTX["blob"]["commits"][k]["buggy"]:
                for f in t["T0"]:
                    risk[f] += 1.0
    tpos = {cid: k for k, cid in enumerate(order_ids)}
    C2 = OUT_[np.array([tpos[x] for x in fids])]
    CTX["C2"] = C2
    record("C2", "F+G + typed multi-rel PPR", stack([CTX["dep"], C2], CTX["y"], CTX["W"]), 3)
    record("C2", "TGC+M1 + typed PPR",
           stack([CTX["dep"], CTX["Xtgc"], CTX["M1"], C2], CTX["y"], CTX["W"]),
           CTX["Xtgc"].shape[1] + CTX["M1"].shape[1] + 3)


# ------------------------------------------------------------------- stage D2
def stage_D2():
    """Selective abstention under the best feature set."""
    c = CTX
    p = stack([c["dep"], c["Xtgc"], c["M1"]], c["y"], c["W"])
    yc, mask = c["yc"], c["mask"]
    conf = np.abs(p - 0.5)
    for cov in (0.95, 0.90, 0.85, 0.80):
        thr = np.quantile(conf, 1 - cov)
        keep = conf >= thr
        if keep.sum() < 100 or len(np.unique(yc[keep])) < 2:
            continue
        m = final_metrics(yc[keep], np.clip(p[keep], 0, 1))
        km = keep & mask
        mo = (final_metrics(yc[km], np.clip(p[km], 0, 1))
              if km.sum() >= 10 and len(np.unique(yc[km])) > 1 else None)
        ROWS.append(["D2", f"abstain -> coverage {cov:.0%}", int(keep.sum()),
                     round(float(m["Macro_F1"]), 4), round(float(m["MCC"]), 4),
                     round(float(m["AUC"]), 4),
                     round(float(mo["Macro_F1"]), 4) if mo else "",
                     round(float(mo["MCC"]), 4) if mo else "", "", "", "", "", ""])
        print(f"  [D2] coverage {cov:.0%}  n={keep.sum():>5}  ALL F1={m['Macro_F1']:.4f} "
              f"MCC={m['MCC']:.4f}" + (f" | OM F1={mo['Macro_F1']:.4f}" if mo else ""),
              flush=True)


# ---------------------------------------------------------------------- driver
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
    F_only = np.asarray(fu["scores"]["F"], float)
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
    sel = np.array([tpos[c] for c in fids])
    M1 = T["M1_position"][sel]

    all_files = sorted({f for t in blob["tiers"].values() if t
                        for k in ("T0", "T1_dep", "T1_dry") for f in t.get(k, ())})

    CTX.update(dict(blob=blob, order_ids=order_ids, fids=fids, y=y, W=W, ev=ev,
                    yc=yc, mask=mask, dep=dep, F_only=F_only, base_p=base_p,
                    Xtgc=Xtgc, M1=M1, all_files=all_files,
                    bm=final_metrics(yc, np.clip(base_p, 0, 1)),
                    bo=final_metrics(yc[mask], np.clip(base_p[mask], 0, 1))))

    print(f"[{PROJECT}] UNTESTED ARMS | rows={len(ev)} omega={mask.sum()}", flush=True)
    print(f"  baseline ALL F1={CTX['bm']['Macro_F1']:.4f} MCC={CTX['bm']['MCC']:.4f} | "
          f"OM F1={CTX['bo']['Macro_F1']:.4f} MCC={CTX['bo']['MCC']:.4f}", flush=True)

    for name, fn in (("R0", stage_R0), ("A4", stage_A4), ("B1", stage_B1),
                     ("C1", stage_C1), ("C2", stage_C2), ("D2", stage_D2)):
        t0 = time.time()
        try:
            fn()
            print(f"  == {name} ok ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  !! {name} FAILED: {e}", flush=True)
            traceback.print_exc()
            ROWS.append([name, f"FAILED: {type(e).__name__}: {e}", "", "", "", "",
                         "", "", "", "", "", "", ""])
        flush("untested_results.csv")     # protected: write after every stage

    print(f"\n{len(ROWS)} rows -> {DEST/'untested_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
