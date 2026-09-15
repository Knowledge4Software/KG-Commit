#!/usr/bin/env python3
"""
Topology-native arms on the Kafka pilot.
========================================

Runs M1-M5 from kg_topology_methods on top of the deployed F+G, and against the
best TGC configuration, so we can tell whether GLOBAL TOPOLOGY adds anything
beyond neighbourhood summarisation.

Snapshotting: the file dependency graph is rebuilt at intervals using only edges
observed before that point, so no future structure leaks. Global quantities
(PageRank, k-core, spectral embedding, risk diffusion) are recomputed per
snapshot and held constant within it.

Out: outputs/<project>/global_context/topology_results.csv
Run: KGC_PROJECT=kafka python inference/run_kg_topology.py
"""
from __future__ import annotations

import csv
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import _kgc_paths  # noqa: E402,F401
from config.project_config import OUT, PROJECT  # noqa: E402
from online_jit import final_metrics  # noqa: E402
from protocol import BLOCK  # noqa: E402
from run_tgc_pilot import build_features, omega_ids  # noqa: E402
from kg_topology_methods import (build_file_graph, architectural_position,  # noqa: E402
                                 risk_ppr, spectral_embedding, dependency_shape)

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

DEST = OUT / "global_context"
ROOT = HERE.parent.parent
N_SNAPSHOTS = 8
SPECTRAL_DIM = 16


def stack(cols, y, W, C=0.2, penalty="l1"):
    Z = np.column_stack(cols)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
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


def boot(y, pa, pb, metric="Macro_F1", n=300, seed=0):
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


def snapshot_features(blob, order_ids):
    """Global topology features per commit, recomputed on causal snapshots."""
    tiers = blob["tiers"]
    commits = blob["commits"]
    pos_of = {c["id"]: k for k, c in enumerate(commits)}

    # accumulate the file graph causally from the tier adjacency
    all_files = sorted({f for t in tiers.values() if t
                        for key in ("T0", "T1_dep", "T1_dry")
                        for f in t.get(key, ())})
    fidx = {f: i for i, f in enumerate(all_files)}
    nF = len(all_files)

    edges = []
    seen_edge = set()
    risk = np.zeros(nF)

    n = len(order_ids)
    bounds = [int(n * k / N_SNAPSHOTS) for k in range(N_SNAPSHOTS + 1)]

    M1 = np.zeros((n, 5))
    M2 = np.zeros((n, 2))
    M3 = np.zeros((n, 4))
    M4 = np.zeros((n, 8))

    A = None
    idx = fidx
    for s in range(N_SNAPSHOTS):
        lo, hi = bounds[s], bounds[s + 1]
        # rebuild global structures from everything observed BEFORE lo
        if edges:
            A, idx = build_file_graph(edges, all_files)
            posn = architectural_position(A)
            emb = spectral_embedding(A, dim=SPECTRAL_DIM)
            rp = risk_ppr(A, risk)
        else:
            posn = None

        for k in range(lo, hi):
            cid = order_ids[k]
            t = tiers.get(cid)
            f0 = t["T0"] if t else []
            ids = [idx[f] for f in f0 if f in idx]
            if posn is not None and ids:
                M1[k] = [np.log1p(posn["pagerank"][ids].mean() * nF),
                         np.log1p(posn["rev_pagerank"][ids].mean() * nF),
                         posn["kcore"][ids].mean(),
                         np.log1p(posn["in_deg"][ids].mean()),
                         np.log1p(posn["out_deg"][ids].mean())]
                M2[k] = [np.log1p(rp[ids].mean() * nF), np.log1p(rp[ids].max() * nF)]
                E = emb[ids]
                M3[k] = [E.mean(0)[:1].item() if E.size else 0.0,
                         float(np.linalg.norm(E.mean(0))),
                         float(E.std(0).mean()) if len(ids) > 1 else 0.0,
                         float(len(ids))]
                M4[k] = dependency_shape(A, idx, f0)
            # fold this commit in AFTER using the snapshot
            if t:
                for f in t["T0"]:
                    for g in t.get("T1_dep", ()):
                        e = (g, f)
                        if e not in seen_edge:
                            seen_edge.add(e)
                            edges.append(e)
                    for g in t.get("T1_dry", ()):
                        e = (f, g)
                        if e not in seen_edge:
                            seen_edge.add(e)
                            edges.append(e)
            c = commits[pos_of[cid]] if cid in pos_of else None
            if c and c["buggy"] and t:
                for f in t["T0"]:
                    if f in fidx:
                        risk[fidx[f]] += 1.0
    return {"M1_position": M1, "M2_diffusion": M2,
            "M3_spectral": M3, "M4_shape": M4}


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
    yc, idc = y[ev], [fids[i] for i in ev]
    mask = np.array([i in om for i in idc])
    print(f"[{PROJECT}] TOPOLOGY | rows={len(ev)} omega={mask.sum()}", flush=True)

    print("  computing snapshot topology ...", flush=True)
    order_ids = [c["id"] for c in blob["commits"]]
    T = snapshot_features(blob, order_ids)
    tpos = {cid: k for k, cid in enumerate(order_ids)}
    sel = np.array([tpos[c] for c in fids])
    T = {k: v[sel] for k, v in T.items()}
    print("  done", flush=True)

    # best TGC block from the sweep (T0 + directional T1)
    X, _, ids_all = build_features(blob, ["T0", "T1_dep", "T1_dry"])
    p2 = {c: k for k, c in enumerate(ids_all)}
    Xtgc = X[np.array([p2[c] for c in fids])]

    base_p = dep[ev]
    bm = final_metrics(yc, np.clip(base_p, 0, 1))
    bo = final_metrics(yc[mask], np.clip(base_p[mask], 0, 1))
    rows = [["deployed (F+G)", 0, round(float(bm["Macro_F1"]), 4), round(float(bm["MCC"]), 4),
             round(float(bm["AUC"]), 4), round(float(bo["Macro_F1"]), 4),
             round(float(bo["MCC"]), 4), "", "", "", ""]]
    print(f"  BASELINE  ALL F1={bm['Macro_F1']:.4f} MCC={bm['MCC']:.4f} | "
          f"OM F1={bo['Macro_F1']:.4f} MCC={bo['MCC']:.4f}", flush=True)

    arms = {
        "M1 position only":        [dep, T["M1_position"]],
        "M2 diffusion(PPR) only":  [dep, T["M2_diffusion"]],
        "M3 spectral only":        [dep, T["M3_spectral"]],
        "M4 dep-shape only":       [dep, T["M4_shape"]],
        "M1+M2":                   [dep, T["M1_position"], T["M2_diffusion"]],
        "M1+M2+M4":                [dep, T["M1_position"], T["M2_diffusion"], T["M4_shape"]],
        "ALL topology (M1-M4)":    [dep, T["M1_position"], T["M2_diffusion"],
                                    T["M3_spectral"], T["M4_shape"]],
        "TGC only (best sweep)":   [dep, Xtgc],
        "TGC + M1":                [dep, Xtgc, T["M1_position"]],
        "TGC + M2":                [dep, Xtgc, T["M2_diffusion"]],
        "TGC + M4":                [dep, Xtgc, T["M4_shape"]],
        "TGC + ALL topology":      [dep, Xtgc, T["M1_position"], T["M2_diffusion"],
                                    T["M3_spectral"], T["M4_shape"]],
    }

    for label, cols in arms.items():
        p = stack(cols, y, W)
        m = final_metrics(yc, np.clip(p, 0, 1))
        mo = final_metrics(yc[mask], np.clip(p[mask], 0, 1))
        sa = boot(yc, p, base_p)
        so = boot(yc[mask], p[mask], base_p[mask])
        both = (m["Macro_F1"] > bm["Macro_F1"]) and (mo["Macro_F1"] > bo["Macro_F1"])
        nf = sum(c.shape[1] if c.ndim > 1 else 1 for c in cols[1:])
        rows.append([label, nf, round(float(m["Macro_F1"]), 4), round(float(m["MCC"]), 4),
                     round(float(m["AUC"]), 4), round(float(mo["Macro_F1"]), 4),
                     round(float(mo["MCC"]), 4),
                     round(sa["diff"], 4) if sa else "", round(sa["p"], 4) if sa else "",
                     round(so["diff"], 4) if so else "", round(so["p"], 4) if so else ""])
        print(f"  {label:<25} n={nf:>3} ALL F1={m['Macro_F1']:.4f} MCC={m['MCC']:.4f} | "
              f"OM F1={mo['Macro_F1']:.4f} MCC={mo['MCC']:.4f}"
              f"{'  <<< BOTH UP' if both else ''}", flush=True)

    with (DEST / "topology_results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "n_features", "all_MacroF1", "all_MCC", "all_AUC",
                    "omega_MacroF1", "omega_MCC", "d_all", "p_all", "d_omega", "p_omega"])
        w.writerows(rows)
    print(f"\nsaved -> {DEST/'topology_results.csv'}")


if __name__ == "__main__":
    main()
