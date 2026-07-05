"""
Five typical, graph-native, online (prequential) KG-inference methods for the v4
"which subgraph is best?" study. These are ADDITIVE: the earlier methods
(M metrics, R priors, T structural TF-IDF, P PPR, E SVD, Fusion) are untouched;
this module supplies a clean, purely graph-information family that can be run on
two graph scopes so the subgraph's contribution is not confounded by the shared
file/developer relational layer:

  scope="subgraph"  commits linked only to their change-token hubs
                    (the graph the structural subgraph actually defines)
  scope="full"      commits linked to token + file + developer hubs
                    (the deployed heterogeneous graph)

The graph is the commit--hub bipartite network G=(commits ∪ hubs, memberships).
Every method predicts a commit's buggy probability from ONLY graph information and
ONLY strictly-past labels/edges (leakage-free, prequential):

  RN   weighted-vote Relational Neighbour  -- collective classification: a
       commit is scored by the label-weighted hub-overlap with past commits.
  PPR  class-seeded Personalized PageRank / RWR on the bipartite graph.
  LP   Label Propagation / spreading activation of past labels through hubs.
  DW   DeepWalk-style random-walk embedding (PPMI-of-co-occurrence + SVD, the
       Levy-Goldberg matrix-factorisation form) + logistic head.
  KGE  DistMult knowledge-graph embedding over typed (commit, rel, hub) triples,
       commits folded in from learned hub embeddings + logistic head.

All are CPU-only and refit on expanding past windows (the standard prequential
approximation). Used by inference/run_kg_methods_rq.py.
"""
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize

METHODS = ["RN", "PPR", "LP", "DW", "KGE"]
METHOD_NAME = {
    "RN":  "Relational neighbour (wvRN)",
    "PPR": "Personalized PageRank (RWR)",
    "LP":  "Label propagation",
    "DW":  "DeepWalk embedding (RW+MF)",
    "KGE": "KG embedding (DistMult)",
}
SCOPES = ["subgraph", "full"]


def _lr():
    return LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")


# ── scoped commit--hub graph ────────────────────────────────────────────────

def build_scoped_graph(cids, tokens, files, devs, scope):
    """Commit x hub TF-IDF incidence C, the symmetric bipartite PPR transition P
    over N=Nc+Nh nodes, and the typed edge list (commit, rel, hub) for KGE.
    scope='subgraph' uses only token hubs; 'full' adds file and developer hubs."""
    import math
    cidx = {c: i for i, c in enumerate(cids)}; Nc = len(cids)
    hub = {}; raw = []; edges = []           # edges: (commit_i, rel, hub_j)
    REL = {"T": 0, "F": 1, "D": 2}
    use_fd = (scope == "full")
    for c in cids:
        i = cidx[c]
        for tok, n in tokens.get(c, ()):
            raw.append((i, "T:" + tok, float(n), "T"))
        if use_fd:
            for f in files.get(c, ()):
                raw.append((i, "F:" + f, 1.0, "F"))
            if c in devs:
                raw.append((i, "D:" + devs[c], 1.0, "D"))
    df = defaultdict(int)
    for i, h, w, t in raw:
        df[h] += 1
    rows, cols, data = [], [], []
    for i, h, w, t in raw:
        j = hub.setdefault(h, len(hub))
        rows.append(i); cols.append(j)
        data.append(w * math.log(1.0 + Nc / df[h]))
        edges.append((i, REL[t], j))
    Nh = len(hub); N = Nc + Nh
    C = sp.csr_matrix((data, (rows, cols)), shape=(Nc, max(Nh, 1)))
    # bipartite PPR transition (column-normalised symmetric adjacency)
    B = sp.csr_matrix((data, (rows, [c + Nc for c in cols])), shape=(N, N))
    A = B + B.T
    deg = np.asarray(A.sum(0)).ravel(); deg[deg == 0] = 1.0
    P = A.multiply(sp.csr_matrix(1.0 / deg)).tocsr()
    return dict(C=C, P=P, Nc=Nc, Nh=Nh, N=N,
                edges=np.array(edges) if edges else np.zeros((0, 3), int))


# ── individual methods (each returns block scores from strictly-past state) ──

def _ppr(P, seeds, N, alpha=0.15, iters=40):
    if not len(seeds):
        return np.zeros(N)
    r = np.zeros(N); r[seeds] = 1.0 / len(seeds); x = r.copy()
    for _ in range(iters):
        x = (1 - alpha) * (P @ x) + alpha * r
    return x


def rn_scores(C, y, past, block):
    """wvRN: score(c) = (sum over past commits sharing hubs, weighted by hub
    overlap and their label) / (weighted count). One hop through hubs."""
    Cp = C[past]
    sim = C[block] @ Cp.T                       # block x |past| (sparse)
    num = sim @ y[past].astype(float)
    den = np.asarray(sim.sum(1)).ravel()
    g = y[past].mean() if len(past) else 0.0
    return np.where(den > 0, num / np.maximum(den, 1e-9), g)


def lp_scores(C, y, past, block, iters=3):
    """Label propagation / spreading activation: seed past commits with labels,
    diffuse commit->hub->commit a few steps, read block-commit activation."""
    Cn = normalize(C, norm="l1", axis=1)        # commit->hub distribution
    Hn = normalize(C, norm="l1", axis=0)        # hub->commit distribution (cols)
    f = np.full(C.shape[0], y[past].mean() if len(past) else 0.0)
    f[past] = y[past].astype(float)
    for _ in range(iters):
        h = Hn.T @ f                            # hubs gather from commits
        f_new = Cn @ h                          # commits gather from hubs
        f = f_new.copy(); f[past] = y[past].astype(float)   # clamp known past
    return f[block]


def dw_embed(C, past, dim, window_pow=3):
    """DeepWalk-as-matrix-factorisation: PPMI of the commit-hub co-occurrence
    (marginals from PAST rows only -> leakage-free), truncated-SVD embedding.
    Returns commit embeddings for ALL commits (block commits folded in via the
    past-derived basis)."""
    from sklearn.utils.extmath import randomized_svd
    Cp = C[past]
    col = np.asarray(Cp.sum(0)).ravel() + 1e-9
    tot = col.sum()
    row_all = np.asarray(C.sum(1)).ravel() + 1e-9
    # PPMI(c,h) = max(log( x_ch * tot / (row_c * col_h) ), 0)
    Coo = C.tocoo()
    pcol = col[Coo.col]; prow = row_all[Coo.row]
    pmi = np.log(np.maximum(Coo.data, 1e-9) * tot / (prow * pcol) + 1e-12)
    pmi = np.maximum(pmi, 0.0)
    M = sp.csr_matrix((pmi, (Coo.row, Coo.col)), shape=C.shape)
    d = min(dim, min(M.shape) - 1)
    if d < 2:
        return np.zeros((C.shape[0], max(dim, 2)))
    U, S, _ = randomized_svd(M, n_components=d, random_state=0)
    return U * np.sqrt(S)


def kge_embed(edges, Nc, Nh, past_mask, dim=32, epochs=3, neg=3, lr=0.05, seed=0):
    """Compact DistMult over typed (commit, rel, hub) triples restricted to past
    commits. Learns hub + relation embeddings by logistic loss with negative
    (tail-corruption) sampling; commits are represented by fold-in aggregation of
    their hubs' embeddings (so unseen block commits get an embedding from their
    known-at-arrival hubs). Returns per-commit fold-in embeddings for ALL commits."""
    rng = np.random.default_rng(seed)
    tr = edges[past_mask[edges[:, 0]]]          # triples of past commits only
    if len(tr) < 10 or Nh == 0:
        return np.zeros((Nc, dim))
    nR = int(edges[:, 1].max()) + 1
    Eh = 0.1 * rng.standard_normal((Nh, dim))   # hub embeddings
    Ec = 0.1 * rng.standard_normal((Nc, dim))   # commit embeddings (trained)
    R = 0.1 * rng.standard_normal((nR, dim)) + 1.0
    ci, ri, hi = tr[:, 0], tr[:, 1], tr[:, 2]
    for _ in range(epochs):
        perm = rng.permutation(len(tr))
        for s in range(0, len(tr), 2048):
            b = perm[s:s + 2048]
            c, r, h = ci[b], ri[b], hi[b]
            # positive
            score = np.sum(Ec[c] * R[r] * Eh[h], axis=1)
            sig = 1.0 / (1.0 + np.exp(-score))
            gpos = (sig - 1.0)[:, None]
            # negative: corrupt hub
            hn = rng.integers(0, Nh, size=(len(b), neg))
            for k in range(neg):
                hk = hn[:, k]
                sn = np.sum(Ec[c] * R[r] * Eh[hk], axis=1)
                sgn = 1.0 / (1.0 + np.exp(-sn)); gneg = sgn[:, None]
                Eh[hk] -= lr * gneg * (Ec[c] * R[r])
                Ec[c]  -= lr * gneg * (R[r] * Eh[hk])
                R[r]   -= lr * gneg * (Ec[c] * Eh[hk])
            Ec[c] -= lr * gpos * (R[r] * Eh[h])
            Eh[h] -= lr * gpos * (Ec[c] * R[r])
            R[r]  -= lr * gpos * (Ec[c] * Eh[h])
    # fold-in EVERY commit from its hubs (known at arrival) -> mean of r*hub
    emb = np.zeros((Nc, dim)); cnt = np.zeros(Nc)
    ci_all, ri_all, hi_all = edges[:, 0], edges[:, 1], edges[:, 2]
    np.add.at(emb, ci_all, R[ri_all] * Eh[hi_all])
    np.add.at(cnt, ci_all, 1.0)
    return emb / np.maximum(cnt[:, None], 1.0)
