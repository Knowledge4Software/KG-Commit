"""
Enhanced ONLINE CSTG features -- brings the graph mechanisms into the streaming
path (the deployment protocol), fixing the audit findings:

  * text stream is now GRAPH-OF-WORDS weighted (TextRank centrality), not plain TF
  * tokens carry ADD/REMOVE/MSG POLARITY (a term added vs removed differs)
  * the prior is NPMI-PROPAGATED ONLINE: incremental co-occurrence counters, with
    the term-risk periodically propagated over the past-only NPMI graph -- the
    relational generalisation a flat bag-of-words cannot do, now available online.

Everything is strictly past-only (predict-then-grow). Per-commit parsing + TextRank
are commit-local (leakage-free) and cached.

Public: build_online_streams(cids, y, warmup) -> dict(Xtext, prior, typed)
aligned to the given commit order (by commit id / SHA).
"""
import pickle
from pathlib import Path
from collections import defaultdict, Counter
import math
import numpy as np, scipy.sparse as sp, pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
import cstg as C

import _kgc_paths  # noqa: F401  (adds package dirs to sys.path)
from config.project_config import OUT as _OUT, DIFF_CSV as DIFFS, PROJECT_KEY as _PKEY
ROOT = _OUT                              # per-project outputs/<project>/ (cache root)
# FINAL RUN: the CSTG refresh joins the single-M rule. NOTE this constant is in
# COMMITS, not blocks -- it becomes BLOCK (=M), i.e. one refresh per block, matching
# the fusion head, the embeddings and the baselines. It was 400 = 2M.
from protocol import HASH_DIM, BLOCK as PROP_REFRESH  # noqa: F401
PROP_ITERS = 4
PROP_DAMP = 0.5
NPMI_MIN = 0.3
TYPES = {"code": 0, "bug": 1, "action": 2, "error": 3, "nl": 4}


def _parse_all():
    """Per-commit list of (source, type, term) with source in {m,a,r}, + centrality.
    Cached (TextRank is the slow part)."""
    cache = ROOT / "cstg_polarity_docs.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    df = pd.read_csv(DIFFS, usecols=["commit_id", "project", "author_date"]).query(f"project=='{_PKEY}'")
    df = df.sort_values("author_date").reset_index(drop=True)
    texts = pd.read_csv(DIFFS, usecols=["commit_id", "project", "diff_text"]).query(f"project=='{_PKEY}'")
    texts = texts.set_index("commit_id")["diff_text"].astype(str)
    out = {}
    for k, cid in enumerate(df["commit_id"].astype(str)):
        p = C.parse_commit_text(texts.get(cid, ""))
        toks = ([("m", t, w) for t, w in C._tokens(p["message"], code=True)]
                + [("a", t, w) for t, w in C._tokens(p["added"], code=True)]
                + [("r", t, w) for t, w in C._tokens(p["removed"], code=True)])
        toks = toks[:4000]
        cen = C.graph_of_words_weights([w for _, _, w in toks]) if toks else {}
        out[cid] = (toks, cen)
        if k % 1500 == 0: print(f"  parsed {k}")
    pickle.dump(out, open(cache, "wb"))
    return out


def build_online_streams(cids, y):
    id2 = _parse_all()
    toks_l = [id2.get(c, ([], {}))[0] for c in cids]
    cens_l = [id2.get(c, ([], {}))[1] for c in cids]
    n = len(cids)

    # ---- text: graph-of-words-weighted, polarity-tagged tokens (stateless hash) ----
    docs = []
    for toks, cen in zip(toks_l, cens_l):
        if not toks:
            docs.append(""); continue
        mx = max(cen.values()) if cen else 1.0
        parts = []
        for src, typ, term in toks:
            reps = 1 + int(4 * cen.get(term, 0) / (mx + 1e-9))     # TextRank -> repetition
            parts += [f"{src}:{typ[:1]}:{term}"] * min(reps, 5)     # polarity + type + term
        docs.append(" ".join(parts))
    Xtext = HashingVectorizer(n_features=HASH_DIM, alternate_sign=False,
                              token_pattern=r"[^\s]+").transform(docs)

    # ---- NPMI-propagated online prior + typed mass (strictly past-only) ----
    tb = defaultdict(float); tt = defaultdict(float); gb = gt = 0.0
    pair = defaultdict(float)                    # co-occurrence counts (past)
    prop = {}                                     # last propagated term-risk
    prior = np.zeros(n); typed = np.zeros((n, 5))

    def refresh():
        """Recompute NPMI from past counts, propagate base risk over it."""
        g = (gb / gt) if gt else 0.0
        base = {t: (tb[t] + g * 5) / (tt[t] + 5) for t in tt}
        adj = defaultdict(list)
        N = gt if gt else 1
        for (a, b), c in pair.items():
            if c < 2: continue
            pa, pb, pab = tt[a] / N, tt[b] / N, c / N
            if pab <= 0 or pa <= 0 or pb <= 0: continue
            npmi = math.log(pab / (pa * pb)) / (-math.log(pab))
            if npmi >= NPMI_MIN:
                adj[a].append((b, npmi)); adj[b].append((a, npmi))
        risk = dict(base)
        for _ in range(PROP_ITERS):
            nr = {}
            for t in base:
                nb = adj.get(t)
                if nb:
                    ws = sum(w for _, w in nb)
                    nr[t] = (1 - PROP_DAMP) * base[t] + PROP_DAMP * sum(w * risk.get(u, g) for u, w in nb) / ws
                else:
                    nr[t] = base[t]
            risk = nr
        return risk, g

    prop, gcur = refresh()
    for i in range(n):
        toks, cen = toks_l[i], cens_l[i]
        g = (gb / gt) if gt else 0.0
        if cen:
            num = den = 0.0
            for src, typ, term in toks:
                w = cen.get(term, 0.0)
                num += w * prop.get(term, g); den += w
                typed[i, TYPES[typ]] += w
            prior[i] = num / den if den else g
        else:
            prior[i] = g
        # grow (past -> now includes i for the NEXT commits)
        gt += 1; gb += int(y[i])
        uniq = sorted({term for _, _, term in toks})
        for t in uniq: tt[t] += 1; tb[t] += int(y[i])
        for a in range(len(uniq)):
            for b in range(a + 1, len(uniq)):
                pair[(uniq[a], uniq[b])] += 1
        if (i + 1) % PROP_REFRESH == 0:
            prop, gcur = refresh()
    return dict(Xtext=Xtext, prior=prior, typed=typed)


if __name__ == "__main__":
    # quick self-check on the diff CSV order
    df = pd.read_csv(DIFFS, usecols=["commit_id", "project", "buggy", "author_date"]).query(f"project=='{_PKEY}'")
    df = df.sort_values("author_date").reset_index(drop=True)
    cids = df["commit_id"].astype(str).tolist(); y = df["buggy"].astype(int).to_numpy()
    S = build_online_streams(cids, y)
    print("Xtext", S["Xtext"].shape, "prior nz", int((S["prior"] > 0).sum()), "typed", S["typed"].shape)
