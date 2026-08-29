"""
=============================================================================
DeepJIT baseline under the EXACT KG-Commit online prequential protocol
=============================================================================
Self-contained Kaggle (T4 GPU) script. No external repo needed; only the two
ApacheJIT CSVs per project (uploaded as a Kaggle Dataset, each < 200 MB).

DeepJIT (Hoang et al., ICSE 2019): a compact CNN over the commit MESSAGE tokens
and the commit CODE (added/removed diff) tokens, fused by an MLP into a buggy
probability. Trained from scratch (no pretraining, no downloads) -- fits on a T4
in minutes.

Evaluation is IDENTICAL to the other baselines in this project so the numbers drop
straight into the same tables:
  * chronological order (author_date)
  * warm-up on the first WARMUP_FRAC = 0.40 of commits (not scored)
  * predict-then-learn in blocks of BLOCK = 200
  * FAITHFUL online refit: retrain DeepJIT on the expanding past every REFIT_EVERY=3
    blocks (Option A -- protocol-faithful, matches KG-Commit's refit discipline)
  * leakage-free ONLINE-tuned threshold (online_decisions, init=300, step=150,
    maximise buggy-F1 on the past) for the operating-point metrics
  * the same 7 metrics: Precision, Recall, Macro_F1, Buggy_F1, G_Mean, AUC, ACC
    (+ trajectories at window=150 and window=800, so the stream figures line up)

HOW TO RUN ON KAGGLE
--------------------
1. Create a Kaggle Dataset containing, per project, the two files:
       apache_<project>.csv          (labels + 12 metrics)
       apache_<project>_diff.csv     (adds a diff_text column)
   (Upload one project at a time or several; each file < 200 MB -- all your
   projects are 18-74 MB, so this is fine.)
2. New Notebook -> add that Dataset -> Settings: Accelerator = GPU T4.
3. Paste this file into a cell (or import it), set PROJECT below, Run All.
4. Download the produced  deepjit_<project>_results.pkl  and place it in
   outputs/<project>/ locally; the repo's make_baselines_doc.py will pick it up.

Run the 5 projects in this order (small -> large):
   zookeeper, zeppelin, kafka, activemq, groovy
=============================================================================
"""
# ---------------------------------------------------------------------------
# 0. CONFIG  (edit PROJECT and DATA_DIR per Kaggle run)
# ---------------------------------------------------------------------------
PROJECT  = "zookeeper"                       # <-- change per run
DATA_DIR = "/kaggle/input/apachejit-diffs"   # <-- your Kaggle dataset mount path
OUT_DIR  = "/kaggle/working"

# FINAL RUN: these MUST mirror kgcommit_repro/inference/protocol.py. This script
# runs standalone on Kaggle and cannot import it, so the values are restated here
# and must be updated together -- otherwise the Deeper baseline is evaluated under
# a different protocol than every other model, which invalidates the comparison.
WARMUP_FRAC = 0.05      # protocol.WARMUP_FRAC (K)
BLOCK       = 200       # protocol.BLOCK (M)
GAP         = 50        # protocol.GAP (G)
REFIT_EVERY = 1         # protocol.REFIT_EVERY -- single-M rule: refit every block
EPOCHS      = 8         # per (re)fit; small CNN converges fast
BATCH       = 64
MSG_LEN     = 64        # token budget for the message channel
CODE_LEN    = 256       # token budget for the code (added+removed) channel
VOCAB_MAX   = 20000
EMB_DIM     = 64
SEED        = 0

import os, re, pickle, time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, matthews_corrcoef,
                             brier_score_loss)

torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV)


# ===========================================================================
# 1. EXACT protocol metrics  (verbatim port of online_jit.py)
# ===========================================================================
def _best_threshold(y, p):
    y = np.asarray(y); p = np.asarray(p, float)
    P = int(y.sum())
    if P == 0 or P == len(y):
        return 0.5
    order = np.argsort(-p); ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / (tp + fp); rec = tp / P
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(p[order][int(np.argmax(f1))])


def online_decisions(p, y, init=300, step=150):
    p = np.asarray(p); y = np.asarray(y)
    yhat = np.zeros(len(y), int); thr = 0.5
    for i in range(len(y)):
        yhat[i] = int(p[i] >= thr)
        if i + 1 >= init and (i + 1) % step == 0:
            thr = _best_threshold(y[:i + 1], p[:i + 1])
    return yhat


def cum_metrics(y, p):
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    yh = (p >= 0.5).astype(int)
    out = dict(ROC_AUC=float("nan"), PR_AUC=float("nan"))
    if len(np.unique(y)) > 1:
        out["ROC_AUC"] = roc_auc_score(y, p); out["PR_AUC"] = average_precision_score(y, p)
    out.update(F1=f1_score(y, yh, zero_division=0), MCC=matthews_corrcoef(y, yh),
               Brier=brier_score_loss(y, p), Acc=float((yh == y).mean()))
    return out


def final_metrics(y, p):
    y = np.asarray(y); p = np.clip(np.asarray(p, float), 0.0, 1.0)
    cm = cum_metrics(y, p)
    yhat = online_decisions(p, y)
    rec = recall_score(y, yhat, pos_label=1, zero_division=0)
    spec = recall_score(y, yhat, pos_label=0, zero_division=0)
    cm.update({
        "F1_online": f1_score(y, yhat, zero_division=0),
        "Precision": float(precision_score(y, yhat, pos_label=1, zero_division=0)),
        "Recall":    float(rec),
        "Buggy_F1":  float(f1_score(y, yhat, pos_label=1, zero_division=0)),
        "Macro_F1":  float(f1_score(y, yhat, average="macro", zero_division=0)),
        "G_Mean":    float(np.sqrt(max(rec, 0.0) * max(spec, 0.0))),
        "AUC":       cm["ROC_AUC"],
        "ACC":       float((yhat == y).mean()),
    })
    return cm


M7 = ["Precision", "Recall", "Macro_F1", "Buggy_F1", "G_Mean", "AUC", "ACC"]


def metric_trajectory(y_ev, p_ev, warmup, roll=150, step=25):
    """Rolling 7-metric online-evaluation trajectory (matches run_subgraph_rq)."""
    yhat = online_decisions(p_ev, y_ev)
    tr = {"idx": []}; tr.update({m: [] for m in M7})
    n = len(y_ev)
    for j in range(step, n + 1, step):
        lo = max(0, j - roll)
        ys, ps, yh = y_ev[lo:j], p_ev[lo:j], yhat[lo:j]
        rec = recall_score(ys, yh, pos_label=1, zero_division=0)
        spec = recall_score(ys, yh, pos_label=0, zero_division=0)
        tr["idx"].append(warmup + j)
        tr["Precision"].append(precision_score(ys, yh, pos_label=1, zero_division=0))
        tr["Recall"].append(rec)
        tr["Macro_F1"].append(f1_score(ys, yh, average="macro", zero_division=0))
        tr["Buggy_F1"].append(f1_score(ys, yh, pos_label=1, zero_division=0))
        tr["G_Mean"].append(float(np.sqrt(max(rec, 0) * max(spec, 0))))
        tr["AUC"].append(roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else np.nan)
        tr["ACC"].append(float((yh == ys).mean()))
    return tr


# ===========================================================================
# 2. Data: parse diff -> message tokens + code tokens (train-only vocab)
# ===========================================================================
def parse_commit_text(diff_text):
    t = str(diff_text)
    m = re.search(r"^(diff --git|---\s|\+\+\+\s|@@ )", t, re.M)
    header = t[:m.start()] if m else t[:400]
    body = t[m.start():] if m else ""
    added, removed, msg = [], [], []
    for ln in header.splitlines():
        s = ln.strip()
        if not s or s.startswith(("commit ", "Author:", "Date:", "git-svn-id",
                                  "Merge:", "index ")):
            continue
        msg.append(s)
    for ln in body.splitlines():
        if ln.startswith("+++") or ln.startswith("---"):
            continue
        elif ln.startswith("+"):
            added.append(ln[1:])
        elif ln.startswith("-"):
            removed.append(ln[1:])
    return " ".join(msg), "\n".join(added) + "\n" + "\n".join(removed)


_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\S")


def tokenize(s):
    return _TOK.findall(str(s).lower())[:4000]


def build_vocab(token_lists, vmax=VOCAB_MAX):
    from collections import Counter
    c = Counter()
    for toks in token_lists:
        c.update(toks)
    vocab = {"<pad>": 0, "<unk>": 1}
    for t, _ in c.most_common(vmax - 2):
        vocab[t] = len(vocab)
    return vocab


def encode(toks, vocab, length):
    ids = [vocab.get(t, 1) for t in toks[:length]]
    if len(ids) < length:
        ids += [0] * (length - len(ids))
    return ids


# ===========================================================================
# 3. DeepJIT model: message-CNN + code-CNN -> fusion MLP
# ===========================================================================
class TextCNN(nn.Module):
    def __init__(self, vocab, emb=EMB_DIM, kernels=(1, 2, 3), nfilt=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, nfilt, k, padding=k // 2)
                                    for k in kernels])
        self.out_dim = nfilt * len(kernels)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)               # (B, emb, L)
        feats = [F.relu(c(e)).max(dim=2).values for c in self.convs]
        return torch.cat(feats, dim=1)                # (B, out_dim)


class DeepJIT(nn.Module):
    def __init__(self, vmsg, vcode):
        super().__init__()
        self.msg = TextCNN(vmsg)
        self.code = TextCNN(vcode)
        d = self.msg.out_dim + self.code.out_dim
        self.head = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(128, 1))

    def forward(self, xm, xc):
        return self.head(torch.cat([self.msg(xm), self.code(xc)], dim=1)).squeeze(1)


def train_model(model, Xm, Xc, y, epochs=EPOCHS):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # class-balanced loss (match the class_weight='balanced' spirit of the baselines)
    pos = float(y.sum()); neg = float(len(y) - pos)
    w = torch.tensor([neg / max(pos, 1.0)], device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=w)
    Xm = torch.tensor(Xm, device=DEV); Xc = torch.tensor(Xc, device=DEV)
    yt = torch.tensor(y, dtype=torch.float32, device=DEV)
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for s in range(0, n, BATCH):
            b = perm[s:s + BATCH]
            opt.zero_grad()
            out = model(Xm[b], Xc[b])
            loss = lossf(out, yt[b])
            loss.backward(); opt.step()
    return model


@torch.no_grad()
def predict(model, Xm, Xc):
    model.eval()
    Xm = torch.tensor(Xm, device=DEV); Xc = torch.tensor(Xc, device=DEV)
    out = []
    for s in range(0, len(Xm), 512):
        out.append(torch.sigmoid(model(Xm[s:s + 512], Xc[s:s + 512])).cpu().numpy())
    return np.concatenate(out)


# ===========================================================================
# 4. Online prequential loop (faithful refit) -- mirrors prequential_scores
# ===========================================================================
def run_deepjit(msg_tokens, code_tokens, y):
    N = len(y); W = int(N * WARMUP_FRAC)
    # vocab is fit on the WARM-UP window only (leakage-free), then frozen
    vmsg = build_vocab(msg_tokens[:W]); vcode = build_vocab(code_tokens[:W])
    Xm = np.array([encode(t, vmsg, MSG_LEN) for t in msg_tokens], dtype=np.int64)
    Xc = np.array([encode(t, vcode, CODE_LEN) for t in code_tokens], dtype=np.int64)

    preds = np.full(N, np.nan)
    model = DeepJIT(len(vmsg), len(vcode)).to(DEV)
    model = train_model(model, Xm[:W], Xc[:W], y[:W])
    i, blk = W, 0
    while i < N:
        j = min(N, i + BLOCK); idx = np.arange(i, j)
        preds[idx] = predict(model, Xm[idx], Xc[idx])
        if blk % REFIT_EVERY == 0:                    # retrain on expanding past
            model = DeepJIT(len(vmsg), len(vcode)).to(DEV)
            model = train_model(model, Xm[:j], Xc[:j], y[:j])
        i = j; blk += 1
        print(f"  block {blk:3d}  scored up to commit {j}/{N}", flush=True)
    return preds, W


def main():
    t0 = time.time()
    lab = pd.read_csv(os.path.join(DATA_DIR, f"apache_{PROJECT}.csv"))
    dif = pd.read_csv(os.path.join(DATA_DIR, f"apache_{PROJECT}_diff.csv"))
    # align diff to label order by commit_id, chronological by author_date
    lab = lab.sort_values("author_date").reset_index(drop=True)
    text_by = dict(zip(dif["commit_id"], dif["diff_text"].fillna("")))
    y = lab["buggy"].astype(int).to_numpy()
    msg_tokens, code_tokens = [], []
    for cid in lab["commit_id"]:
        m, c = parse_commit_text(text_by.get(cid, ""))
        msg_tokens.append(tokenize(m)); code_tokens.append(tokenize(c))
    print(f"[{PROJECT}] {len(y)} commits, bug rate {y.mean():.3f}, "
          f"warm-up {int(len(y)*WARMUP_FRAC)}")

    preds, W = run_deepjit(msg_tokens, code_tokens, y)

    ev = np.arange(W, len(y)); yt = y[ev]
    pe = np.nan_to_num(preds[ev], nan=float(yt.mean()))
    metrics = {k: float(v) for k, v in final_metrics(yt, pe).items()}
    traj = metric_trajectory(yt, pe, W, roll=150)
    traj_smooth = metric_trajectory(yt, pe, W, roll=800)

    out = dict(project=PROJECT, model="DeepJIT", N=len(y), warmup=W,
               metrics=metrics, traj=traj, traj_smooth=traj_smooth,
               raw={"idx": (W + np.arange(len(ev))).tolist(),
                    "y": yt.astype(int).tolist(), "pred": pe.astype(float).tolist()},
               config=dict(WARMUP_FRAC=WARMUP_FRAC, BLOCK=BLOCK, REFIT_EVERY=REFIT_EVERY,
                           EPOCHS=EPOCHS, MSG_LEN=MSG_LEN, CODE_LEN=CODE_LEN,
                           EMB_DIM=EMB_DIM))
    path = os.path.join(OUT_DIR, f"deepjit_{PROJECT}_results.pkl")
    pickle.dump(out, open(path, "wb"))
    print("\n=== DeepJIT [{}] (online protocol) ===".format(PROJECT))
    for k in M7:
        print(f"  {k:10} {metrics[k]:.3f}")
    print(f"\nsaved -> {path}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
