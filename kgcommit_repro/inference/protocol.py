"""
Single source of truth for the online-protocol constants (FINAL RUN).
=====================================================================

Every module that learns, refits, refreshes or scores on a schedule imports its
constants from HERE. Before this module the same knobs were declared independently
in six places and had drifted apart (WARMUP_FRAC was 0.30 in online_jit.py and 0.40
in online_infer.py; the refit cadence was 3, 4, 5, 6 or 8 blocks depending on the
file). That made the deployed configuration impossible to state in one sentence.

THE SINGLE-M RULE
-----------------
Everything refreshes on exactly ONE cadence: once per BLOCK (M) commits. Nothing
refits on a multiple of M any more. This applies to:

  * the fusion / tree heads            (was REFIT_EVERY = 3 or 4 blocks)
  * the SVD embedding                  (was SVD_REFRESH = 5 or 8 blocks)
  * DeepWalk + KGE embeddings          (was REFIT_EMB   = 5 or 6 blocks)
  * the CSTG NPMI propagation          (was PROP_REFRESH = 400 commits = 2M)
  * every baseline that refits online  (incl. JITLine_fully_online)

so REFIT_EVERY = 1 means "every block", i.e. every M commits.

IMPORTANT FOR RQ2 COST
----------------------
The cost estimators amortise a refit over the commits it serves, dividing by
(REFIT_EVERY * BLOCK). They must import REFIT_EVERY from here rather than hardcode
it -- a stale literal (5) would understate the refit term by 5x.

Overriding for the parameter sweeps
-----------------------------------
run_param_experiments.py deliberately sweeps K, M and ROLL. It passes explicit
values into the run functions; it does not mutate this module.
"""

# ── protocol ────────────────────────────────────────────────────────────────
WARMUP_FRAC = 0.05      # K -- initial-fit fraction (commits 0..W are not scored)
BLOCK = 200             # M -- learn/refresh granularity
GAP = 50                # G -- commits skipped between fitting and scoring
ROLL = 800              # rolling-metric window (visualisation only)

# ── the single refresh cadence (in BLOCKs) ──────────────────────────────────
REFIT_EVERY = 1         # every component refits once per BLOCK. Do not fork this.

# ── stacking-head initialisation window ─────────────────────────────────────
# The fusion head is a stacker: it needs a labelled window AFTER the warm-up before
# it can score anything. This used to be a flat INIT = 300 commits, which quietly
# defeated K = 0.05 -- on zookeeper a 41-commit warm-up was followed by a 300-commit
# dead zone, so "score almost immediately" was false in practice.
#
# It is now ADAPTIVE: advance from INIT_FLOOR in steps of INIT_STEP until the window
# holds at least INIT_MIN_MINORITY examples of the rarer class, then stop; never go
# past INIT_CAP. Projects with a healthy bug rate start scoring almost at once
# (zookeeper/spark/kafka/cassandra/camel at the floor), while a severely imbalanced
# project like flink still gets the window it needs rather than a degenerate fit.
INIT_FLOOR = 50           # earliest the head may start
INIT_CAP = 300            # previous flat value, now only an upper bound
INIT_STEP = 10
INIT_MIN_MINORITY = 10    # minimum rarer-class examples before the head is fit


def init_window(y, W, floor=INIT_FLOOR, cap=INIT_CAP, step=INIT_STEP,
                min_minority=INIT_MIN_MINORITY):
    """Smallest init >= floor whose window [W, W+init) holds >= min_minority of the
    rarer class; capped at `cap`. Deterministic and label-only -- it looks at the
    warm-up-adjacent window the head is entitled to train on, never at the future."""
    import numpy as _np
    y = _np.asarray(y)
    init = floor
    while init < cap:
        seg = y[W:W + init]
        if seg.size:
            pos = int(seg.sum())
            if min(pos, int(seg.size - pos)) >= min_minority:
                break
        init += step
    return min(init, cap)

# ── representation sizes ────────────────────────────────────────────────────
HASH_DIM = 2 ** 18
SVD_DIM = 64
DW_DIM = 64
KGE_DIM = 32

# ── trajectory rendering (visualisation only; never affects metrics) ────────
TRAJ_STRIDE = 25
TRAJ_WINDOW = 150         # the FINAL reported resolution
TRAJ_WINDOW_SMOOTH = 800  # kept for the smoothed companion view

# ── learners ────────────────────────────────────────────────────────────────
CLASS_WEIGHT = "balanced"   # every classifier, no exceptions
RANDOM_STATE = 0

# ── seed-robustness check (light: fusion head + DW/KGE are the stochastic parts)
SEEDS = [0, 1, 2, 3, 4]

# ── fusion ──────────────────────────────────────────────────────────────────
# The overall-best combination, fixed across all projects. Chosen by macro-mean
# Macro-F1 over the 11 projects (Macro-F1 is this work's primary metric).
# Per-project best F is computed separately, per project, at run time.
OVERALL_F = ("RN", "PPR")

PROJECTS = ["activemq", "camel", "cassandra", "flink", "groovy", "hbase",
            "hive", "kafka", "spark", "zeppelin", "zookeeper"]


def as_dict():
    """The frozen configuration, for provenance (written to final_run/config.json)."""
    return {
        "WARMUP_FRAC": WARMUP_FRAC, "BLOCK": BLOCK, "GAP": GAP, "ROLL": ROLL,
        "REFIT_EVERY": REFIT_EVERY, "HASH_DIM": HASH_DIM, "SVD_DIM": SVD_DIM,
        "DW_DIM": DW_DIM, "KGE_DIM": KGE_DIM,
        "TRAJ_STRIDE": TRAJ_STRIDE, "TRAJ_WINDOW": TRAJ_WINDOW,
        "TRAJ_WINDOW_SMOOTH": TRAJ_WINDOW_SMOOTH,
        "CLASS_WEIGHT": CLASS_WEIGHT, "RANDOM_STATE": RANDOM_STATE,
        "SEEDS": list(SEEDS), "OVERALL_F": list(OVERALL_F),
        "PROJECTS": list(PROJECTS),
    }
