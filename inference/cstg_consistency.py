"""
CSTG enrichment: cross-modal INTENT-REALIZATION CONSISTENCY (+ temporal).

The signal: the (in)consistency between what a commit SAYS it does (message
intent, from the text layer) and what it ACTUALLY does (the AST-delta edge mix +
size, from the delta layer). A "minor fix" that rewrites hundreds of nodes, a
"refactor" that is mostly ADDS/REMOVES rather than MOVES, a terse message on a
huge change -- these mismatches are defect signals (cf. Herzig et al. ICSE'13 on
intent-vs-reality mismatch; naturalness/bimodality, Hindle/Allamanis; commit
message quality, Buse & Weimer).

Every feature is computed from a commit's OWN message + OWN delta + OWN timestamp
-- all known at arrival -> O(change-size) per commit, incremental, leakage-free,
online. No global structure, no history recompute (unlike NPMI/co-change).

Public: build(commits, tokens, files, cids) -> (X dense [n, K], feature_names)
"""
import re
from datetime import datetime, timezone
import numpy as np

INTENTS = ["fix", "feat", "refactor", "test", "docs", "perf", "revert"]
INTENT_KW = {
    "fix":      ["fix", "bug", "issue", "error", "fault", "defect", "npe", "crash",
                 "fail", "correct", "resolve", "patch", "wrong", "broken"],
    "feat":     ["add", "feature", "implement", "introduce", "support", "new",
                 "allow", "enable", "provide"],
    "refactor": ["refactor", "cleanup", "clean up", "simplify", "rename", "reorganize",
                 "restructure", "tidy", "inline", "extract", "deprecat"],
    "test":     ["test", "junit", "assert", "testcase", "coverage", "spec"],
    "docs":     ["doc", "documentation", "javadoc", "readme", "comment", "license"],
    "perf":     ["perf", "performance", "optimi", "speed", "faster", "cache", "latency"],
    "revert":   ["revert", "rollback", "roll back", "undo", "back out"],
}
SMALL_KW = ["minor", "small", "trivial", "typo", "tiny", "simple", "cosmetic",
            "nit", "quick", "slight", "little"]

FEATURE_NAMES = ([f"intent_{i}" for i in INTENTS] +
                 ["mm_small_size", "mm_refactor_addrem", "mm_fix_size",
                  "mm_feat_noadds", "mm_terse_bigchange", "is_revert",
                  "grounding_gap"] +
                 ["hour_sin", "hour_cos", "wday_sin", "wday_cos", "is_weekend", "is_latenight"])


def classify_intent(msg):
    m = (msg or "").lower()
    sc = {k: sum(m.count(w) for w in kw) for k, kw in INTENT_KW.items()}
    best = max(sc, key=sc.get)
    return best if sc[best] > 0 else "other"


def _delta_mix(toks):
    """(adds, removes, updates, moves) totals from a commit's AST-change tokens
    'EDGE:ast_type'."""
    a = r = u = mv = 0
    for tok, cnt in toks:
        e = tok.split(":", 1)[0]
        if e == "ADDS": a += cnt
        elif e == "REMOVES": r += cnt
        elif e == "UPDATES": u += cnt
        elif e == "MOVES": mv += cnt
    return a, r, u, mv


def build(commits, tokens, files, cids):
    n = len(cids); K = len(FEATURE_NAMES)
    X = np.zeros((n, K), float)
    for i, c in enumerate(cids):
        msg = str(commits[c].get("message", "") or "")
        ml = msg.lower()
        a, r, u, mv = _delta_mix(tokens.get(c, ()))
        total = a + r + u + mv
        la = float(commits[c].get("la", 0) or 0); ld = float(commits[c].get("ld", 0) or 0)
        intent = classify_intent(msg)
        claims_small = any(w in ml for w in SMALL_KW)
        msg_terms = re.findall(r"[A-Za-z]{2,}", ml)
        terse = len(msg_terms) <= 4
        code_terms = set(re.findall(r"[a-z]+[A-Z]\w*|[A-Z][a-z]+[A-Z]\w*|\w+_\w+", msg))
        # grounding gap: named code entities but the change is tiny/none (talk >> action)
        grounding_gap = np.log1p(len(code_terms)) - np.log1p(total)

        j = 0
        for it in INTENTS:
            X[i, j] = 1.0 if intent == it else 0.0; j += 1
        # --- cross-modal mismatch scores ---
        X[i, j] = (1.0 if claims_small else 0.0) * np.log1p(total); j += 1  # small-word vs big delta
        X[i, j] = (1.0 if intent == "refactor" else 0.0) * ((a + r) / (total + 1)); j += 1  # refactor should MOVE, not add/remove
        X[i, j] = (1.0 if intent == "fix" else 0.0) * np.log1p(total); j += 1  # fix should be small
        X[i, j] = (1.0 if intent == "feat" else 0.0) * (1.0 - a / (total + 1)); j += 1  # feat with few ADDS
        X[i, j] = (1.0 if terse else 0.0) * np.log1p(la + ld); j += 1  # under-described big change
        X[i, j] = 1.0 if intent == "revert" else 0.0; j += 1
        X[i, j] = grounding_gap; j += 1
        # --- temporal (Eyolfson et al.): cyclical hour/weekday + late-night/weekend ---
        ts = commits[c].get("ts", 0) or 0
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        hr, wd = dt.hour, dt.weekday()
        X[i, j] = np.sin(2 * np.pi * hr / 24); j += 1
        X[i, j] = np.cos(2 * np.pi * hr / 24); j += 1
        X[i, j] = np.sin(2 * np.pi * wd / 7); j += 1
        X[i, j] = np.cos(2 * np.pi * wd / 7); j += 1
        X[i, j] = 1.0 if wd >= 5 else 0.0; j += 1
        X[i, j] = 1.0 if 0 <= hr < 5 else 0.0; j += 1
    return X, FEATURE_NAMES


def intent_of(commits, cids):
    """Convenience: {cid: intent} for KG ingestion."""
    return {c: classify_intent(str(commits[c].get("message", "") or "")) for c in cids}
