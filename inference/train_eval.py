"""
Light, CPU-only buggy/benign classification on KG commit features.

Mimics the ONLINE setting with a time-ordered split (train on older commits,
test on newer ones -- no look-ahead leakage). Compares three feature sets to
show whether the KG's delta-graph features add value over the classic JIT
metrics:
    metrics  - the 12 ApacheJIT change metrics
    delta    - the AST delta-graph features only
    combined - both

Models (all CPU, fast): Logistic Regression, Random Forest, HistGradientBoosting.
Reports ROC-AUC, PR-AUC (avg precision), F1, precision, recall, MCC at a 0.5
threshold, plus class balance. Also prints top feature importances.

Run:  python inference/train_eval.py
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, matthews_corrcoef,
                             balanced_accuracy_score)

CSV=Path("outputs/commit_features.csv")
METRICS=["la","ld","nf","nd","ns","ent","ndev","age","nuc","aexp","arexp","asexp"]
DELTA=["n_adds","n_removes","n_updates","n_moves","delta_total","n_delta_files",
       "n_changed_types","add_remove_ratio",
       "grp_declaration","grp_statement","grp_expression","grp_literal","grp_leaf"]
TEST_FRAC=0.30   # last 30% of commits (by time) = test

def models():
    return {
        "LogReg": make_pipeline(StandardScaler(),
                  LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1,
                  class_weight="balanced", random_state=0),
        "HistGBoost": HistGradientBoostingClassifier(max_iter=300, random_state=0,
                  class_weight="balanced"),
    }

def evaluate(y, p):
    yhat=(p>=0.5).astype(int)
    return dict(
        ROC_AUC=roc_auc_score(y,p), PR_AUC=average_precision_score(y,p),
        F1=f1_score(y,yhat,zero_division=0), Prec=precision_score(y,yhat,zero_division=0),
        Recall=recall_score(y,yhat,zero_division=0), MCC=matthews_corrcoef(y,yhat),
        BalAcc=balanced_accuracy_score(y,yhat))

def main():
    df=pd.read_csv(CSV).sort_values("author_ts").reset_index(drop=True)
    # only evaluate where delta features exist if we want a fair delta test;
    # but keep all for metrics. Report coverage.
    n=len(df); cut=int(n*(1-TEST_FRAC))
    tr,te=df.iloc[:cut], df.iloc[cut:]
    print(f"Commits: {n}  train={len(tr)} test={len(te)} (time-ordered split)")
    print(f"  train bug rate={tr.buggy.mean():.3f}  test bug rate={te.buggy.mean():.3f}")
    cov=(df.delta_total>0).mean()
    print(f"  delta-feature coverage: {cov:.0%} "
          f"({'partial build' if cov<0.95 else 'full'})\n")

    sets={"metrics":METRICS, "delta":DELTA, "combined":METRICS+DELTA}
    ytr, yte = tr.buggy.values, te.buggy.values

    header=f"{'feature set':<10} {'model':<14} {'ROC_AUC':>7} {'PR_AUC':>7} {'F1':>6} {'Prec':>6} {'Recall':>7} {'MCC':>6}"
    print(header); print("-"*len(header))
    best=None
    for sname,cols in sets.items():
        Xtr=tr[cols].fillna(0).values; Xte=te[cols].fillna(0).values
        for mname,model in models().items():
            model.fit(Xtr,ytr)
            p=model.predict_proba(Xte)[:,1]
            m=evaluate(yte,p)
            print(f"{sname:<10} {mname:<14} {m['ROC_AUC']:7.3f} {m['PR_AUC']:7.3f} "
                  f"{m['F1']:6.3f} {m['Prec']:6.3f} {m['Recall']:7.3f} {m['MCC']:6.3f}")
            if best is None or m["PR_AUC"]>best[0]:
                best=(m["PR_AUC"], sname, mname, cols, model)
        print()

    # feature importance from the best tree model on the combined set
    _,bs,bm,bcols,bmodel=best
    print(f"Best by PR-AUC: {bs} / {bm}")
    try:
        Xtr=tr[bcols].fillna(0).values
        rf=RandomForestClassifier(n_estimators=300,n_jobs=-1,
            class_weight="balanced",random_state=0).fit(Xtr,ytr)
        imp=sorted(zip(bcols,rf.feature_importances_),key=lambda x:-x[1])[:12]
        print("Top features (RandomForest importance):")
        for f,v in imp: print(f"  {f:<20} {v:.3f}")
    except Exception as e:
        print("  (importance skipped:",e,")")

    # naive baselines for context
    base_rate=yte.mean()
    print(f"\nBaselines: always-benign acc={1-base_rate:.3f}; "
          f"random PR-AUC ~= base rate {base_rate:.3f}")

if __name__=="__main__":
    main()
