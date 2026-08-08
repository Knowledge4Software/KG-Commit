"""
Master driver: run every C2 (cache-only) paper-material generator for one setting.
Usage:  python inference/make_all_c2.py [--gap 0|50] [--warmup 0.40|0.20]
gap=0/warmup=0.40 = Setting A (default); gap=50/warmup=0.20 = Setting B.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent

GEN_GAPAWARE = ["make_rq1_tables.py", "make_appendix_CD.py"]     # take --gap/--warmup
GEN_SETTINGA = ["make_rq3_matrix.py", "make_rq4_tables.py",       # Setting-A only
                "make_rq1_figures.py", "make_rq3_figures.py", "make_rq1_streams.py"]


def run(script, args, env):
    cmd = [sys.executable, str(HERE / script)] + args
    print(f"\n=== {script} {' '.join(args)} ===", flush=True)
    r = subprocess.run(cmd, env=env)
    print(f"  [{script}] exit {r.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=0.40)
    args = ap.parse_args()
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(PKG), str(HERE), str(PKG / "baselines")]) + \
        os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("KGC_PROJECT", "kafka")   # generators iterate all projects; env just to satisfy config

    gapargs = ["--gap", str(args.gap), "--warmup", str(args.warmup)]
    for s in GEN_GAPAWARE:
        run(s, gapargs, env)
    if args.gap == 0 and abs(args.warmup - 0.40) < 1e-9:
        for s in GEN_SETTINGA:
            run(s, [], env)
    else:
        print("\n(Setting B: gap-aware tables regenerated; figures/matrices are "
              "Setting-A shared or need gap-aware extension -- streams/ROC at gap only "
              "affect threshold, ROC identical; matrices are representation-level.)")
    print("\nC2 pass complete.")


if __name__ == "__main__":
    main()
