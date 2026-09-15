"""
Driver: cross-project (general) evaluation over ALL built projects.

Runs, in order:
  1. aggregate/aggregate_projects.py       -> outputs/aggregate/aggregate_results.pkl
  2. aggregate/make_aggregate_tables.py    -> outputs/aggregate/tables/*.{tex,csv}
  3. aggregate/build_aggregate_notebook.py -> outputs/aggregate/notebooks/*.ipynb (executed)

Unlike the per-project drivers, this needs NO KGC_PROJECT: it scans every
project under outputs/ that has result files, aggregates whatever is present,
and skips the rest. Re-run it after building each new project to refresh the
"Total_Results" with the new project folded in.

Run:  python drivers/run_aggregate.py
      python drivers/run_aggregate.py --only activemq kafka   # restrict set
"""
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
AGG = PKG / "aggregate"


def _run(script, *args):
    print(f"\n[{'='*3}] {script.name} {' '.join(args)}")
    subprocess.run([sys.executable, str(script), *args], check=True)


def main():
    # pass through any --only ... args to the aggregator step
    passthru = sys.argv[1:]
    _run(AGG / "aggregate_projects.py", *passthru)
    _run(AGG / "make_aggregate_tables.py")
    _run(AGG / "make_aggregate_figures.py")
    _run(AGG / "build_aggregate_notebook.py")
    print("\nAGGREGATE COMPLETE -> outputs/aggregate/ "
          "(aggregate_results.pkl, tables/, figures/, notebooks/).")


if __name__ == "__main__":
    main()
