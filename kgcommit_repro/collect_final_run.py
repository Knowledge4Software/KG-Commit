"""
Collect every artifact of the final run into outputs/<project>/final_run/.
==========================================================================

The pipeline writes results wherever each stage happens to put them (the project
root, scalability/, tables/v4/, figures/v4/, param_experiments/, ...). This script
gathers the ones that constitute the FINAL RUN into a single, self-describing tree
so the paper can be rebuilt -- or a float redesigned -- without touching Neo4j, and
so the run can be archived as one unit.

Nothing is moved: files are COPIED, leaving the working layout intact.

Also writes final_run/config.json (the frozen protocol constants) and
final_run/MANIFEST.json (what was collected, with sizes and mtimes), so a reader can
tell exactly which configuration produced which artifact.

Run:  KGC_PROJECT=groovy python collect_final_run.py
      python collect_final_run.py --all          # every project in protocol.PROJECTS
      python collect_final_run.py --all --dry-run
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "inference"))
import _kgc_paths  # noqa: E402,F401
import protocol as P  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
LOGS = Path(__file__).resolve().parent / "logs"

# destination subdir -> list of source paths relative to outputs/<project>/
# (glob patterns allowed; missing sources are reported, not fatal)
LAYOUT = {
    "fusion": [
        "final_fusion_results.pkl",
        "raw_fusion_scores.pkl", "raw_fusion_scores.csv",
        "raw_fusion_scores_overall.pkl", "raw_fusion_scores_overall.csv",
    ],
    "experiments": [
        "final_experiments_results.pkl",
        "raw_method_scores.pkl", "raw_method_scores.csv",
        "subgraph_rq_results.pkl",
        "subgraph_kg_methods_results.pkl",
        "subgraph_layer_stats.json",
    ],
    "baselines": [
        "baseline_results.pkl",
        "baseline_extra_results.pkl",
        "effort_results.pkl",
    ],
    "complexity": [
        "scalability/*.json",
        "scalability/*.tex",
    ],
    "params": [
        "param_experiments/*.json",
    ],
    "seeds": [
        "final_run/seeds/*",        # written in place by run_seed_robustness.py
    ],
    "streams": [
        "online_jit_streams_v5.pkl",
        "kg_stream_cache.pkl",
    ],
    "tables": ["tables/v4/*"],
    "figures": ["figures/v4/**/*"],
}

# Extracted for insurance but deliberately NOT part of the paper (see the audit):
AST_METHOD = ["v4_ast_method/**/*"]


def _iter_sources(base, patterns):
    for pat in patterns:
        if any(ch in pat for ch in "*?["):
            yield from sorted(base.glob(pat))
        else:
            p = base / pat
            if p.exists():
                yield p


def collect(project, dry=False, include_streams=False):
    base = OUTPUTS / project
    if not base.is_dir():
        print(f"### {project}: no outputs dir -- skipped")
        return None
    dest = base / "final_run"
    manifest = {"project": project,
                "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "config": P.as_dict(), "items": []}
    n_ok = n_miss = 0

    print(f"\n### {project}")
    for sub, pats in LAYOUT.items():
        if sub == "streams" and not include_streams:
            continue            # large caches; opt in with --include-streams
        if sub == "seeds":
            # already written directly into final_run/seeds/ -- report, don't copy
            n = len([f for f in (dest / "seeds").glob("*") if f.is_file()]) \
                if (dest / "seeds").is_dir() else 0
            print(f"   [{'ok' if n else '--'}] {sub:<12} {n:>4} files  (in place)")
            n_ok += bool(n); n_miss += (not n)
            continue
        got = list(_iter_sources(base, pats))
        if not got:
            print(f"   [--] {sub:<12} (nothing found)")
            n_miss += 1
            continue
        outdir = dest / sub
        tot = 0
        for src in got:
            if src.is_dir():
                continue
            # Never copy out of final_run/ into itself. `seeds` is written directly
            # into final_run/seeds/ by run_seed_robustness.py, so it is already in
            # place -- copying it would be a same-file copy (SameFileError) and,
            # because that raised mid-loop, it previously aborted the whole
            # collection and silently lost tables/ and figures/.
            if dest in src.parents:
                continue
            rel = src.relative_to(base)
            # keep the tail structure for nested trees (tables/figures)
            target = outdir / rel.name if len(rel.parts) <= 2 else \
                outdir / Path(*rel.parts[2:])
            if not dry:
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if src.resolve() != target.resolve():
                        shutil.copy2(src, target)
                except (shutil.SameFileError, OSError) as e:
                    print(f"      skip {rel}: {type(e).__name__}")
                    continue
            st = src.stat()
            tot += st.st_size
            manifest["items"].append(
                {"dest": str((Path(sub) / target.name)), "src": str(rel),
                 "bytes": st.st_size,
                 "mtime": time.strftime("%Y-%m-%d %H:%M",
                                        time.localtime(st.st_mtime))})
        print(f"   [ok] {sub:<12} {len(got):>4} files  {tot/1e6:8.1f} MB")
        n_ok += 1

    # ast_method: extracted for potential revision questions, not for the paper
    am = list(_iter_sources(base, AST_METHOD))
    if am:
        outdir = dest / "ast_method"
        for src in am:
            if src.is_dir():
                continue
            rel = src.relative_to(base / "v4_ast_method")
            if not dry:
                (outdir / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, outdir / rel)
        print(f"   [ok] ast_method   {len(am):>4} files  (not for the paper)")

    if not dry:
        dest.mkdir(parents=True, exist_ok=True)
        json.dump(P.as_dict(), open(dest / "config.json", "w"), indent=2)
        json.dump(manifest, open(dest / "MANIFEST.json", "w"), indent=2)
    print(f"   -> {dest}  ({n_ok} groups collected, {n_miss} empty)")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-streams", action="store_true",
                    help="also copy the large feature caches")
    a = ap.parse_args()

    projects = P.PROJECTS if a.all else [os.environ.get("KGC_PROJECT")]
    if not projects or projects == [None]:
        sys.exit("set KGC_PROJECT or pass --all")

    for p in projects:
        collect(p, dry=a.dry_run, include_streams=a.include_streams)
    if a.dry_run:
        print("\n(dry run -- nothing copied)")


if __name__ == "__main__":
    main()
