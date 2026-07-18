"""
Snapshot / restore the Neo4j graph per project -- so building a NEW project never
destroys a previously-built one, and no graph is ever rebuilt from scratch twice.

Neo4j Community edition allows only ONE user database, so projects cannot live in
separate databases simultaneously. Instead we DUMP the current database to a
compact file per project and LOAD it back on demand:

  * dump    : save the currently-resident graph to
              outputs/<project>/neo4j_dump/<project>.dump   (do this before you
              wipe/switch away from a built project)
  * restore : load <project>.dump back into Neo4j            (seconds/minutes, vs.
              a multi-hour rebuild) -- makes <project> the resident graph again
  * list    : show which projects have a saved snapshot and their sizes

Mechanism: the offline `neo4j-admin database dump/load`. On Neo4j COMMUNITY edition
a single database cannot be stopped while the server runs (`STOP DATABASE` is
Enterprise-only), so we stop the whole Neo4j container, run `neo4j-admin` in a
transient one-off container that mounts the SAME data volume, then start the main
container again. The dump lands on the host /data bind mount and is copied into
this package's per-project outputs. The interruption is a brief container
stop/start -- minutes, not the multi-hour rebuild it replaces.

Typical multi-project workflow (no rebuilds, nothing lost):
    # finished with groovy, want to build zeppelin:
    KGC_PROJECT=groovy   python snapshot_neo4j.py dump
    KGC_PROJECT=zeppelin python reset_neo4j.py --yes
    KGC_PROJECT=zeppelin python drivers/build_project.py   ... etc ...
    KGC_PROJECT=zeppelin python snapshot_neo4j.py dump
    # come back to groovy later WITHOUT rebuilding:
    KGC_PROJECT=groovy   python snapshot_neo4j.py restore

Configuration (override via env if your setup differs):
    KGC_NEO4J_CONTAINER   docker container name        (default: neo4j-local)
    KGC_NEO4J_DB          database name                (default: neo4j)
    KGC_NEO4J_DATADIR     host path bind-mounted to /data
                          (default: auto-detected from `docker inspect`)
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import _kgc_paths  # noqa: F401  (adds package dirs to sys.path)
from config.project_config import PROJECT, OUT

CONTAINER = os.environ.get("KGC_NEO4J_CONTAINER", "neo4j-local")
DB = os.environ.get("KGC_NEO4J_DB", "neo4j")
IMAGE = os.environ.get("KGC_NEO4J_IMAGE", "neo4j:latest")
STAGE = "_kgc_dumps"                               # subdir under /data for staging


def _sh(*args, check=True, capture=False, env=None):
    r = subprocess.run(list(args), text=True, capture_output=capture, env=env)
    if check and r.returncode != 0:
        sys.exit(f"command failed ({r.returncode}): {' '.join(args)}\n"
                 f"{(r.stderr or '') if capture else ''}")
    return r


def _normalise_win_mount(src: str) -> str:
    r"""Translate a Docker-Desktop WSL-internal mount source back to a real host
    path. After a container is (re)created, `docker inspect` may report the /data
    bind mount as e.g. \run\desktop\mnt\host\c\Users\... (Docker Desktop's internal
    view) which Python cannot open. Recover the true Windows path from the
    '...\mnt\host\<drive>\...' tail."""
    if not src:
        return src
    low = src.replace("/", "\\").lower()
    marker = "\\mnt\\host\\"
    i = low.find(marker)
    if i != -1:
        tail = src.replace("/", "\\")[i + len(marker):]      # e.g. c\Users\sinab\...
        drive, _, rest = tail.partition("\\")
        return f"{drive.upper()}:\\{rest}"
    return src


def _host_datadir() -> Path:
    """Host path bind-mounted to /data in the container (where dumps stage)."""
    env = os.environ.get("KGC_NEO4J_DATADIR")
    if env:
        return Path(env)
    fmt = '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Source}}{{end}}{{end}}'
    r = _sh("docker", "inspect", CONTAINER, "--format", fmt, capture=True)
    src = _normalise_win_mount((r.stdout or "").strip())
    p = Path(src) if src else None
    if not p or not p.exists():
        sys.exit(f"Could not resolve the host /data bind mount (got {src!r}); set "
                 f"KGC_NEO4J_DATADIR to the host directory mounted at /data.")
    return p


def _docker_vol(host_datadir: Path) -> str:
    """Docker -v spec for the data dir, MSYS-safe on Windows (//c/... : //data)."""
    p = str(host_datadir).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":            # C:/... -> //c/...
        p = "//" + p[0].lower() + p[2:]
    return f"{p}://data"


def _admin(host_datadir: Path, *admin_args):
    """Run neo4j-admin in a transient container (main server must be stopped)."""
    env = dict(os.environ); env["MSYS_NO_PATHCONV"] = "1"
    _sh("docker", "run", "--rm", "--user", "neo4j",
        "-v", _docker_vol(host_datadir), IMAGE, "neo4j-admin", *admin_args, env=env)


def _stop_server():
    # Graceful stop with a generous timeout so Neo4j checkpoints and clears its
    # logical log before shutdown -- otherwise the offline dump refuses to run
    # ("Active logical log detected ... Please recover database"). 5 min is ample
    # even for multi-million-node databases.
    print(f"  stopping Neo4j container '{CONTAINER}' (graceful, up to 300s) ...")
    _sh("docker", "stop", "-t", "300", CONTAINER, capture=True)


def _start_server():
    print(f"  starting Neo4j container '{CONTAINER}' ...")
    _sh("docker", "start", CONTAINER, capture=True)


def _dump_path_host(project: str) -> Path:
    return OUT / "neo4j_dump" / f"{project}.dump"


def cmd_dump(_args):
    dst = _dump_path_host(PROJECT)
    dst.parent.mkdir(parents=True, exist_ok=True)
    datadir = _host_datadir()
    (datadir / STAGE).mkdir(parents=True, exist_ok=True)
    print(f"Dumping resident Neo4j graph as snapshot for '{PROJECT}' ...")
    _stop_server()
    try:
        _admin(datadir, "database", "dump", DB,
               f"--to-path=//data/{STAGE}", "--overwrite-destination=true")
    finally:
        _start_server()
    produced = datadir / STAGE / f"{DB}.dump"
    if not produced.exists():
        sys.exit(f"expected dump not found at {produced}")
    shutil.copy2(produced, dst)
    print(f"snapshot saved: {dst}  ({dst.stat().st_size/1e6:.1f} MB)")


def cmd_restore(_args):
    src = _dump_path_host(PROJECT)
    if not src.exists():
        sys.exit(f"No snapshot for '{PROJECT}' at {src}. "
                 f"(Dump it first with `snapshot_neo4j.py dump`.)")
    datadir = _host_datadir()
    (datadir / STAGE).mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, datadir / STAGE / f"{DB}.dump")     # stage where admin can read it
    print(f"Restoring '{PROJECT}' snapshot into Neo4j (resident graph replaced) ...")
    _stop_server()
    try:
        _admin(datadir, "database", "load", DB,
               f"--from-path=//data/{STAGE}", "--overwrite-destination=true")
    finally:
        _start_server()
    print(f"'{PROJECT}' is now the resident Neo4j graph. No rebuild needed.")


def cmd_list(_args):
    root = OUT.parent                       # outputs/
    print("Saved Neo4j snapshots (outputs/<project>/neo4j_dump/):")
    any_found = False
    for d in sorted(root.glob("*/neo4j_dump/*.dump")):
        any_found = True
        proj = d.parent.parent.name
        mb = d.stat().st_size / 1e6
        print(f"  {proj:<18} {d.name:<24} {mb:8.1f} MB   {d}")
    if not any_found:
        print("  (none yet)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dump", help="save the resident graph as this project's snapshot")
    sub.add_parser("restore", help="load this project's snapshot into Neo4j")
    sub.add_parser("list", help="list saved snapshots")
    args = ap.parse_args()
    {"dump": cmd_dump, "restore": cmd_restore, "list": cmd_list}[args.cmd](args)


if __name__ == "__main__":
    main()
