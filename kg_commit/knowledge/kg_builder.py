"""
Knowledge Graph Builder with tiered entity support.

Builds and maintains a directed graph representing commits, files, authors,
functions, classes, issues, and their relationships.

Supports:
- Multi-tier entity model (Core, Author, File, Within-file, Semantic)
- Interval nodes tracking entity lifecycles
- Node counters for aggregated statistics
- Customizable entity extraction
"""

from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import networkx as nx
import pandas as pd

from kg_commit.knowledge.parsers import CommitParser


class IntervalManager:
    """Manages interval nodes for tracking entity lifecycles."""

    def __init__(self):
        """Initialize interval manager."""
        self._itvl_seq = [0]

    def create_interval(self, graph: nx.MultiDiGraph, entity_id: str, ts: float) -> str:
        """
        Create interval node for an entity.
        
        Parameters
        ----------
        graph : nx.MultiDiGraph
            Knowledge graph
        entity_id : str
            Entity node ID
        ts : float
            Start timestamp (Unix seconds)
            
        Returns
        -------
        str
            Interval node ID
        """
        self._itvl_seq[0] += 1
        iid = f"itvl:{self._itvl_seq[0]}"
        graph.add_node(iid, type="INTERVAL", begin=float(ts), end=float("inf"))
        graph.add_edge(entity_id, iid, rel="has_interval")
        graph.nodes[entity_id]["_cur_itvl"] = iid
        return iid

    def update_interval_end(self, graph: nx.MultiDiGraph, entity_id: str, ts: float):
        """
        Update the end time of an entity's interval.
        
        Parameters
        ----------
        graph : nx.MultiDiGraph
            Knowledge graph
        entity_id : str
            Entity node ID
        ts : float
            New end timestamp
        """
        cur = graph.nodes[entity_id].get("_cur_itvl")
        if cur and graph.has_node(cur):
            graph.nodes[cur]["end"] = float(ts)


class KnowledgeGraphBuilder:
    """
    Builds a tiered knowledge graph from commit data.
    
    Tiers:
    1. Core: COMMIT, TIME, INTERVAL, LABEL
    2. Author: AUTHOR, AUTHOR_INTERVAL
    3. File: FILE, DIR, FILE_TYPE, EXTERNAL_PACKAGE
    4. Within-file: CLASS, FUNCTION, FUNCTION_SIGNATURE, VARIABLE, DATA_TYPE
    5. Semantic: ISSUE, BRANCH
    """

    def __init__(self, enable_tiers: Optional[List[int]] = None):
        """
        Initialize builder.
        
        Parameters
        ----------
        enable_tiers : list of int, optional
            Which tiers to enable (1-5). Default: all
        """
        self.enable_tiers = enable_tiers or [1, 2, 3, 4, 5]
        self.parser = CommitParser()
        self.intervals = IntervalManager()
        self._branch_cache = {}

    def build(
        self,
        commits_data: List[dict],
        diff_map: Optional[Dict[str, str]] = None,
        commit_issue_map: Optional[Dict[str, List[str]]] = None,
        issue_meta: Optional[Dict[str, dict]] = None,
    ) -> nx.MultiDiGraph:
        """
        Build complete KG from commit list.
        
        Parameters
        ----------
        commits_data : list of dict
            Commit records with keys: commit_id, project, author_date, buggy, etc.
        diff_map : dict, optional
            {commit_id: diff_text}
        commit_issue_map : dict, optional
            {commit_id: [issue_ids]}
        issue_meta : dict, optional
            {issue_id: {title, description, status, ...}}
            
        Returns
        -------
        nx.MultiDiGraph
            Knowledge graph
        """
        G = nx.MultiDiGraph()
        diff_map = diff_map or {}
        commit_issue_map = commit_issue_map or {}
        issue_meta = issue_meta or {}
        prev_cid = None

        for commit_data in commits_data:
            diff = diff_map.get(commit_data.get("commit_id"))
            self._add_commit(
                G,
                commit_data,
                diff,
                prev_cid,
                commit_issue_map,
                issue_meta,
            )
            prev_cid = f"commit:{commit_data.get('commit_id')}"

        return G

    def _add_commit(
        self,
        G: nx.MultiDiGraph,
        row: dict,
        diff: Optional[str],
        prev_cid: Optional[str],
        commit_issue_map: Dict,
        issue_meta: Dict,
    ):
        """Add one commit's knowledge to graph."""
        cid = f"commit:{row['commit_id']}"
        ts = float(row.get("author_date", 0))
        
        # Parse diff
        parsed = self.parser.parse(diff)

        # ── TIER 1: Core ──────────────────────────────────────────────
        if 1 in self.enable_tiers:
            self._add_tier_core(G, row, cid, ts, prev_cid)

        # ── TIER 2: Author ────────────────────────────────────────────
        if 2 in self.enable_tiers:
            email = self._get_author(row, parsed)
            self._add_tier_author(G, cid, ts, email, row)

        # ── TIER 3: Files ─────────────────────────────────────────────
        if 3 in self.enable_tiers:
            self._add_tier_files(G, cid, ts, parsed, email, row)

        # ── TIER 4: Within-file ───────────────────────────────────────
        if 4 in self.enable_tiers:
            self._add_tier_within_file(G, cid, ts, parsed)

        # ── TIER 5: Semantic ──────────────────────────────────────────
        if 5 in self.enable_tiers:
            self._add_tier_semantic(
                G, cid, row, parsed, commit_issue_map, issue_meta
            )

    def _add_tier_core(
        self,
        G: nx.MultiDiGraph,
        row: dict,
        cid: str,
        ts: float,
        prev_cid: Optional[str],
    ):
        """Add core tier: COMMIT, TIME, INTERVAL, LABEL."""
        # COMMIT node
        G.add_node(
            cid,
            type="COMMIT",
            commit_id=row.get("commit_id"),
            project=row.get("project", "unknown"),
            year=int(row.get("year", 0)),
            author_date=ts,
        )
        self.intervals.create_interval(G, cid, ts)

        # TIME node
        tid = f"time:{int(ts)}"
        if not G.has_node(tid):
            G.add_node(tid, type="TIME", datetime=ts)
        G.add_edge(cid, tid, rel="at_time")

        # Precedence edge
        if prev_cid:
            G.add_edge(prev_cid, cid, rel="precedes")

        # LABEL node
        status = -1 if row.get("buggy") else (1 if row.get("fix") else 0)
        lid = f"label:{status}"
        if not G.has_node(lid):
            G.add_node(lid, type="LABEL", status=status)
        G.add_edge(cid, lid, rel="is")

    def _add_tier_author(
        self,
        G: nx.MultiDiGraph,
        cid: str,
        ts: float,
        email: str,
        row: dict,
    ):
        """Add author tier: AUTHOR with counters."""
        aid = f"author:{email}"
        if not G.has_node(aid):
            G.add_node(
                aid,
                type="AUTHOR",
                email=email,
                commit_count=0,
                bug_count=0,
            )
        
        G.add_edge(cid, aid, rel="by")
        G.nodes[aid]["commit_count"] += 1
        if row.get("buggy"):
            G.nodes[aid]["bug_count"] += 1

    def _add_tier_files(
        self,
        G: nx.MultiDiGraph,
        cid: str,
        ts: float,
        parsed: dict,
        email: str,
        row: dict,
    ):
        """Add file tier: FILE, DIR, FILE_TYPE, EXTERNAL_PACKAGE."""
        files = parsed.get("files", [])

        for filepath, change_type in files:
            p = Path(filepath)
            fid = f"file:{filepath}"
            ftid = f"filetype:{p.suffix or 'none'}"

            if not G.has_node(fid):
                G.add_node(
                    fid,
                    type="FILE",
                    name=p.name,
                    path=filepath,
                    change_count=0,
                    bug_count=0,
                    authors=set(),
                )
                self.intervals.create_interval(G, fid, ts)
            else:
                self.intervals.update_interval_end(G, fid, ts)

            G.add_edge(cid, fid, rel=change_type)
            G.nodes[fid]["change_count"] += 1
            if row.get("buggy"):
                G.nodes[fid]["bug_count"] += 1
            G.nodes[fid]["authors"].add(email)

            # FILE_TYPE
            if not G.has_node(ftid):
                G.add_node(ftid, type="FILE_TYPE", format=p.suffix or "none")
            G.add_edge(fid, ftid, rel="type")

            # Directory hierarchy
            parents = list(reversed(list(p.parents)))
            for i, part in enumerate(parents):
                if str(part) in (".", ""):
                    continue
                did = f"dir:{part}"
                if not G.has_node(did):
                    G.add_node(did, type="DIR", name=part.name, path=str(part))
                
                if i == len(parents) - 1:
                    G.add_edge(fid, did, rel="parent")
                if i > 0:
                    pdid = f"dir:{parents[i - 1]}"
                    if G.has_node(pdid):
                        G.add_edge(did, pdid, rel="parent")
                        G.add_edge(pdid, did, rel="child")

        # EXTERNAL_PACKAGE
        for pkg in parsed.get("imports", set()):
            pid = f"extpkg:{pkg}"
            if not G.has_node(pid):
                G.add_node(pid, type="EXTERNAL_PACKAGE", name=pkg)
            for fp, _ in files:
                G.add_edge(f"file:{fp}", pid, rel="imports")

    def _add_tier_within_file(
        self,
        G: nx.MultiDiGraph,
        cid: str,
        ts: float,
        parsed: dict,
    ):
        """Add within-file tier: CLASS, FUNCTION, VARIABLE, etc."""
        fids = [f"file:{fp}" for fp, _ in parsed.get("files", [])]

        # Classes
        for cls in parsed.get("classes", []):
            clid = f"class:{cls}"
            if not G.has_node(clid):
                G.add_node(clid, type="CLASS", name=cls)
                self.intervals.create_interval(G, clid, ts)
            else:
                self.intervals.update_interval_end(G, clid, ts)
            for fid in fids:
                G.add_edge(fid, clid, rel="contains")

        # Functions
        for fn, args, ret in parsed.get("functions", []):
            fnid = f"func:{fn}"
            sig_id = f"sig:{fn}({args})->{ret}"
            
            if not G.has_node(fnid):
                G.add_node(fnid, type="FUNCTION", name=fn)
                self.intervals.create_interval(G, fnid, ts)
            else:
                self.intervals.update_interval_end(G, fnid, ts)
            
            if not G.has_node(sig_id):
                G.add_node(
                    sig_id,
                    type="FUNCTION_SIGNATURE",
                    args=args,
                    returns=ret,
                )
                self.intervals.create_interval(G, sig_id, ts)
            else:
                self.intervals.update_interval_end(G, sig_id, ts)
            
            G.add_edge(fnid, sig_id, rel="has_signature")
            for fid in fids:
                G.add_edge(fid, fnid, rel="contains")

        # Variables
        for var, dtype in parsed.get("variables", []):
            vid = f"var:{var}"
            dtid = f"dtype:{dtype}"
            
            if not G.has_node(vid):
                G.add_node(vid, type="VARIABLE", name=var)
            if not G.has_node(dtid):
                G.add_node(dtid, type="DATA_TYPE", id=dtype)
            
            G.add_edge(vid, dtid, rel="has_type")
            for fid in fids:
                G.add_edge(fid, vid, rel="contains")

    def _add_tier_semantic(
        self,
        G: nx.MultiDiGraph,
        cid: str,
        row: dict,
        parsed: dict,
        commit_issue_map: Dict,
        issue_meta: Dict,
    ):
        """Add semantic tier: ISSUE, BRANCH."""
        # Issues
        issue_ids = list(commit_issue_map.get(row.get("commit_id"), []))
        if not issue_ids:
            issue_ids = sorted(parsed.get("issues", set()))
        
        for issue_id in issue_ids:
            inode = f"issue:{issue_id}"
            if not G.has_node(inode):
                meta = issue_meta.get(issue_id, {})
                G.add_node(
                    inode,
                    type="ISSUE",
                    id=issue_id,
                    title=meta.get("title", ""),
                    description=meta.get("description", ""),
                    status=meta.get("status", ""),
                    created=meta.get("created", ""),
                    updated=meta.get("updated", ""),
                )
            G.add_edge(cid, inode, rel="for")

    def _get_author(self, row: dict, parsed: dict) -> str:
        """Get author email from row or parsed diff."""
        for col in ("author_email", "author", "author_name"):
            val = row.get(col)
            if val and isinstance(val, str):
                return val.strip()
        
        _, email = parsed.get("author", (None, None))
        return email or "unknown"


def extract_kg_features(G: nx.MultiDiGraph, row: dict, parsed: dict) -> Dict[str, float]:
    """
    Extract KG-derived features for a commit.
    
    Called BEFORE the commit is added to graph (query-only, no modifications).
    
    Parameters
    ----------
    G : nx.MultiDiGraph
        Knowledge graph state before this commit
    row : dict
        Commit data
    parsed : dict
        Parsed commit information
        
    Returns
    -------
    dict
        Feature dictionary with keys like kg_project_commit_count, etc.
    """
    feats = {}

    # ── Project-level ─────────────────────────────────────────────────
    commit_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "COMMIT"]
    n = len(commit_nodes)
    feats["kg_project_commit_count"] = float(n)
    
    if n > 0:
        n_bug = sum(
            1 for cn in commit_nodes
            if any(
                d.get("rel") == "is" and v == "label:-1"
                for _, v, d in G.out_edges(cn, data=True)
            )
        )
        feats["kg_project_bug_rate"] = n_bug / n
    else:
        feats["kg_project_bug_rate"] = 0.0

    # ── Author-level ──────────────────────────────────────────────────
    email = None
    for col in ("author_email", "author", "author_name"):
        val = row.get(col)
        if val:
            email = val
            break
    if not email:
        _, email = parsed.get("author", (None, None))
    email = email or "unknown"
    
    aid = f"author:{email}"
    ad = G.nodes[aid] if G.has_node(aid) else {}
    ac = ad.get("commit_count", 0)
    bc = ad.get("bug_count", 0)
    feats["kg_author_commit_count"] = float(ac)
    feats["kg_author_bug_rate"] = bc / ac if ac > 0 else 0.0

    # ── File-level ────────────────────────────────────────────────────
    change_counts, bug_rates, uniq_authors = [], [], []
    for fp, _ in parsed.get("files", []):
        fid = f"file:{fp}"
        if not G.has_node(fid):
            continue
        fd = G.nodes[fid]
        cc = fd.get("change_count", 0)
        bcc = fd.get("bug_count", 0)
        change_counts.append(cc)
        bug_rates.append(bcc / cc if cc > 0 else 0.0)
        uniq_authors.append(len(fd.get("authors", set())))

    feats["kg_file_change_count"] = float(sum(change_counts))
    feats["kg_file_bug_rate"] = (
        sum(bug_rates) / len(bug_rates) if bug_rates else 0.0
    )
    feats["kg_file_unique_authors"] = (
        sum(uniq_authors) / len(uniq_authors) if uniq_authors else 0.0
    )

    return feats
