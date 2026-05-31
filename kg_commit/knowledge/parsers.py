"""
Parser module for extracting structured information from commit diffs.

Provides functions to parse:
- Files (added, modified, deleted)
- Authors (from git headers)
- Imports (Python and Java/Groovy)
- Classes
- Functions
- Variables
- Issues (JIRA-style)
- Branches

All parsers are pure functions with no side effects.
"""

from __future__ import annotations
import re
from typing import List, Set, Tuple, Optional


# ============================================================================
# REGEX PATTERNS (tuned for Python and Java/Groovy)
# ============================================================================

_RE_AUTHOR = re.compile(r'^Author:\s+(.+?)\s+<(.+?)>', re.MULTILINE)
_RE_SHOW = re.compile(r'^(?:commit [0-9a-f]{7,}|Author:\s)', re.MULTILINE)

# Imports — both added (+) and removed (-) lines
_RE_PY_IMP = re.compile(r'^[+-]\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))', re.MULTILINE)
_RE_JV_IMP = re.compile(r'^[+-]\s*import\s+(?:static\s+)?([\w.]+)\s*;', re.MULTILINE)

# Classes — any line (added, removed, context)
_RE_CLASS = re.compile(
    r'^\s*(?:public\s+|private\s+|protected\s+|abstract\s+|final\s+|static\s+)*'
    r'(?:class|interface|enum)\s+(\w+)',
    re.MULTILINE
)

# Function signatures — strict modifier requirement
_RE_PY_FN = re.compile(r'^\s*(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?')
_RE_JV_FN = re.compile(
    r'^\s*(?:(?:public|private|protected|static|final|synchronized|abstract|native|default)\s+)+'
    r'([\w.<>\[\]]+)\s+(\w+)\s*\(([^)]*)\)'
)

_RE_HUNK = re.compile(r'^@@ .*? @@\s*(.*)$', re.MULTILINE)

# Variables — primitives or capitalized types
_RE_PY_VAR = re.compile(
    r'^\+\s*(?:self\.)?(\w+)\s*:\s*([\w][\w\[\], .]*?)\s*=',
    re.MULTILINE
)
_RE_JV_VAR = re.compile(
    r'^\+\s*(?:final\s+)?'
    r'(int|long|float|double|boolean|char|byte|short|[A-Z]\w*(?:<[^>]+>)?(?:\[\])?)'
    r'\s+(\w+)\s*[=;]',
    re.MULTILINE
)

# Issues — JIRA-style keys
_RE_ISSUE = re.compile(r'\b([A-Z][A-Z0-9]+-\d+)\b')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _strip_prefix(p: str) -> str:
    """Remove 'a/' or 'b/' prefix from unified diff paths."""
    p = p.strip().strip('"')
    if p.startswith(("a/", "b/")):
        p = p[2:]
    return p


# ============================================================================
# PARSER FUNCTIONS
# ============================================================================

def parse_files(diff: Optional[str]) -> List[Tuple[str, str]]:
    """
    Extract (filepath, change_type) from unified diff.
    
    Parameters
    ----------
    diff : str | None
        Unified diff text
        
    Returns
    -------
    list of (filepath, 'add'|'remove'|'modify')
        Change operations on files
    """
    if not isinstance(diff, str):
        return []
    
    lines = diff.split("\n")
    out = []
    
    for i in range(len(lines) - 2):
        l1, l2, l3 = lines[i], lines[i + 1], lines[i + 2]
        if l1.startswith("--- ") and l2.startswith("+++ ") and l3.startswith("@@"):
            a = _strip_prefix(l1[4:])
            b = _strip_prefix(l2[4:])
            
            if a == "/dev/null":
                out.append((b, "add"))
            elif b == "/dev/null":
                out.append((a, "remove"))
            else:
                out.append((b, "modify"))
    
    return out


def parse_author(diff: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract (name, email) from git show header.
    
    Returns
    -------
    (name, email) or (None, None)
    """
    if not isinstance(diff, str):
        return None, None
    
    m = _RE_AUTHOR.search(diff)
    if m:
        name = m.group(1).strip() if m.group(1) else None
        email = m.group(2).strip() if m.group(2) else None
        return (name, email)
    return (None, None)


def parse_imports(diff: Optional[str]) -> Set[str]:
    """
    Extract full package names from imports (added or removed).
    
    Returns
    -------
    set of full package names
    """
    if not isinstance(diff, str):
        return set()
    
    pkgs = set()
    
    for m in _RE_PY_IMP.finditer(diff):
        pkg = (m.group(1) or m.group(2) or "").strip()
        if pkg:
            pkgs.add(pkg)
    
    for m in _RE_JV_IMP.finditer(diff):
        pkg = m.group(1).strip()
        if pkg:
            pkgs.add(pkg)
    
    return pkgs


def parse_classes(diff: Optional[str]) -> List[str]:
    """
    Extract class names from added lines.
    
    Returns
    -------
    list of class names
    """
    if not isinstance(diff, str):
        return []
    return [m.group(1) for m in _RE_CLASS.finditer(diff)]


def _match_fn(text: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """
    Try to match Python or Java function signature.
    
    Returns
    -------
    (name, args, return_type|None) or None
    """
    m = _RE_PY_FN.search(text)
    if m:
        return m.group(1), m.group(2).strip(), (m.group(3) or "").strip() or None
    
    m = _RE_JV_FN.search(text)
    if m:
        return m.group(2), m.group(3).strip(), m.group(1).strip()
    
    return None


def parse_functions(diff: Optional[str]) -> List[Tuple[str, str, Optional[str]]]:
    """
    Extract functions with signature info.
    
    Scans EVERY line (added/removed/context) plus hunk headers
    to catch methods modified without signature change.
    
    Returns
    -------
    list of (name, args, return_type|None)
    """
    if not isinstance(diff, str):
        return []
    
    out, seen = [], set()
    
    # Scan all lines
    for line in diff.splitlines():
        stripped = line[1:] if line and line[0] in "+- " else line
        hit = _match_fn(stripped)
        if hit and hit[0] not in seen:
            seen.add(hit[0])
            out.append(hit)
    
    # Scan hunk headers
    for m in _RE_HUNK.finditer(diff):
        hit = _match_fn(m.group(1))
        if hit and hit[0] not in seen:
            seen.add(hit[0])
            out.append(hit)
    
    return out


def parse_variables(diff: Optional[str]) -> List[Tuple[str, str]]:
    """
    Extract (variable_name, type_name) from typed declarations on added lines.
    
    Returns
    -------
    list of (var_name, type_name)
    """
    if not isinstance(diff, str):
        return []
    
    out, seen = [], set()
    
    for m in _RE_PY_VAR.finditer(diff):
        v, t = m.group(1), m.group(2).strip()
        if (v, t) not in seen:
            seen.add((v, t))
            out.append((v, t))
    
    for m in _RE_JV_VAR.finditer(diff):
        v, t = m.group(2), m.group(1)
        if (v, t) not in seen:
            seen.add((v, t))
            out.append((v, t))
    
    return out


def parse_issues(diff: Optional[str]) -> Set[str]:
    """
    Extract JIRA-style issue IDs from commit message.
    
    Only scans the header region of 'git show' output
    to avoid false positives in code.
    
    Returns
    -------
    set of issue IDs
    """
    if not isinstance(diff, str) or not _RE_SHOW.search(diff):
        return set()
    
    # Extract header region (before first diff --git or ---)
    header = re.split(r'^(?:diff --git |--- )', diff, maxsplit=1, flags=re.MULTILINE)[0]
    return set(_RE_ISSUE.findall(header))


def parse_commit_message(diff: Optional[str]) -> str:
    """
    Extract the commit message from git show output.
    
    Returns
    -------
    str
        Commit message
    """
    if not isinstance(diff, str):
        return ""
    
    # Find the first diff --git line or triple dash
    match = re.search(r'^(?:diff --git |--- )', diff, re.MULTILINE)
    if match:
        message = diff[:match.start()].strip()
    else:
        message = diff.strip()
    
    # Remove git show headers: lines starting with 'commit ', 'Author:', 'Date:', etc.
    # Stop removing headers at first blank line
    lines = []
    in_header = True
    for line in message.split('\n'):
        # Check if this is a header line
        is_header = any(line.startswith(p) for p in ('commit ', 'Author:', 'Date:', 'Merge:'))
        
        if in_header:
            if is_header:
                continue  # Skip header line
            elif line.strip() == '':
                in_header = False  # Blank line ends headers, skip it
                continue
            else:
                in_header = False  # Non-header content found, process it
        
        lines.append(line)
    
    return '\n'.join(lines).strip()


class CommitParser:
    """
    Wrapper for all parsing functions.
    Provides a unified interface for extracting all information from a diff.
    """

    def __init__(self):
        """Initialize parser."""
        pass

    def parse(self, diff: Optional[str]) -> dict:
        """
        Parse all information from a diff.
        
        Parameters
        ----------
        diff : str | None
            Unified diff text
            
        Returns
        -------
        dict with keys:
            - files: list of (filepath, change_type)
            - author: (name, email) tuple
            - imports: set of package names
            - classes: list of class names
            - functions: list of (name, args, return_type)
            - variables: list of (var_name, type_name)
            - issues: set of issue IDs
            - message: commit message
        """
        return {
            "files": parse_files(diff),
            "author": parse_author(diff),
            "imports": parse_imports(diff),
            "classes": parse_classes(diff),
            "functions": parse_functions(diff),
            "variables": parse_variables(diff),
            "issues": parse_issues(diff),
            "message": parse_commit_message(diff),
        }
