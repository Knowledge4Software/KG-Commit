"""
Shared table-emphasis helpers.

Bold marks "best in its comparison group" and is applied consistently across the
paper: within an aggregate row (which model wins), within a project block (which
configuration wins), or down a column (which model wins on that project/metric).
Underline is reserved for a single distinguished row: the configuration the
methodology actually deploys.

Keeping these in one place means the emphasis rule is the same everywhere, which is
what makes it readable rather than decorative.
"""


def fmt(v, nd=3, strip0=False):
    if v is None or v != v:                     # None or NaN
        return "--"
    s = f"{v:.{nd}f}"
    return s.lstrip("0") if strip0 and abs(v) < 1 else s


def bold(s):
    return r"\textbf{%s}" % s


def uline(s):
    return r"\underline{%s}" % s


def bold_max(values, nd=3, strip0=False, key=None, tol=0.0):
    """Format a list of numbers, bolding the maximum (and anything within `tol`).

    `key` maps an entry to its comparable number when the list holds richer objects.
    Returns a list of formatted strings aligned with the input.
    """
    nums = [(key(v) if key else v) for v in values]
    valid = [x for x in nums if x is not None and x == x]
    if not valid:
        return [fmt(None) for _ in values]
    top = max(valid)
    out = []
    for x in nums:
        s = fmt(x, nd, strip0)
        if x is not None and x == x and x >= top - tol:
            s = bold(s)
        out.append(s)
    return out


def bold_max_column(rows, col, nd=3, strip0=False):
    """In-place: bold the maximum of column `col` across `rows` (list of lists)."""
    vals = []
    for r in rows:
        try:
            vals.append(float(str(r[col]).replace(r"\textbf{", "").rstrip("}")))
        except (ValueError, TypeError):
            vals.append(None)
    valid = [v for v in vals if v is not None]
    if not valid:
        return rows
    top = max(valid)
    for r, v in zip(rows, vals):
        if v is not None and v >= top - 1e-12:
            r[col] = bold(fmt(v, nd, strip0))
    return rows
