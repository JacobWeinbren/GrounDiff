"""Ordnance Survey National Grid tile names (e.g. TL4378nw) -> extents.

EA tiles are 500 m quadrants (TL4378nw) or, with a 4-digit name, squares
named by their south-west km: 2 km tiles in the EA time-stamped archive (e.g.
NX9410 = 294000-296000 E, 510000-512000 N; even digits) and 1 km tiles
elsewhere. The size of a 4-digit tile is therefore only a guess from its
digits (size_known=False): use the file's own bounds when you have the file.
Names are found anywhere in a file name, so survey suffixes like
_P_12534_20220315 are ignored.
"""
from __future__ import annotations

import re
from pathlib import Path

TILE_RE = re.compile(r"(?<![A-Z])([A-Z]{2})(\d{2})(\d{2})(ne|nw|se|sw)?", re.IGNORECASE)
QUAD_OFFSET = {"sw": (0, 0), "se": (500, 0), "nw": (0, 500), "ne": (500, 500)}
LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"          # OS grid letters (no I)


def os_origin(sq: str, e_km: int, n_km: int, quad: str | None = None) -> tuple[int, int] | None:
    """South-west corner (BNG metres) of a 1 km square or 500 m quadrant."""
    try:
        l1, l2 = LETTERS.index(sq[0].upper()), LETTERS.index(sq[1].upper())
    except ValueError:
        return None
    e100 = ((l1 - 2) % 5) * 5 + (l2 % 5)
    n100 = (19 - (l1 // 5) * 5) - (l2 // 5)
    dx, dy = QUAD_OFFSET.get(quad or "", (0, 0))
    return e100 * 100_000 + e_km * 1000 + dx, n100 * 100_000 + n_km * 1000 + dy


def parse_tile(name: str) -> dict | None:
    m = TILE_RE.search(Path(name).name)
    if not m:
        return None
    sq, e, n, q = m.group(1).upper(), int(m.group(2)), int(m.group(3)), (m.group(4) or "").lower()
    origin = os_origin(sq, e, n, q)
    if origin is None:
        return None
    size = 500 if q else (2000 if e % 2 == 0 and n % 2 == 0 else 1000)
    return {"grid": sq, "square": f"{sq}{m.group(2)}{m.group(3)}", "quad": q or None, "origin": origin,
            "size": size, "size_known": bool(q),
            "extent": (origin[0], origin[1], origin[0] + size, origin[1] + size)}
