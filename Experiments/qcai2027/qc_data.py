#!/usr/bin/env python3
"""Data of the QCAI 2027 proxy experiment (plan phases QB and QC).

Implemented steps
-----------------
QB1.1 - Region R.  R (``region`` in ``qc_config.yaml``) is covered by several
products per modality (2 WAC tiles, 4 SLDEM tiles).  Each product is an
equirectangular (simple cylindrical) grid; its geometry is read from the PDS
label (``MAP_RESOLUTION`` and projection offsets), not from the rounded
bounding box of the inventory.  ``region_layout`` finds the pixel window of R
in every product and where that window goes in the grid of R, and checks that
the windows tile R exactly (no gap, no overlap, integer alignment).
``read_region`` then reads any window of R in physical units, whatever the
products it crosses, with the readers of the dataset (``data_quality``: PDS
labels, no-data sentinels, georeferencing correction D15 of the WAC tiles).

QB2.1 - Craters of R.  The Robbins catalogue is read (centre of the circle
fit, diameter in km), the craters whose centre is in R or within one tile
(``tiling.tile_px``) of it are kept, and their centre is converted to
fractional row/column of the WAC grid of R (``craters_on_grid``); the
longitude is wrapped around the centre of R, so a margin across 0/360 E works.

QB2.2 - Cell labels.  R is divided into cells of ``tiling.cell_px`` (16) px,
aligned with the tiles (cell (i, j) = pixels [16i, 16i+16) x [16j, 16j+16));
only full cells are kept.  Per cell: number of crater centres in the target
diameter range (``labels.diam_range_km``, 1-8 km), above and below it, and
for the largest target crater its diameter, the offset of its centre from the
cell centre (``dx`` east, ``dy`` south, cell units, [-0.5, 0.5)) and
``log_r`` = ln(radius in km) (``cell_labels``).

QB2.3 - Cell classes (QD5).  positive = at least one centre with 1 <= D < 8 km,
negative = no centre, ignore = only centres outside the target range (D >= 8 km);
positive cells get the diameter band (``labels.bands_km``) of their largest
target crater (``cell_classes``).  Stored with the labels of QB2.2.

QB2.4 - Label report.  Positive cells with more than one target centre, the
craters that the one-crater-per-cell labels cannot represent (not the largest
of their cell), and the class balance per diameter band (``run_label_report``).

QB3.1 - Geographic split.  R is divided into blocks of ``split.block_deg``
(5 deg) and every cell belongs to the block that contains its centre
(``cell_blocks``).  The blocks are ranked by the density of target craters
(1-8 km centres per 1,000 km2 of the block, spherical area) and divided into
``split.n_strata`` equal strata (terciles); in each stratum the blocks are
shuffled with the seed and the first ``floor(fraction * n)`` go to test and
QK-pool, the rest to S1, of which ``floor(val_fraction_of_s1 * n_s1)`` to
S1-val (``assign_split``).  The table of the blocks goes to
``<output>/split_blocks.csv``.

Pixel convention: pixel (r, c) of a grid covers latitudes
[north - (r + 1) / ppd, north - r / ppd] and longitudes
[west + c / ppd, west + (c + 1) / ppd]; fractional indices have the pixel
centres at integers.

Usage
-----
    python Experiments/qcai2027/qc_data.py region      # QB1.1: geometry and checks -> report section QB1.1
    python Experiments/qcai2027/qc_data.py craters     # QB2.1: craters of R on the WAC grid -> <output>/craters_R.csv
    python Experiments/qcai2027/qc_data.py cells       # QB2.2-QB2.3: labels and classes of the 16 x 16 px cells -> <output>/cell_labels.npz
    python Experiments/qcai2027/qc_data.py label-report   # QB2.4: multiple centres and class balance -> section QB2.4
    python Experiments/qcai2027/qc_data.py split       # QB3.1: 5 deg blocks -> sets -> <output>/split_blocks.csv
    python Experiments/qcai2027/qc_data.py --selftest  # synthetic checks, no data needed
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import math
import os
import sys
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().with_name("qc_config.yaml")
MOON_RADIUS_M = 1_737_400.0
ALIGN_TOL_PX = 0.01          # a window edge must fall within this distance of a pixel edge


# ---------------------------------------------------------------------------
# Configuration and dataset modules
# ---------------------------------------------------------------------------


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Configuration with the ``paths`` resolved against the project root."""
    cfg = yaml.safe_load(path.read_text())
    cfg["paths"] = {k: (PROJECT_ROOT / v).resolve() for k, v in cfg["paths"].items()}
    _import_dataset_modules(cfg)
    return cfg


# Reports: one JSON per phase of the plan in ``paths.results``, one section per step id
REPORTS = {"QA": "qa_checks.json", "QB": "qb_region_labels.json", "QC": "qc_preprocessing.json",
           "QD": "qd_data_analysis.json", "QE": "qe_features.json", "QF": "qf_fusion.json",
           "QG": "qg_probes.json", "QH": "qh_stage1.json", "QI": "qi_kernels.json",
           "QJ": "qj_classical.json", "QK": "qk_noise.json"}


_WRITTEN: list[str] = []


def written(path) -> None:
    """Record a file written or modified by the running command; the list is printed at exit
    (skill output-artefatti-parsimoniosi, section 7)."""
    p = str(Path(path).resolve())
    if not _WRITTEN:
        import atexit
        atexit.register(lambda: print("\nFiles written or modified:\n" + "\n".join(f"  {f}" for f in _WRITTEN)))
    if p not in _WRITTEN:
        _WRITTEN.append(p)


def _meta(step: str) -> dict:
    """Provenance of a report section: step, script and command, time, MD5 of the configuration."""
    import datetime
    import hashlib
    return {"step": step, "script": Path(sys.argv[0]).name, "command": " ".join(sys.argv[1:]),
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "config_md5": hashlib.md5(CONFIG_PATH.read_bytes()).hexdigest()}


def _strip_paths(x):
    """Copy of a report without the absolute file paths (machine-specific; the files are in the plan)."""
    if isinstance(x, dict):
        out = {k: _strip_paths(v) for k, v in x.items() if not (isinstance(v, str) and v.startswith("/"))}
        return {k: v for k, v in out.items() if not (v == [] and x[k] != [])}       # lists made only of paths
    if isinstance(x, list):
        return [_strip_paths(v) for v in x if not (isinstance(v, str) and v.startswith("/"))]
    return x


def report_path(cfg: dict, step: str) -> Path:
    return cfg["paths"]["results"] / REPORTS[step[:2].upper()]


def save_report(cfg: dict, step: str, data: dict) -> Path:
    """Write ``data`` as the section ``step`` (e.g. "QC4.1") of the report of its phase, keeping the other
    sections; sections sorted by step id."""
    out = report_path(cfg, step)
    report = json.loads(out.read_text()) if out.exists() else {}
    report[step] = {"_meta": _meta(step), **_strip_paths(data)}
    key = lambda k: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", k)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: report[k] for k in sorted(report, key=key)}, indent=1) + "\n")
    written(out)
    return out


def save_npz_atomic(path: Path, arrays: dict) -> None:
    """Write ``arrays`` to the npz ``path`` through a temporary file (a job killed while writing leaves the previous
    file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("tmp_" + path.name)
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_report(cfg: dict, step: str) -> dict:
    """Section ``step`` of the report of its phase."""
    section = json.loads(report_path(cfg, step).read_text())[step]
    return {k: v for k, v in section.items() if k != "_meta"}


def _import_dataset_modules(cfg: dict) -> None:
    """Make the dataset scripts (``data_quality``, ``data_inventory``, ...) importable."""
    scripts = cfg["paths"]["dataset_scripts"]
    for sub in ("A_sources", "B_diagnostics"):
        p = str(scripts / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


def read_inventory(cfg: dict) -> dict[str, dict[str, str]]:
    """Rows of ``data_inventory.csv`` by product_id."""
    with open(cfg["paths"]["inventory"], newline="", encoding="utf-8-sig") as f:
        return {row["product_id"]: row for row in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Equirectangular grids
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CylGrid:
    """Equirectangular grid: edge latitude/longitude of pixel (0, 0), pixels per degree, size."""
    north: float
    west: float
    ppd: float
    lines: int
    samples: int

    @property
    def south(self) -> float:
        return self.north - self.lines / self.ppd

    @property
    def east(self) -> float:
        return self.west + self.samples / self.ppd

    def rc_to_latlon(self, rows, cols):
        """(lat, lon) of fractional pixel indices (pixel centres at integers)."""
        return self.north - (np.asarray(rows) + 0.5) / self.ppd, self.west + (np.asarray(cols) + 0.5) / self.ppd

    def latlon_to_rc(self, lat, lon):
        """Fractional (row, col) of (lat, lon); inverse of ``rc_to_latlon``."""
        return (self.north - np.asarray(lat)) * self.ppd - 0.5, (np.asarray(lon) - self.west) * self.ppd - 0.5

    def window(self, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> tuple[slice, slice] | None:
        """Pixel window (edges snapped to the nearest pixel edge) of a lat/lon box, or None if outside."""
        r0 = max(0, round((self.north - lat_max) * self.ppd))
        r1 = min(self.lines, round((self.north - lat_min) * self.ppd))
        c0 = max(0, round((lon_min - self.west) * self.ppd))
        c1 = min(self.samples, round((lon_max - self.west) * self.ppd))
        return (slice(r0, r1), slice(c0, c1)) if r1 > r0 and c1 > c0 else None


def grid_from_label(label) -> CylGrid:
    """Grid of a PDS3 simple-cylindrical product from its label.

    PDS: line (sample) projection offset = position of the projection centre
    in pixels, pixel centres at x.0 counted from 1, hence edge = offset + 0.5.
    """
    from data_inventory import lookup, number  # noqa: E402  (dataset module)

    proj = label.projection
    ppd = float(number(lookup(proj, "MAP_RESOLUTION")))
    north = float(number(lookup(proj, "CENTER_LATITUDE"))) + (float(number(lookup(proj, "LINE_PROJECTION_OFFSET"))) + 0.5) / ppd
    west = float(number(lookup(proj, "CENTER_LONGITUDE"))) - (float(number(lookup(proj, "SAMPLE_PROJECTION_OFFSET"))) + 0.5) / ppd
    image = label.image
    return CylGrid(north=north, west=west % 360.0, ppd=ppd,
                   lines=int(number(lookup(image, "LINES"))), samples=int(number(lookup(image, "LINE_SAMPLES"))))


# ---------------------------------------------------------------------------
# Layout of the products in R
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Piece:
    """Window of one product (src) and its place in the grid of R (dst); same size."""
    product_id: str
    src_rows: slice
    src_cols: slice
    dst_rows: slice
    dst_cols: slice


def tile_layout(grids: dict[str, CylGrid], bounds: dict) -> tuple[CylGrid, list[Piece]]:
    """Grid of R and the pieces of the products that cover it; checks exact tiling.

    Pure geometry (no file access), so that it can be tested on synthetic grids.
    """
    ppds = {g.ppd for g in grids.values()}
    if max(ppds) - min(ppds) > 1e-9 * max(ppds):
        raise ValueError(f"products with different resolutions: {sorted(ppds)}")
    ppd = next(iter(grids.values())).ppd
    windows = {}
    for pid, g in grids.items():
        w = g.window(bounds["lat_min"], bounds["lat_max"], bounds["lon_min"], bounds["lon_max"])
        if w is not None:
            windows[pid] = (g, w)
    if not windows:
        raise ValueError("no product covers the region")
    # edges of R = outermost window edges
    north = max(g.north - w[0].start / ppd for g, w in windows.values())
    west = min(g.west + w[1].start / ppd for g, w in windows.values())
    pieces = []
    for pid, (g, (rows, cols)) in windows.items():
        dr = (north - (g.north - rows.start / ppd)) * ppd
        dc = (g.west + cols.start / ppd - west) * ppd
        if abs(dr - round(dr)) > ALIGN_TOL_PX or abs(dc - round(dc)) > ALIGN_TOL_PX:
            raise ValueError(f"{pid}: pixel grid not aligned with the other products ({dr:.4f}, {dc:.4f} px)")
        r0, c0 = round(dr), round(dc)
        pieces.append(Piece(pid, rows, cols, slice(r0, r0 + rows.stop - rows.start), slice(c0, c0 + cols.stop - cols.start)))
    lines = max(p.dst_rows.stop for p in pieces)
    samples = max(p.dst_cols.stop for p in pieces)
    _check_tiling(pieces, lines, samples)
    return CylGrid(north=north, west=west, ppd=ppd, lines=lines, samples=samples), pieces


def _check_tiling(pieces: list[Piece], lines: int, samples: int) -> None:
    """The dst windows cover the lines x samples grid exactly once."""
    area = sum((p.dst_rows.stop - p.dst_rows.start) * (p.dst_cols.stop - p.dst_cols.start) for p in pieces)
    for i, a in enumerate(pieces):
        for b in pieces[i + 1:]:
            if (a.dst_rows.start < b.dst_rows.stop and b.dst_rows.start < a.dst_rows.stop
                    and a.dst_cols.start < b.dst_cols.stop and b.dst_cols.start < a.dst_cols.stop):
                raise ValueError(f"{a.product_id} and {b.product_id} overlap in the region")
    if area != lines * samples:
        raise ValueError(f"the products leave gaps in the region ({area} of {lines * samples} px)")


@dataclass
class RegionSource:
    """One modality over R: its grid, the pieces and the open products."""
    modality: str
    grid: CylGrid
    pieces: list[Piece]
    rasters: dict            # product_id -> (data_quality.Raster, sentinels)


def product_grids(cfg: dict, modality: str, inventory: dict | None = None) -> dict[str, CylGrid]:
    """Grids of the products of one modality (``products`` in the config), from the labels only."""
    from data_inventory import PdsLabel, read_embedded_label  # noqa: E402

    inventory = inventory or read_inventory(cfg)
    grids = {}
    for pid in cfg["products"][modality]:
        label_path = cfg["paths"]["dataset_root"] / inventory[pid]["label_file"]
        text = read_embedded_label(label_path) if label_path.suffix.lower() == ".img" else None
        grids[pid] = grid_from_label(PdsLabel(label_path, text=text))
    return grids


def region_grid(cfg: dict, modality: str = "wac", inventory: dict | None = None) -> CylGrid:
    """Grid of R for one modality (no image is opened).  The WAC grid is the reference grid."""
    return tile_layout(product_grids(cfg, modality, inventory), cfg["region"])[0]


def region_layout(cfg: dict, modality: str, inventory: dict | None = None) -> RegionSource:
    """Open the products of one modality (``products`` in the config) and lay them out on R."""
    from data_quality import open_image  # noqa: E402

    inventory = inventory or read_inventory(cfg)
    grids = product_grids(cfg, modality, inventory)
    root = cfg["paths"]["dataset_root"]
    rasters = {}
    for pid, g in grids.items():
        raster, _geom, sentinels = open_image(inventory[pid], root)
        if (raster.lines, raster.samples) != (g.lines, g.samples):
            raise ValueError(f"{pid}: raster size differs from the label")
        rasters[pid] = (raster, sentinels)
    grid, pieces = tile_layout(grids, cfg["region"])
    return RegionSource(modality, grid, pieces, rasters)


def read_region(src: RegionSource, rows: slice, cols: slice, d15: str = "mosaic") -> np.ndarray:
    """Window of R in physical units (float64, no-data NaN), assembled from the products.

    ``rows``/``cols`` are slices of the grid of R (step allowed).

    Georeferencing correction D15: the corrected value at (r, c) is the stored
    value at (r + dr, c + dc).  The products of one modality are tiles of one
    mosaic with the same shift, so the shift is applied to the mosaic, not tile
    by tile: the row that a shifted tile lacks at a tile border is read from
    the neighbouring tile (``data_quality.read_physical`` alone would leave it
    NaN, e.g. the whole WAC row just north of the equator).  Only the pixels
    shifted in from outside every product stay NaN.  ``d15="tile"`` reproduces the
    per-tile correction of ``read_physical``, only to document the difference (QB1.1).
    """
    from data_quality import read_physical  # noqa: E402

    if d15 == "tile":
        return _read_region_per_tile(src, rows, cols)

    shifts = {src.rasters[p.product_id][0].shift_rc for p in src.pieces}
    if len(shifts) != 1:
        raise ValueError(f"{src.modality}: products with different D15 shifts {shifts}")
    dr, dc = shifts.pop()
    rs = np.arange(src.grid.lines)[rows] + dr          # stored-mosaic coordinates of the wanted pixels
    cs = np.arange(src.grid.samples)[cols] + dc
    out = np.full((rs.size, cs.size), np.nan)
    for p in src.pieces:
        raster, sentinels = src.rasters[p.product_id]
        sr = rs - (p.dst_rows.start - p.src_rows.start)   # indices in the whole product, not only its window of R
        sc = cs - (p.dst_cols.start - p.src_cols.start)
        mr = (sr >= 0) & (sr < raster.lines)
        mc = (sc >= 0) & (sc < raster.samples)
        if not (mr.any() and mc.any()):
            continue
        sr, sc = sr[mr], sc[mc]
        step_r = int(sr[1] - sr[0]) if sr.size > 1 else 1
        step_c = int(sc[1] - sc[0]) if sc.size > 1 else 1
        stored = replace(raster, shift_rc=(0, 0))        # shift already applied above
        out[np.ix_(mr, mc)] = read_physical(stored, sentinels, slice(sr[0], sr[-1] + 1, step_r),
                                            slice(sc[0], sc[-1] + 1, step_c))
    return out


def _read_region_per_tile(src: RegionSource, rows: slice, cols: slice) -> np.ndarray:
    """As ``read_region`` but with the D15 shift applied to each product separately."""
    from data_quality import read_physical  # noqa: E402

    r_idx = np.arange(src.grid.lines)[rows]
    c_idx = np.arange(src.grid.samples)[cols]
    out = np.full((r_idx.size, c_idx.size), np.nan)
    for p in src.pieces:
        mr = (r_idx >= p.dst_rows.start) & (r_idx < p.dst_rows.stop)
        mc = (c_idx >= p.dst_cols.start) & (c_idx < p.dst_cols.stop)
        if not (mr.any() and mc.any()):
            continue
        sr = r_idx[mr] - p.dst_rows.start + p.src_rows.start
        sc = c_idx[mc] - p.dst_cols.start + p.src_cols.start
        step_r = int(sr[1] - sr[0]) if sr.size > 1 else 1
        step_c = int(sc[1] - sc[0]) if sc.size > 1 else 1
        raster, sentinels = src.rasters[p.product_id]
        out[np.ix_(mr, mc)] = read_physical(raster, sentinels, slice(sr[0], sr[-1] + 1, step_r),
                                            slice(sc[0], sc[-1] + 1, step_c))
    return out


# ---------------------------------------------------------------------------
# QB1.1 checks
# ---------------------------------------------------------------------------


def _seams(src: RegionSource) -> list[tuple[str, int]]:
    """Internal edges between pieces: ("row", r) = between rows r-1 and r; ("col", c) likewise."""
    rows = {p.dst_rows.start for p in src.pieces} - {0}
    cols = {p.dst_cols.start for p in src.pieces} - {0}
    return [("row", r) for r in sorted(rows)] + [("col", c) for c in sorted(cols)]


def seam_check(src: RegionSource, axis: str, at: int, half: int = 4, step: int = 4, d15: str = "mosaic") -> dict:
    """Continuity across a seam: per-line validity and the mean absolute step between
    neighbouring lines at the seam vs. the median of the steps of the lines around it
    (ratio ~ 1: no jump, no missing or duplicated line)."""
    if axis == "row":
        strip = read_region(src, slice(at - half, at + half), slice(0, src.grid.samples, step), d15)
        lines = strip
    else:
        strip = read_region(src, slice(0, src.grid.lines, step), slice(at - half, at + half), d15)
        lines = strip.T
    valid = np.isfinite(lines).mean(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)               # all-NaN lines give NaN steps
        steps = np.nanmean(np.abs(np.diff(lines, axis=0)), axis=1)  # step between line k and k+1
    seam_step = steps[half - 1]
    others = np.delete(steps, half - 1)
    ref = float(np.nanmedian(others)) if np.isfinite(others).any() else float("nan")
    return {
        "axis": axis, "index": at,
        "valid_fraction_per_line": [round(float(v), 4) for v in valid],
        "seam_step": float(seam_step), "median_other_steps": ref,
        "seam_ratio": float(seam_step / ref) if ref and np.isfinite(seam_step) else float("nan"),
    }


def validity_summary(src: RegionSource, step: int = 16) -> dict:
    """Valid fraction of R on a regular subsample, plus the invalid full lines/columns."""
    sub = read_region(src, slice(0, src.grid.lines, step), slice(0, src.grid.samples, step))
    edge = {
        "first_row": read_region(src, slice(0, 1), slice(0, src.grid.samples)),
        "last_row": read_region(src, slice(src.grid.lines - 1, src.grid.lines), slice(0, src.grid.samples)),
        "first_col": read_region(src, slice(0, src.grid.lines), slice(0, 1)),
        "last_col": read_region(src, slice(0, src.grid.lines), slice(src.grid.samples - 1, src.grid.samples)),
    }
    return {"subsample_step": step, "valid_fraction": round(float(np.isfinite(sub).mean()), 6),
            **{f"{k}_valid_fraction": round(float(np.isfinite(v).mean()), 6) for k, v in edge.items()}}


def pixel_m(g: CylGrid) -> float:
    """N-S pixel size (m) of an equirectangular grid on the reference sphere."""
    return 2 * math.pi * MOON_RADIUS_M / 360.0 / g.ppd


def pixel_km(g: CylGrid) -> float:
    """N-S pixel size (km) of an equirectangular grid on the reference sphere."""
    return pixel_m(g) / 1000.0


def token_grid(cfg: dict) -> int:
    """Sub-tiles (backbone tokens) per tile side: tile_px // cell_px."""
    return int(cfg["tiling"]["tile_px"]) // int(cfg["tiling"]["cell_px"])


def describe(src: RegionSource) -> dict:
    g = src.grid
    px_ns_m = pixel_m(g)
    lat_abs_max = max(abs(g.north), abs(g.south))
    return {
        "modality": src.modality,
        "grid": asdict(g),
        "extent_deg": {"lat": [g.south, g.north], "lon": [g.west, g.east]},
        "shape": [g.lines, g.samples],
        "pixel_ns_m": px_ns_m,
        "pixel_ew_m_at_equator": px_ns_m,
        "pixel_ew_m_at_max_lat": px_ns_m * math.cos(math.radians(lat_abs_max)),
        "ew_oversampling_max": 1.0 / math.cos(math.radians(lat_abs_max)),
        "size_km": [g.lines * px_ns_m / 1000.0, g.samples * px_ns_m / 1000.0],
        "pieces": [{"product_id": p.product_id,
                    "src_rows": [p.src_rows.start, p.src_rows.stop], "src_cols": [p.src_cols.start, p.src_cols.stop],
                    "dst_rows": [p.dst_rows.start, p.dst_rows.stop], "dst_cols": [p.dst_cols.start, p.dst_cols.stop],
                    "shift_rc_D15": list(src.rasters[p.product_id][0].shift_rc)} for p in src.pieces],
    }


def run_region(cfg: dict) -> dict:
    inventory = read_inventory(cfg)
    report = {"region": cfg["region"]}
    for modality in cfg["products"]:
        src = region_layout(cfg, modality, inventory)
        info = describe(src)
        info["seams"] = [seam_check(src, a, i) for a, i in _seams(src)]
        if any(r[0].shift_rc != (0, 0) for r in src.rasters.values()):
            # evidence for applying D15 to the mosaic: the same seams with the per-tile correction
            info["seams_d15_per_tile"] = [seam_check(src, a, i, d15="tile") for a, i in _seams(src)]
        info["validity"] = validity_summary(src)
        report[modality] = info
    # the grids of the modalities must cover the same region (to within one pixel of the coarser grid)
    gw, gd = report["wac"]["grid"], report["dem"]["grid"]
    tol = 1.0 / min(gw["ppd"], gd["ppd"])
    for k_edge, (a, b) in {"north": (gw["north"], gd["north"]), "west": (gw["west"], gd["west"]),
                           "south": tuple(r["extent_deg"]["lat"][0] for r in (report["wac"], report["dem"])),
                           "east": tuple(r["extent_deg"]["lon"][1] for r in (report["wac"], report["dem"]))}.items():
        if abs(a - b) > tol:
            raise ValueError(f"WAC and DEM extents of R differ at the {k_edge} edge: {a} vs {b}")
    save_report(cfg, "QB1.1", report)
    return report


def print_report(report: dict) -> None:
    for modality in ("wac", "dem"):
        r = report[modality]
        print(f"[{modality}] shape {r['shape'][0]} x {r['shape'][1]} px, ppd {r['grid']['ppd']:.6f}, "
              f"lat {r['extent_deg']['lat'][0]:.6f}..{r['extent_deg']['lat'][1]:.6f}, "
              f"lon {r['extent_deg']['lon'][0]:.6f}..{r['extent_deg']['lon'][1]:.6f}, "
              f"{r['size_km'][0]:.0f} x {r['size_km'][1]:.0f} km, pixel N-S {r['pixel_ns_m']:.2f} m, "
              f"E-W oversampling max {r['ew_oversampling_max']:.3f}")
        for p in r["pieces"]:
            print(f"    {p['product_id']}: src rows {p['src_rows']} cols {p['src_cols']} -> R rows {p['dst_rows']} "
                  f"cols {p['dst_cols']}, D15 shift {p['shift_rc_D15']}")
        for key, tag in (("seams", "D15 on the mosaic"), ("seams_d15_per_tile", "D15 per tile")):
            for s in r.get(key, []):
                print(f"    seam {s['axis']} {s['index']} ({tag}): valid per line {s['valid_fraction_per_line']}, "
                      f"step ratio {s['seam_ratio']:.2f}")
        print(f"    validity: {r['validity']}")


# ---------------------------------------------------------------------------
# QB2.1 - Crater catalogue on the grid of R
# ---------------------------------------------------------------------------

CATALOG_COLUMNS = {"CRATER_ID": "crater_id", "LAT_CIRC_IMG": "lat", "LON_CIRC_IMG": "lon", "DIAM_CIRC_IMG": "diam_km"}


def wrap_lon(lon, centre: float) -> np.ndarray:
    """Longitudes moved to [centre - 180, centre + 180), so that a box crossing 0/360 is contiguous."""
    return centre + (np.asarray(lon, dtype=np.float64) - centre + 180.0) % 360.0 - 180.0


def craters_on_grid(cat, grid: CylGrid, margin_px: int):
    """Craters whose centre is in the grid or within ``margin_px`` of it, with their pixel position.

    ``row``/``col`` are fractional indices of the centre (pixel centres at
    integers); ``in_region`` is True when the centre falls inside a pixel of
    the grid (the pixel ``round(row), round(col)``).
    """
    lon = wrap_lon(cat["lon"].to_numpy(), (grid.west + grid.east) / 2.0)
    row, col = grid.latlon_to_rc(cat["lat"].to_numpy(), lon)
    keep = ((row >= -0.5 - margin_px) & (row < grid.lines - 0.5 + margin_px)
            & (col >= -0.5 - margin_px) & (col < grid.samples - 0.5 + margin_px))
    out = cat.loc[keep].copy()
    out["row"], out["col"] = row[keep], col[keep]
    out["in_region"] = ((out["row"] >= -0.5) & (out["row"] < grid.lines - 0.5)
                        & (out["col"] >= -0.5) & (out["col"] < grid.samples - 0.5))
    return out.reset_index(drop=True)


def load_catalog(cfg: dict):
    """Robbins catalogue (QA3.1): id, centre (planetocentric lat, lon 0-360 E) and diameter (km)."""
    import pandas as pd

    cat = pd.read_csv(cfg["paths"]["catalog"], usecols=list(CATALOG_COLUMNS)).rename(columns=CATALOG_COLUMNS)
    if cat.isna().any().any() or not cat["crater_id"].is_unique:
        raise ValueError("catalogue with missing values or duplicate ids")
    return cat


def run_craters(cfg: dict) -> dict:
    """QB2.1: craters of R (+ margin of one tile) on the WAC grid of R -> ``<output>/craters_R.csv``."""
    grid = region_grid(cfg, "wac")
    cat = load_catalog(cfg)
    margin = int(cfg["tiling"]["tile_px"])
    cr = craters_on_grid(cat, grid, margin)
    out = cfg["paths"]["output"] / "craters_R.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cr.to_csv(out, index=False, float_format="%.6f")
    written(out)

    inside = cr[cr["in_region"]]
    edges = [0, 1, 2, 4, 8, np.inf]
    bands = np.histogram(inside["diam_km"], bins=edges)[0]
    summary = {
        "catalog_rows": int(len(cat)), "margin_px": margin, "kept": int(len(cr)),
        "in_region": int(len(inside)), "in_margin_only": int(len(cr) - len(inside)),
        "in_region_per_band_km": {f"{a:g}-{b:g}": int(n) for a, b, n in zip(edges[:-1], edges[1:], bands)},
        "in_region_diam_km_max": float(inside["diam_km"].max()),
        "lat_range": [float(inside["lat"].min()), float(inside["lat"].max())],
        "output": str(out),
    }
    save_report(cfg, "QB2.1", summary)
    return summary


# ---------------------------------------------------------------------------
# QB2.2 - Sub-tile (cell) labels
# ---------------------------------------------------------------------------


def cell_labels(row, col, diam_km, shape: tuple[int, int], cell_px: int, d_range: tuple[float, float]) -> dict:
    """Per-cell crater labels on a grid of ``cell_px`` x ``cell_px`` pixel cells.

    Cell (i, j) covers the pixels [i*cell_px, (i+1)*cell_px) x [j*cell_px, (j+1)*cell_px)
    of the grid of R (aligned with the tiles of QC7); a crater belongs to the cell
    of the pixel that contains its centre.  ``shape`` = number of full cells.
    Returns arrays of that shape:
      n_target   centres with d_range[0] <= D < d_range[1]
      n_above    centres with D >= d_range[1] (for the "ignore" class of QB2.3)
      n_below    centres with D < d_range[0]
      diam_km, dx, dy, log_r  of the largest target crater of the cell (NaN if none):
                 dx, dy = offset of its centre from the cell centre in cell units,
                 in [-0.5, 0.5), dx towards east (columns), dy towards south (rows);
                 log_r = natural log of the radius in km.
    """
    row, col, diam_km = (np.asarray(a, dtype=np.float64) for a in (row, col, diam_km))
    ci = np.floor((row + 0.5) / cell_px).astype(np.int64)        # pixel of the centre = floor(x + 0.5)
    cj = np.floor((col + 0.5) / cell_px).astype(np.int64)
    inside = (ci >= 0) & (ci < shape[0]) & (cj >= 0) & (cj < shape[1])
    ci, cj, row, col, diam_km = ci[inside], cj[inside], row[inside], col[inside], diam_km[inside]
    flat = ci * shape[1] + cj
    size = shape[0] * shape[1]
    target = (diam_km >= d_range[0]) & (diam_km < d_range[1])
    out = {
        "n_target": np.bincount(flat[target], minlength=size),
        "n_above": np.bincount(flat[diam_km >= d_range[1]], minlength=size),
        "n_below": np.bincount(flat[diam_km < d_range[0]], minlength=size),
    }
    for k in ("diam_km", "dx", "dy", "log_r"):
        out[k] = np.full(size, np.nan, dtype=np.float32)
    # largest target crater per cell: sort by (cell, -D) and keep the first of each cell
    t = np.flatnonzero(target)
    order = t[np.lexsort((-diam_km[t], flat[t]))]
    first = order[np.unique(flat[order], return_index=True)[1]]
    cells = flat[first]
    out["diam_km"][cells] = diam_km[first]
    out["dy"][cells] = (row[first] - (ci[first] * cell_px + (cell_px - 1) / 2.0)) / cell_px
    out["dx"][cells] = (col[first] - (cj[first] * cell_px + (cell_px - 1) / 2.0)) / cell_px
    out["log_r"][cells] = np.log(diam_km[first] / 2.0)
    out = {k: v.reshape(shape) for k, v in out.items()}
    for k in ("n_target", "n_above", "n_below"):
        out[k] = out[k].astype(np.uint16)
    out["n_outside"] = int((~inside).sum())
    return out


POSITIVE, NEGATIVE, IGNORE = 1, 0, -1


def cell_classes(lab: dict, bands_km) -> tuple[np.ndarray, np.ndarray]:
    """QB2.3 / QD5: class and diameter band of every cell.

    class: POSITIVE = at least one centre in the target range; NEGATIVE = no
    centre at all (every catalogue crater has D >= 1 km); IGNORE = only centres
    above the target range.  band: index in ``bands_km`` (edges, e.g. 1, 2, 4, 8)
    of the diameter of the largest target crater of a positive cell, -1 otherwise.
    """
    n_t, n_a, n_b = lab["n_target"], lab["n_above"], lab["n_below"]
    cls = np.full(n_t.shape, NEGATIVE, dtype=np.int8)
    cls[(n_t == 0) & ((n_a > 0) | (n_b > 0))] = IGNORE
    cls[n_t > 0] = POSITIVE
    edges = np.asarray(bands_km, dtype=np.float64)
    band = np.full(n_t.shape, -1, dtype=np.int8)
    pos = cls == POSITIVE
    band[pos] = np.clip(np.searchsorted(edges, lab["diam_km"][pos], side="right") - 1, 0, len(edges) - 2)
    return cls, band


def run_cells(cfg: dict) -> dict:
    """QB2.2-QB2.3: labels and classes of the 16 x 16 px cells of R -> ``<output>/cell_labels.npz``."""
    import pandas as pd

    grid = region_grid(cfg, "wac")
    cell = int(cfg["tiling"]["cell_px"])
    shape = (grid.lines // cell, grid.samples // cell)
    cr = pd.read_csv(cfg["paths"]["output"] / "craters_R.csv")
    cr = cr[cr["in_region"]]
    d_range = tuple(float(x) for x in cfg["labels"]["diam_range_km"])
    lab = cell_labels(cr["row"], cr["col"], cr["diam_km"], shape, cell, d_range)
    n_outside = lab.pop("n_outside")
    bands = [float(x) for x in cfg["labels"]["bands_km"]]
    lab["cls"], lab["band"] = cell_classes(lab, bands)
    out = cfg["paths"]["output"] / "cell_labels.npz"
    np.savez_compressed(out, **lab, cell_px=cell, grid_shape=np.array([grid.lines, grid.samples]),
                        diam_range_km=np.array(d_range), bands_km=np.array(bands))
    written(out)
    n_t = lab["n_target"]
    has = n_t > 0
    summary = {
        "cell_px": cell, "cells_shape": list(shape), "cells": int(n_t.size),
        "pixels_not_in_full_cells": {"rows": grid.lines - shape[0] * cell, "cols": grid.samples - shape[1] * cell},
        "craters_in_region": int(len(cr)), "craters_outside_full_cells": n_outside,
        "target_craters_in_cells": int(n_t.sum()), "craters_above_range_in_cells": int(lab["n_above"].sum()),
        "craters_below_range_in_cells": int(lab["n_below"].sum()),
        "cells_with_target": int(has.sum()), "cells_with_target_fraction": float(has.mean()),
        "cells_with_ge2_target": int((n_t >= 2).sum()), "max_target_per_cell": int(n_t.max()),
        "dx_range": [float(np.nanmin(lab["dx"])), float(np.nanmax(lab["dx"]))],
        "dy_range": [float(np.nanmin(lab["dy"])), float(np.nanmax(lab["dy"]))],
        "classes": {name: int((lab["cls"] == v).sum()) for name, v in
                    (("positive", POSITIVE), ("negative", NEGATIVE), ("ignore", IGNORE))},
        "positive_fraction": float((lab["cls"] == POSITIVE).mean()),
        "positives_per_band_km": {f"{bands[k]:g}-{bands[k + 1]:g}": int((lab["band"] == k).sum())
                                  for k in range(len(bands) - 1)},
        "output": str(out),
    }
    save_report(cfg, "QB2.2-QB2.3", summary)
    return summary


# ---------------------------------------------------------------------------
# QB2.4 - Report on the labels
# ---------------------------------------------------------------------------


def represented_craters(cr, cell_px: int, shape: tuple[int, int], d_range: tuple[float, float]):
    """Target craters in full cells, with their cell and whether they are the one the cell
    represents (the largest of the cell, QB2.2); the others are lost by the one-crater-per-cell labels."""
    ci = np.floor((cr["row"].to_numpy() + 0.5) / cell_px).astype(np.int64)
    cj = np.floor((cr["col"].to_numpy() + 0.5) / cell_px).astype(np.int64)
    d = cr["diam_km"].to_numpy()
    keep = (ci >= 0) & (ci < shape[0]) & (cj >= 0) & (cj < shape[1]) & (d >= d_range[0]) & (d < d_range[1])
    out = cr.loc[keep, ["crater_id", "diam_km"]].copy()
    out["cell"] = ci[keep] * shape[1] + cj[keep]
    out["represented"] = out.groupby("cell")["diam_km"].rank(method="first", ascending=False) == 1
    return out


def run_label_report(cfg: dict) -> dict:
    """QB2.4: multiple centres per cell and class balance per band -> report section QB2.4."""
    import pandas as pd

    lab = np.load(cfg["paths"]["output"] / "cell_labels.npz")
    cls, band, n_t = lab["cls"], lab["band"], lab["n_target"]
    bands = [float(x) for x in lab["bands_km"]]
    names = [f"{bands[k]:g}-{bands[k + 1]:g}" for k in range(len(bands) - 1)]
    pos = cls == POSITIVE
    n_pos, n_neg, n_ign = int(pos.sum()), int((cls == NEGATIVE).sum()), int((cls == IGNORE).sum())

    cr = pd.read_csv(cfg["paths"]["output"] / "craters_R.csv")
    rep = represented_craters(cr[cr["in_region"]], int(lab["cell_px"]), cls.shape, tuple(lab["diam_range_km"]))
    rep["band"] = np.clip(np.searchsorted(bands, rep["diam_km"], side="right") - 1, 0, len(names) - 1)
    if rep["represented"].sum() != n_pos or len(rep) != int(n_t.sum()):
        raise ValueError("crater table and cell labels disagree")

    per_band = {}
    for k, name in enumerate(names):
        cells_k = int((band == k).sum())
        craters_k = rep[rep["band"] == k]
        per_band[name] = {
            "positive_cells": cells_k,
            "share_of_positives": cells_k / n_pos,
            "fraction_of_labelled_cells": cells_k / (n_pos + n_neg),
            "negatives_per_positive": n_neg / cells_k,
            "positive_cells_with_ge2_centres": int(((band == k) & (n_t >= 2)).sum()),
            "craters": int(len(craters_k)),
            "craters_not_represented": int((~craters_k["represented"]).sum()),
            "craters_not_represented_fraction": float((~craters_k["represented"]).mean()),
        }
    report = {
        "cells": int(cls.size),
        "classes": {"positive": n_pos, "negative": n_neg, "ignore": n_ign},
        "positive_fraction_of_labelled_cells": n_pos / (n_pos + n_neg),
        "negatives_per_positive": n_neg / n_pos,
        "positive_cells_by_number_of_centres": {str(k): int(((n_t == k) & pos).sum()) for k in range(1, int(n_t.max()) + 1)},
        "positive_cells_with_ge2_centres_fraction": float((n_t[pos] >= 2).mean()),
        "target_craters": int(len(rep)),
        "craters_not_represented": int((~rep["represented"]).sum()),
        "craters_not_represented_fraction": float((~rep["represented"]).mean()),
        "per_band_km": per_band,
    }
    save_report(cfg, "QB2.4", report)
    return report


def print_label_report(r: dict) -> None:
    c = r["classes"]
    print(f"cells {r['cells']:,}: positive {c['positive']:,} ({100 * r['positive_fraction_of_labelled_cells']:.2f} % of "
          f"labelled), negative {c['negative']:,}, ignore {c['ignore']:,}; negatives per positive {r['negatives_per_positive']:.1f}")
    print(f"positive cells by number of centres: {r['positive_cells_by_number_of_centres']} -> with >= 2: "
          f"{100 * r['positive_cells_with_ge2_centres_fraction']:.2f} %")
    print(f"target craters {r['target_craters']:,}, not represented (not the largest of their cell): "
          f"{r['craters_not_represented']:,} ({100 * r['craters_not_represented_fraction']:.2f} %)")
    print(f"{'band km':>8} {'pos cells':>10} {'% of pos':>9} {'% labelled':>11} {'neg/pos':>8} {'>=2 ctr':>8} {'craters':>8} {'lost':>6} {'% lost':>7}")
    for name, b in r["per_band_km"].items():
        print(f"{name:>8} {b['positive_cells']:>10,} {100 * b['share_of_positives']:>9.1f} {100 * b['fraction_of_labelled_cells']:>11.2f} "
              f"{b['negatives_per_positive']:>8.1f} {b['positive_cells_with_ge2_centres']:>8,} {b['craters']:>8,} "
              f"{b['craters_not_represented']:>6,} {100 * b['craters_not_represented_fraction']:>7.2f}")


# ---------------------------------------------------------------------------
# QB3.1 - Geographic split into blocks
# ---------------------------------------------------------------------------

SETS = ("s1_train", "s1_val", "qk_pool", "test")


def block_of(lat, lon, bounds: dict, block_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Block (row from the north, column from the west) of the points (lat, lon) of R."""
    n_r = round((bounds["lat_max"] - bounds["lat_min"]) / block_deg)
    n_c = round((bounds["lon_max"] - bounds["lon_min"]) / block_deg)
    bi = np.clip(np.floor((bounds["lat_max"] - np.asarray(lat)) / block_deg).astype(np.int64), 0, n_r - 1)
    bj = np.clip(np.floor((np.asarray(lon) - bounds["lon_min"]) / block_deg).astype(np.int64), 0, n_c - 1)
    return bi, bj


def cell_blocks(grid: CylGrid, shape: tuple[int, int], cell_px: int, bounds: dict, block_deg: float) -> np.ndarray:
    """Index (bi * n_block_cols + bj) of the block that contains the centre of every cell."""
    rows = (np.arange(shape[0]) + 0.5) * cell_px - 0.5            # fractional pixel index of the cell centres
    cols = (np.arange(shape[1]) + 0.5) * cell_px - 0.5
    lat, lon = grid.rc_to_latlon(rows, cols)
    bi, _ = block_of(lat, np.full_like(lat, bounds["lon_min"]), bounds, block_deg)
    _, bj = block_of(np.full_like(lon, bounds["lat_max"]), lon, bounds, block_deg)
    n_c = round((bounds["lon_max"] - bounds["lon_min"]) / block_deg)
    return bi[:, None] * n_c + bj[None, :]


def region_cell_blocks(cfg: dict, shape: tuple[int, int]) -> np.ndarray:
    """``cell_blocks`` on the cell grid of R with the tiling, region and block size of the config."""
    return cell_blocks(region_grid(cfg, "wac"), shape, int(cfg["tiling"]["cell_px"]), cfg["region"],
                       float(cfg["split"]["block_deg"]))


def block_area_km2(lat_min, lat_max, dlon_deg: float) -> np.ndarray:
    """Area of a latitude-longitude box on the sphere of radius ``MOON_RADIUS_M``."""
    r_km = MOON_RADIUS_M / 1000.0
    return r_km ** 2 * math.radians(dlon_deg) * (np.sin(np.radians(lat_max)) - np.sin(np.radians(lat_min)))


def assign_split(strata: np.ndarray, fractions: dict, val_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Set of every block: in each stratum the shuffled blocks go first to test and QK-pool
    (``floor(fraction * n)`` each), then to S1-val (``floor(val_fraction * n_s1)``), the rest to S1-train."""
    out = np.empty(len(strata), dtype=object)
    for s in np.unique(strata):
        idx = rng.permutation(np.flatnonzero(strata == s))
        n_test, n_qk = int(fractions["test"] * len(idx)), int(fractions["qk_pool"] * len(idx))
        n_val = int(val_fraction * (len(idx) - n_test - n_qk))
        sizes = {"test": n_test, "qk_pool": n_qk, "s1_val": n_val}
        start = 0
        for name, n in sizes.items():
            out[idx[start:start + n]] = name
            start += n
        out[idx[start:]] = "s1_train"
    return out


def run_split(cfg: dict) -> dict:
    """QB3.1: blocks of R, density strata and sets -> ``<output>/split_blocks.csv``, report section QB3.1."""
    import pandas as pd

    reg, sp = cfg["region"], cfg["split"]
    deg = float(sp["block_deg"])
    n_r = round((reg["lat_max"] - reg["lat_min"]) / deg)
    n_c = round((reg["lon_max"] - reg["lon_min"]) / deg)
    bi, bj = np.divmod(np.arange(n_r * n_c), n_c)
    blocks = pd.DataFrame({"block_id": [f"B{i:02d}_{j:02d}" for i, j in zip(bi, bj)], "bi": bi, "bj": bj})
    blocks["lat_max"] = reg["lat_max"] - deg * bi
    blocks["lat_min"] = blocks["lat_max"] - deg
    blocks["lon_min"] = reg["lon_min"] + deg * bj
    blocks["lon_max"] = blocks["lon_min"] + deg
    blocks["area_km2"] = block_area_km2(blocks["lat_min"], blocks["lat_max"], deg)

    # crater density: target craters (labels.diam_range_km) with the centre in R, per 1,000 km2
    d_range = [float(x) for x in cfg["labels"]["diam_range_km"]]
    cr = pd.read_csv(cfg["paths"]["output"] / "craters_R.csv")
    cr = cr[cr["in_region"] & (cr["diam_km"] >= d_range[0]) & (cr["diam_km"] < d_range[1])]
    cbi, cbj = block_of(cr["lat"], wrap_lon(cr["lon"], (reg["lon_min"] + reg["lon_max"]) / 2), reg, deg)
    blocks["n_target_craters"] = np.bincount(cbi * n_c + cbj, minlength=n_r * n_c)
    blocks["density_per_1000km2"] = 1000.0 * blocks["n_target_craters"] / blocks["area_km2"]
    n_strata = int(sp["n_strata"])
    rank = blocks["density_per_1000km2"].rank(method="first").to_numpy() - 1    # ties broken by block order
    blocks["stratum"] = (rank * n_strata // len(blocks)).astype(int)            # 0 = lowest density
    blocks["set"] = assign_split(blocks["stratum"].to_numpy(), sp["fractions"], float(sp["val_fraction_of_s1"]),
                                 np.random.default_rng(int(cfg["seed"])))

    # cells and labels per set (cells assigned by their centre, before the tile buffer of QB3.2)
    lab = np.load(cfg["paths"]["output"] / "cell_labels.npz")
    cls, band = lab["cls"], lab["band"]
    cb = cell_blocks(region_grid(cfg, "wac"), cls.shape, int(lab["cell_px"]), reg, deg)
    blocks["cells"] = np.bincount(cb.ravel(), minlength=len(blocks))
    blocks["positive_cells"] = np.bincount(cb.ravel(), weights=(cls == POSITIVE).ravel(), minlength=len(blocks)).astype(int)
    out = cfg["paths"]["output"] / "split_blocks.csv"
    blocks.to_csv(out, index=False, float_format="%.6f")
    written(out)

    bands = [float(x) for x in lab["bands_km"]]
    set_of_cell = blocks["set"].to_numpy()[cb]
    per_set = {}
    for name in SETS:
        b = blocks[blocks["set"] == name]
        m = set_of_cell == name
        n_pos, n_neg = int((cls[m] == POSITIVE).sum()), int((cls[m] == NEGATIVE).sum())
        per_set[name] = {
            "blocks": int(len(b)), "blocks_per_stratum": {str(k): int((b["stratum"] == k).sum()) for k in range(n_strata)},
            "area_fraction": float(b["area_km2"].sum() / blocks["area_km2"].sum()),
            "density_per_1000km2": {"mean": float(b["density_per_1000km2"].mean()),
                                    "median": float(b["density_per_1000km2"].median())},
            "mean_abs_lat_centre": float(((b["lat_min"] + b["lat_max"]) / 2).abs().mean()),
            "cells": int(m.sum()), "positive": n_pos, "negative": n_neg, "ignore": int((cls[m] == IGNORE).sum()),
            "positive_fraction_of_labelled": n_pos / (n_pos + n_neg),
            "positives_per_band_km": {f"{bands[k]:g}-{bands[k + 1]:g}": int((band[m] == k).sum()) for k in range(len(bands) - 1)},
        }
    edges = [float(blocks.loc[blocks["stratum"] == k, "density_per_1000km2"].min()) for k in range(n_strata)]
    summary = {
        "block_deg": deg, "block_grid": [n_r, n_c], "blocks": int(len(blocks)), "seed": int(cfg["seed"]),
        "density": {"craters_km": d_range, "min": float(blocks["density_per_1000km2"].min()),
                    "max": float(blocks["density_per_1000km2"].max()), "stratum_lower_edges": edges},
        "per_set": per_set,
        "check_cells": int(blocks["cells"].sum()) == int(cls.size),
        "check_craters": int(blocks["n_target_craters"].sum()) == int(len(cr)),
        "output": str(out),
    }
    save_report(cfg, "QB3.1", summary)
    return summary


def print_split(r: dict) -> None:
    d = r["density"]
    print(f"{r['blocks']} blocks ({r['block_grid'][0]} x {r['block_grid'][1]} of {r['block_deg']:g} deg), seed {r['seed']}; "
          f"density of {d['craters_km'][0]:g}-{d['craters_km'][1]:g} km craters {d['min']:.2f}-{d['max']:.2f} per 1,000 km2, "
          f"strata from {', '.join(f'{e:.2f}' for e in d['stratum_lower_edges'])}")
    bands = list(next(iter(r["per_set"].values()))["positives_per_band_km"])
    print(f"{'set':>9} {'blocks':>6} {'strata':>10} {'area %':>7} {'dens mean':>9} {'|lat|':>6} {'cells':>9} "
          f"{'positive':>9} {'% pos':>6} " + " ".join(f"{b:>8}" for b in bands))
    for name, s in r["per_set"].items():
        print(f"{name:>9} {s['blocks']:>6} {'/'.join(map(str, s['blocks_per_stratum'].values())):>10} "
              f"{100 * s['area_fraction']:>7.1f} {s['density_per_1000km2']['mean']:>9.2f} {s['mean_abs_lat_centre']:>6.1f} "
              f"{s['cells']:>9,} {s['positive']:>9,} {100 * s['positive_fraction_of_labelled']:>6.2f} "
              + " ".join(f"{v:>8,}" for v in s["positives_per_band_km"].values()))
    print(f"checks: all cells in a block {r['check_cells']}, all target craters in a block {r['check_craters']}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def selftest() -> None:
    # WAC-like: 2 tiles 0-60N / 60S-0 with the awkward resolution of the WAC mosaic
    ppd = 303.23350424149
    n_lines, n_samp = 18194, 27291
    wac = {"N": CylGrid(n_lines / ppd, 0.0, ppd, n_lines, n_samp), "S": CylGrid(0.0, 0.0, ppd, n_lines, n_samp)}
    grid, pieces = tile_layout(wac, {"lat_min": -30, "lat_max": 30, "lon_min": 0, "lon_max": 90})
    assert (grid.lines, grid.samples) == (2 * 9097, 27291), (grid.lines, grid.samples)
    byid = {p.product_id: p for p in pieces}
    assert byid["N"].src_rows == slice(9097, 18194) and byid["N"].dst_rows == slice(0, 9097)
    assert byid["S"].src_rows == slice(0, 9097) and byid["S"].dst_rows == slice(9097, 18194)
    # round trip and the equator on the edge between rows 9096 and 9097
    lat, lon = grid.rc_to_latlon(np.array([0.0, 9096.5, 18193.0]), np.array([0.0, 100.0, 27290.0]))
    r, c = grid.latlon_to_rc(lat, lon)
    assert np.allclose(r, [0.0, 9096.5, 18193.0]) and np.allclose(c, [0.0, 100.0, 27290.0])
    assert abs(lat[1]) < 1e-12
    # SLDEM-like 2 x 2 tiles at 512 ppd
    dem = {f"{h}{w}": CylGrid(30.0 if h == "N" else 0.0, w, 512.0, 15360, 23040) for h in "NS" for w in (0.0, 45.0)}
    grid, pieces = tile_layout(dem, {"lat_min": -30, "lat_max": 30, "lon_min": 0, "lon_max": 90})
    assert (grid.lines, grid.samples, grid.north, grid.west) == (30720, 46080, 30.0, 0.0)
    # gap and misalignment are detected
    try:
        tile_layout({"a": dem["N0.0"], "b": dem["S45.0"]}, {"lat_min": -30, "lat_max": 30, "lon_min": 0, "lon_max": 90})
        raise AssertionError("gap not detected")
    except ValueError:
        pass
    try:
        tile_layout({"a": dem["N0.0"], "b": CylGrid(0.0 - 0.3 / 512, 0.0, 512.0, 15360, 23040)},
                    {"lat_min": -30, "lat_max": 30, "lon_min": 0, "lon_max": 45})
        raise AssertionError("misalignment not detected")
    except ValueError:
        pass
    # QB2.1: crater centres -> pixels, region membership, wrap of longitude across 0/360
    import pandas as pd
    g = CylGrid(1.0, 0.0, 10.0, 20, 30)                      # lat -1..1, lon 0..3, 0.1 deg pixels
    cat = pd.DataFrame({"crater_id": list("abcdef"),
                        "lat": [0.95, -0.999, 0.0, 1.15, 0.0, 0.0],
                        "lon": [0.05, 2.999, 359.95, 0.0, 3.05, 3.25],
                        "diam_km": [1.0] * 6})
    cr = craters_on_grid(cat, g, margin_px=2).set_index("crater_id")
    assert set(cr.index) == {"a", "b", "c", "d", "e"}           # f is 2.5 px east of the grid
    assert np.allclose(cr.loc["a", ["row", "col"]].to_numpy(float), [0.0, 0.0])
    assert cr.loc["a", "in_region"] and cr.loc["b", "in_region"]
    assert not cr.loc["c", "in_region"] and np.isclose(cr.loc["c", "col"], -1.0)   # 359.95 E -> 0.5 px west
    assert not cr.loc["d", "in_region"] and not cr.loc["e", "in_region"]
    assert np.allclose(wrap_lon([359.0, 1.0, 181.0], 45.0), [-1.0, 1.0, 181.0])
    # QB2.2: cells of 16 px on a 3 x 2 cell grid
    lab = cell_labels(row=[7.5, 0.0, 15.4, 15.6, 20.0, 40.0, 47.6],
                      col=[7.5, 0.0, 3.0, 3.0, 20.0, 20.0, 31.0],
                      diam_km=[2.0, 5.0, 9.0, 0.5, 1.0, 3.0, 1.5],
                      shape=(3, 2), cell_px=16, d_range=(1.0, 8.0))
    assert lab["n_outside"] == 1                               # row 47.6 -> pixel 48 -> cell row 3
    assert lab["n_target"].tolist() == [[2, 0], [0, 1], [0, 1]]
    assert lab["n_above"][0, 0] == 1 and lab["n_below"][1, 0] == 1   # 15.4 -> cell 0; 15.6 -> cell 1
    assert np.isclose(lab["diam_km"][0, 0], 5.0)                # the largest target crater of the cell
    assert np.isclose(lab["dy"][0, 0], -7.5 / 16) and np.isclose(lab["dx"][0, 0], -7.5 / 16)
    assert np.isclose(lab["log_r"][0, 0], np.log(2.5)) and np.isnan(lab["dx"][0, 1])
    assert np.isclose(lab["dy"][1, 1], (20.0 - 23.5) / 16) and np.isclose(lab["dx"][1, 1], (20.0 - 23.5) / 16)
    # QB2.3: classes and bands (cell (1, 0) has only a D < 1 km centre -> ignore, as D >= 8)
    cls, band = cell_classes(lab, [1, 2, 4, 8])
    assert cls.tolist() == [[POSITIVE, NEGATIVE], [IGNORE, POSITIVE], [NEGATIVE, POSITIVE]]
    assert band.tolist() == [[2, -1], [-1, 0], [-1, 1]]              # 5 km -> 4-8, 1 km -> 1-2, 3 km -> 2-4
    # QB2.4: in cell (0, 0) the 5 km crater is represented, the 2 km one is not
    cr = pd.DataFrame({"crater_id": list("abcde"), "row": [7.5, 0.0, 20.0, 40.0, 15.4], "col": [7.5, 0.0, 20.0, 20.0, 3.0],
                       "diam_km": [2.0, 5.0, 1.0, 3.0, 9.0]})
    rep = represented_craters(cr, 16, (3, 2), (1.0, 8.0)).set_index("crater_id")
    assert list(rep.index) == ["a", "b", "c", "d"] and rep["represented"].tolist() == [False, True, True, True]
    # QB3.1: blocks of cells, areas, stratified split
    bounds = {"lat_min": -1.0, "lat_max": 1.0, "lon_min": 0.0, "lon_max": 3.0}
    cb = cell_blocks(g, (5, 7), 4, bounds, 1.0)   # 0.4 deg cells: centres lat 0.8 ... -0.8, lon 0.2 ... 2.6
    assert cb.shape == (5, 7) and cb[1, 1] == 0 and cb[3, 0] == 3 and cb[0, 3] == 1 and cb[0, 5] == 2 and cb[4, 6] == 5
    assert np.isclose(block_area_km2(-90, 90, 360), 4 * math.pi * (MOON_RADIUS_M / 1000) ** 2)
    strata = np.repeat([0, 1, 2], 72)
    sets = assign_split(strata, {"qk_pool": 0.2, "test": 0.2}, 0.2, np.random.default_rng(0))
    for k in range(3):
        v = sets[strata == k]
        assert [(v == n).sum() for n in SETS] == [36, 8, 14, 14]
    assert (sets == assign_split(strata, {"qk_pool": 0.2, "test": 0.2}, 0.2, np.random.default_rng(0))).all()
    print("selftest OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["region", "craters", "cells", "label-report", "split"], help="step to run")
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if args.command is None:
        ap.error("give a command or --selftest")
    cfg = load_config(args.config)
    if args.command == "region":
        print_report(run_region(cfg))
    elif args.command == "craters":
        print(json.dumps(run_craters(cfg), indent=2))
    elif args.command == "cells":
        print(json.dumps(run_cells(cfg), indent=2))
    elif args.command == "label-report":
        print_label_report(run_label_report(cfg))
    elif args.command == "split":
        print_split(run_split(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
