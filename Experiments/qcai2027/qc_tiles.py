#!/usr/bin/env python3
"""Tiles of the region R (plan QB3.2 and QC7).

Tile grid (QC7.1).  Tiles of ``tiling.tile_px`` (256) px with stride
``tiling.stride_px`` (176) px on the WAC grid of R; tile (ti, tj) covers the
pixels [176 ti, 176 ti + 256) x [176 tj, 176 tj + 256); only tiles fully inside
R are kept (102 x 154 tiles).  The stride is a whole number of cells (11).

Cell ownership (QC7.2, QD8).  Each tile owns only its central 11 x 11 cells:
cells [11 ti + 2, 11 ti + 13) x [11 tj + 2, 11 tj + 13); the central blocks of
adjacent tiles tile the plane, so every cell is owned by at most one tile and
always has at least 2 cells of context.  Cells at the border of R (not covered
by a central block) are owned by no tile.

QB3.2 - Buffer of the geographic split.  A tile takes the set of the 5 deg
block (QB3.1, ``<output>/split_blocks.csv``) that contains its centre, and is
dropped if its footprint (the blocks of its pixel centres) touches a block of
another set; S1-train and S1-val count as different sets.  So tiles of
different sets never share pixels; the cells owned by a dropped tile are not
used.  -> ``<output>/tiles.csv`` (one row per tile), ``<output>/split_cells.npz``
(per cell: owner tile, set of the owner if kept), report section QB3.2.

QC7.1 - Tile grid check (``grid``).  The tiles are not written as files: a tile is
the window ``tile_window`` of the rasters of the stage (QC7.3).  The command checks
the grid against the rasters and ``tiles.csv``: shape, origins on the cell grid,
overlap, every tile inside R, centres, pixels of R covered by 0 / 1 / 2 / 4 tiles,
tile size in km.  -> report section QC7.1.

QC7.2 - Ownership check (``ownership``).  ``owned_cells`` gives, for a tile, its
owned cells as a local window of its 16 x 16 cell (= token) grid and as a window of
the cell grid of R.  The command checks on the real grid that ``owner_tile`` of
``split_cells.npz`` (QB3.2) equals ``owned_cells`` of every tile, that every tile
owns 11 x 11 cells, that no cell is owned twice, the context of the owned cells, and
counts the target craters in the cells owned by no tile.
-> report section QC7.2.

QC7.3 - Tile reading (``TileReader``).  The rasters of the stage stay on disk as
memory-mapped ``.npy`` (WAC and DEM float32, DEM derivatives float16) and a tile is
read as a window, never the whole region in RAM: window (QC7.1) -> validity mask
of WAC AND DEM and fill of the invalid pixels with the tile median, flagged
(QC3.1) -> robust min-max of every layer (QC5.1) -> the 3-channel inputs of the
backbone (``normalisation.backbone_inputs``: wac, dem, early) with the DINOv2
mean / std; also the owned cells of the tile (QC7.2).  The ``read-check`` command
reads every tile once and checks dtypes, values, channel order (through the
inverses) and memory.  -> report section QC7.3.

QC8.2 - Counts (``counts``).  Final dataset of the experiment per set: tiles kept by
the buffer (QB3.2) and valid (QC3.1), sub-tiles owned by them (QC7.2) per class
(positive / negative / ignore) and positives per diameter band, target craters,
flagged sub-tiles; checked against the counts of QB3.2 and for consistency.
-> report section QC8.2 (one row per set + total).

Usage
-----
    python Experiments/qcai2027/qc_tiles.py buffer      # QB3.2
    python Experiments/qcai2027/qc_tiles.py grid        # QC7.1
    python Experiments/qcai2027/qc_tiles.py ownership   # QC7.2
    python Experiments/qcai2027/qc_tiles.py read-check  # QC7.3
    python Experiments/qcai2027/qc_tiles.py counts      # QC8.2
    python Experiments/qcai2027/qc_tiles.py --selftest  # synthetic checks, no data needed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_data import (CONFIG_PATH, IGNORE, NEGATIVE, POSITIVE, SETS, CylGrid, block_of,  # noqa: E402
                     cell_blocks, load_config, load_report, region_grid, save_report, written)

NO_SET = -1


# ---------------------------------------------------------------------------
# Tile grid and cell ownership (QC7.1, QC7.2)
# ---------------------------------------------------------------------------


def tile_grid_shape(lines: int, samples: int, tile_px: int, stride_px: int) -> tuple[int, int]:
    """Number of tile rows and columns fully inside a lines x samples grid."""
    return (lines - tile_px) // stride_px + 1, (samples - tile_px) // stride_px + 1


def tile_window(ti, tj, tile_px: int, stride_px: int) -> tuple[slice, slice]:
    """Pixel window (rows, cols) of tile (ti, tj) on the WAC grid of R."""
    r0, c0 = int(ti) * stride_px, int(tj) * stride_px
    return slice(r0, r0 + tile_px), slice(c0, c0 + tile_px)


def tiles_per_pixel(n_px: int, n_tiles: int, tile_px: int, stride_px: int) -> np.ndarray:
    """Number of tiles covering every pixel along one axis."""
    cover = np.zeros(n_px + 1, dtype=np.int64)
    starts = np.arange(n_tiles) * stride_px
    np.add.at(cover, starts, 1)
    np.add.at(cover, starts + tile_px, -1)
    return cover.cumsum()[:n_px]


def owned_cell_offset(tile_px: int, stride_px: int, cell_px: int) -> int:
    """Offset (in cells) of the central block owned by a tile: (cells per tile - cells per stride) // 2."""
    if stride_px % cell_px or tile_px % cell_px:
        raise ValueError("tile and stride must be whole numbers of cells")
    return (tile_px // cell_px - stride_px // cell_px) // 2


def owned_cells(ti, tj, tile_px: int, stride_px: int, cell_px: int) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """Cells owned by tile (ti, tj): (local window in its tile_px / cell_px cell grid = token grid,
    window in the cell grid of R)."""
    off, k = owned_cell_offset(tile_px, stride_px, cell_px), stride_px // cell_px
    local = (slice(off, off + k), slice(off, off + k))
    r0, c0 = int(ti) * k + off, int(tj) * k + off
    return local, (slice(r0, r0 + k), slice(c0, c0 + k))


def cell_owner(cells_shape: tuple[int, int], tiles_shape: tuple[int, int], tile_px: int, stride_px: int,
               cell_px: int) -> np.ndarray:
    """Index (ti * n_tile_cols + tj) of the tile that owns every cell, -1 if none."""
    k = stride_px // cell_px
    off = owned_cell_offset(tile_px, stride_px, cell_px)

    def axis(n_cells: int, n_tiles: int) -> np.ndarray:
        t = np.full(n_cells, -1, dtype=np.int64)
        i = np.arange(n_cells) - off
        ok = (i >= 0) & (i < n_tiles * k)
        t[ok] = i[ok] // k
        return t

    ti, tj = axis(cells_shape[0], tiles_shape[0]), axis(cells_shape[1], tiles_shape[1])
    owner = ti[:, None] * tiles_shape[1] + tj[None, :]
    owner[(ti[:, None] < 0) | (tj[None, :] < 0)] = -1
    return owner


# ---------------------------------------------------------------------------
# QB3.2 - Buffer of the split
# ---------------------------------------------------------------------------


def tile_sets(grid: CylGrid, tiles_shape: tuple[int, int], tile_px: int, stride_px: int, bounds: dict,
              block_deg: float, block_set: np.ndarray, n_block_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Set (index in ``SETS``) of the block containing each tile centre, and whether the
    footprint of the tile touches only blocks of that set."""
    r0 = np.arange(tiles_shape[0]) * stride_px
    c0 = np.arange(tiles_shape[1]) * stride_px
    # blocks of the first/last pixel centres and of the centre (pixel edge tile_px / 2) of the tiles
    lat_first, lon_first = grid.rc_to_latlon(r0, c0)
    lat_last, lon_last = grid.rc_to_latlon(r0 + tile_px - 1, c0 + tile_px - 1)
    lat_mid, lon_mid = grid.rc_to_latlon(r0 + tile_px / 2 - 0.5, c0 + tile_px / 2 - 0.5)
    lon0 = np.full_like(lat_first, bounds["lon_min"])
    lat0 = np.full_like(lon_first, bounds["lat_max"])
    bi_a, bi_b, bi_m = (block_of(lat, lon0, bounds, block_deg)[0] for lat in (lat_first, lat_last, lat_mid))
    bj_a, bj_b, bj_m = (block_of(lat0, lon, bounds, block_deg)[1] for lon in (lon_first, lon_last, lon_mid))

    centre_set = block_set[bi_m[:, None] * n_block_cols + bj_m[None, :]]
    clean = np.ones(centre_set.shape, dtype=bool)
    # footprint blocks: rows bi_a..bi_b x cols bj_a..bj_b (at most 2 x 2 since a tile is smaller than a block)
    for bi in (bi_a, bi_b):
        for bj in (bj_a, bj_b):
            clean &= block_set[bi[:, None] * n_block_cols + bj[None, :]] == centre_set
    return centre_set, clean


def neighbour_pairs(s: np.ndarray, di: int, dj: int) -> tuple[np.ndarray, np.ndarray]:
    """Views a, b with b[i, j] = s[i + di, j + dj] for every pair inside the array (di >= 0)."""
    h, w = s.shape
    return (s[:h - di, max(0, -dj):w - max(0, dj)], s[di:, max(0, dj):w - max(0, -dj)])


def run_buffer(cfg: dict) -> dict:
    """QB3.2: set of every tile, tiles dropped by the buffer, cells used per set."""
    import pandas as pd

    reg, deg = cfg["region"], float(cfg["split"]["block_deg"])
    tile_px, stride_px, cell_px = (int(cfg["tiling"][k]) for k in ("tile_px", "stride_px", "cell_px"))
    grid = region_grid(cfg, "wac")
    blocks = pd.read_csv(cfg["paths"]["output"] / "split_blocks.csv")
    n_block_cols = int(blocks["bj"].max()) + 1
    block_set = blocks.sort_values(["bi", "bj"])["set"].map({s: k for k, s in enumerate(SETS)}).to_numpy()

    tshape = tile_grid_shape(grid.lines, grid.samples, tile_px, stride_px)
    centre_set, clean = tile_sets(grid, tshape, tile_px, stride_px, reg, deg, block_set, n_block_cols)
    ti, tj = np.divmod(np.arange(tshape[0] * tshape[1]), tshape[1])
    lat_c, lon_c = grid.rc_to_latlon(ti * stride_px + tile_px / 2 - 0.5, tj * stride_px + tile_px / 2 - 0.5)
    tiles = pd.DataFrame({"tile": np.arange(len(ti)), "ti": ti, "tj": tj, "row0": ti * stride_px, "col0": tj * stride_px,
                          "lat_centre": lat_c, "lon_centre": lon_c,
                          "set": np.array(SETS, dtype=object)[centre_set.ravel()], "kept": clean.ravel()})

    lab = np.load(cfg["paths"]["output"] / "cell_labels.npz")
    cls, band = lab["cls"], lab["band"]
    owner = cell_owner(cls.shape, tshape, tile_px, stride_px, cell_px)
    owned = owner >= 0
    cell_set = np.full(cls.shape, NO_SET, dtype=np.int8)
    use = owned & clean.ravel()[np.where(owned, owner, 0)]
    cell_set[use] = centre_set.ravel()[owner[use]]
    out_cells = cfg["paths"]["output"] / "split_cells.npz"
    np.savez_compressed(out_cells, owner_tile=owner.astype(np.int32), cell_set=cell_set, sets=np.array(SETS))
    out_tiles = cfg["paths"]["output"] / "tiles.csv"
    tiles.to_csv(out_tiles, index=False, float_format="%.6f")
    written(out_cells)
    written(out_tiles)

    # checks: (1) overlapping kept tiles (neighbours) have the same set; (2) the cells used by a set
    # have their centre in a block of that set (consistent with QB3.1)
    s = np.where(clean, centre_set, NO_SET)
    conflicts = sum(int(((a >= 0) & (b >= 0) & (a != b)).sum())
                    for a, b in (neighbour_pairs(s, di, dj) for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1))))
    cb = cell_blocks(grid, cls.shape, cell_px, reg, deg)
    block_set_of_cell = block_set[cb]

    bands = [float(x) for x in lab["bands_km"]]
    per_set = {}
    for k, name in enumerate(SETS):
        in_blocks = block_set_of_cell == k                         # cells of the set by QB3.1 (cell centre)
        m = cell_set == k
        n_pos, n_neg = int((cls[m] == POSITIVE).sum()), int((cls[m] == NEGATIVE).sum())
        t = tiles[tiles["set"] == name]
        per_set[name] = {
            "tiles_with_centre_in_set": int(len(t)), "tiles_kept": int(t["kept"].sum()),
            "tiles_dropped": int((~t["kept"]).sum()), "tiles_dropped_fraction": float((~t["kept"]).mean()),
            "cells_in_blocks": int(in_blocks.sum()), "cells_used": int(m.sum()),
            "cells_used_fraction": float(m.sum() / in_blocks.sum()),
            "positive": n_pos, "negative": n_neg, "ignore": int((cls[m] == IGNORE).sum()),
            "positive_fraction_of_labelled": n_pos / (n_pos + n_neg),
            "positives_per_band_km": {f"{bands[b]:g}-{bands[b + 1]:g}": int((band[m] == b).sum()) for b in range(len(bands) - 1)},
            "positives_in_blocks_used_fraction": float((m & (cls == POSITIVE)).sum() / (in_blocks & (cls == POSITIVE)).sum()),
        }
    summary = {
        "tile_px": tile_px, "stride_px": stride_px, "tiles_shape": list(tshape), "tiles": int(len(tiles)),
        "tiles_kept": int(tiles["kept"].sum()), "tiles_dropped": int((~tiles["kept"]).sum()),
        "pixels_not_in_tiles": {"rows": grid.lines - ((tshape[0] - 1) * stride_px + tile_px),
                                "cols": grid.samples - ((tshape[1] - 1) * stride_px + tile_px)},
        "cells": int(cls.size), "cells_owned_by_a_tile": int(owned.sum()),
        "cells_not_owned_border_of_R": int((~owned).sum()), "cells_used": int((cell_set >= 0).sum()),
        "per_set": per_set,
        "check_overlapping_kept_tiles_of_different_sets": conflicts,
        "check_used_cells_outside_their_set_blocks": int(((cell_set >= 0) & (cell_set != block_set_of_cell)).sum()),
        "outputs": [str(out_tiles), str(out_cells)],
    }
    save_report(cfg, "QB3.2", summary)
    return summary


def print_buffer(r: dict) -> None:
    print(f"tiles {r['tiles_shape'][0]} x {r['tiles_shape'][1]} = {r['tiles']:,} ({r['tile_px']} px, stride {r['stride_px']}): "
          f"kept {r['tiles_kept']:,}, dropped by the buffer {r['tiles_dropped']:,}; pixels of R not in a tile: {r['pixels_not_in_tiles']}")
    print(f"cells {r['cells']:,}: owned by a tile {r['cells_owned_by_a_tile']:,} (border of R not owned: "
          f"{r['cells_not_owned_border_of_R']:,}), used {r['cells_used']:,}")
    bands = list(next(iter(r["per_set"].values()))["positives_per_band_km"])
    print(f"{'set':>9} {'tiles':>6} {'kept':>6} {'% drop':>7} {'cells blk':>10} {'used':>9} {'% used':>7} {'positive':>9} "
          f"{'% pos':>6} {'%pos used':>9} " + " ".join(f"{b:>7}" for b in bands))
    for name, s in r["per_set"].items():
        print(f"{name:>9} {s['tiles_with_centre_in_set']:>6,} {s['tiles_kept']:>6,} {100 * s['tiles_dropped_fraction']:>7.1f} "
              f"{s['cells_in_blocks']:>10,} {s['cells_used']:>9,} {100 * s['cells_used_fraction']:>7.1f} {s['positive']:>9,} "
              f"{100 * s['positive_fraction_of_labelled']:>6.2f} {100 * s['positives_in_blocks_used_fraction']:>9.1f} "
              + " ".join(f"{v:>7,}" for v in s["positives_per_band_km"].values()))
    print(f"checks: overlapping kept tiles of different sets {r['check_overlapping_kept_tiles_of_different_sets']}, "
          f"used cells outside the blocks of their set {r['check_used_cells_outside_their_set_blocks']}")


# ---------------------------------------------------------------------------
# QC7.1 - Tile grid check
# ---------------------------------------------------------------------------


def run_grid(cfg: dict) -> dict:
    """QC7.1: geometry of the tile grid on the WAC grid of R, checked against the rasters and tiles.csv."""
    import pandas as pd

    from qc_preprocess import pixel_m, raster_path

    tile_px, stride_px, cell_px = (int(cfg["tiling"][k]) for k in ("tile_px", "stride_px", "cell_px"))
    grid = region_grid(cfg, "wac")
    shapes = {n: list(np.load(raster_path(cfg, n), mmap_mode="r").shape) for n in cfg["preprocess"]["layers"]}
    tshape = tile_grid_shape(grid.lines, grid.samples, tile_px, stride_px)
    tiles = pd.read_csv(cfg["paths"]["output"] / "tiles.csv")
    ti, tj = tiles["ti"].to_numpy(), tiles["tj"].to_numpy()

    # (1) tiles.csv = the grid: one row per (ti, tj) in row-major order, origins = tile_window
    wins = [tile_window(a, b, tile_px, stride_px) for a, b in zip(ti, tj)]
    csv_ok = (len(tiles) == tshape[0] * tshape[1]
              and np.array_equal(tiles["tile"].to_numpy(), ti * tshape[1] + tj)
              and np.array_equal(tiles["row0"].to_numpy(), [w[0].start for w in wins])
              and np.array_equal(tiles["col0"].to_numpy(), [w[1].start for w in wins]))
    inside = all(w[0].start >= 0 and w[1].start >= 0 and w[0].stop <= grid.lines and w[1].stop <= grid.samples for w in wins)
    # one more tile row / column would leave R
    maximal = (tshape[0] * stride_px + tile_px > grid.lines) and (tshape[1] * stride_px + tile_px > grid.samples)
    # (2) centres: lat/lon of tiles.csv = centre of the window (pixel edge tile_px / 2)
    lat_c, lon_c = grid.rc_to_latlon(tiles["row0"].to_numpy() + tile_px / 2 - 0.5, tiles["col0"].to_numpy() + tile_px / 2 - 0.5)
    centre_err = float(max(np.abs(lat_c - tiles["lat_centre"]).max(), np.abs(lon_c - tiles["lon_centre"]).max()))
    # (3) coverage of R: tiles per pixel (product of the two axes)
    cr = tiles_per_pixel(grid.lines, tshape[0], tile_px, stride_px)
    cc = tiles_per_pixel(grid.samples, tshape[1], tile_px, stride_px)
    cover = {int(k): int(np.sum(cr[:, None] * cc[None, :] == k)) for k in (0, 1, 2, 4)}
    if sum(cover.values()) != grid.lines * grid.samples:
        raise AssertionError("pixels covered by a number of tiles other than 0, 1, 2, 4")
    # (4) size of a tile on the ground
    px = pixel_m(grid)
    lat_edges = np.array([grid.north, grid.north - grid.lines / grid.ppd])
    report = {
        "raster_shapes": shapes, "grid_shape": [grid.lines, grid.samples],
        "rasters_on_the_grid": all(v == [grid.lines, grid.samples] for v in shapes.values()),
        "tile_px": tile_px, "stride_px": stride_px, "overlap_px": tile_px - stride_px,
        "stride_cells": stride_px / cell_px, "tile_cells": tile_px / cell_px,
        "tiles_shape": list(tshape), "tiles": tshape[0] * tshape[1],
        "check_tiles_csv_equal_grid": bool(csv_ok), "check_all_tiles_inside_R": bool(inside),
        "check_no_further_tile_fits": bool(maximal), "check_centre_max_error_deg": centre_err,
        "pixels_not_covered": {"rows_south": int(grid.lines - ((tshape[0] - 1) * stride_px + tile_px)),
                               "cols_east": int(grid.samples - ((tshape[1] - 1) * stride_px + tile_px))},
        "pixels_by_number_of_tiles": cover,
        "fraction_by_number_of_tiles": {k: v / (grid.lines * grid.samples) for k, v in cover.items()},
        "tile_size_km": {"ns": tile_px * px / 1000.0,
                         "ew_equator": tile_px * px / 1000.0,
                         "ew_at_edges_of_R": float(tile_px * px / 1000.0 * np.cos(np.radians(np.abs(lat_edges).max())))},
    }
    save_report(cfg, "QC7.1", report)
    return report


# ---------------------------------------------------------------------------
# QC7.2 - Ownership check
# ---------------------------------------------------------------------------


def run_ownership(cfg: dict) -> dict:
    """QC7.2: every cell owned at most once, by the tile in which it is central; checked on the real grid."""
    from qc_preprocess import pixel_m

    tile_px, stride_px, cell_px = (int(cfg["tiling"][k]) for k in ("tile_px", "stride_px", "cell_px"))
    grid = region_grid(cfg, "wac")
    tshape = tile_grid_shape(grid.lines, grid.samples, tile_px, stride_px)
    lab = np.load(cfg["paths"]["output"] / "cell_labels.npz")
    n_target = lab["n_target"]
    stored = np.load(cfg["paths"]["output"] / "split_cells.npz")["owner_tile"]
    n_tile_cells = tile_px // cell_px

    # (1) owner map rebuilt tile by tile from owned_cells: no cell twice, 11 x 11 per tile,
    #     equal to the stored map of QB3.2 and to cell_owner
    rebuilt = np.full(n_target.shape, -1, dtype=np.int64)
    twice, sizes = 0, set()
    for ti in range(tshape[0]):
        for tj in range(tshape[1]):
            (lr, lc), (gr, gc) = owned_cells(ti, tj, tile_px, stride_px, cell_px)
            block = rebuilt[gr, gc]
            twice += int((block >= 0).sum())
            sizes.add(block.shape)
            rebuilt[gr, gc] = ti * tshape[1] + tj
            # the local window is the same cells as the global one (tile origin in cells + local index)
            assert gr.start == ti * (stride_px // cell_px) + lr.start and gc.start == tj * (stride_px // cell_px) + lc.start
    ctx = {"top_left_cells": owned_cell_offset(tile_px, stride_px, cell_px),
           "bottom_right_cells": n_tile_cells - owned_cell_offset(tile_px, stride_px, cell_px) - stride_px // cell_px}
    owned = rebuilt >= 0
    # (2) owned cells per row / column of R: the central blocks of adjacent tiles leave no gap
    rows_owned = np.flatnonzero(owned.any(1))
    cols_owned = np.flatnonzero(owned.any(0))
    contiguous = bool(np.all(np.diff(rows_owned) == 1) and np.all(np.diff(cols_owned) == 1)
                      and owned[rows_owned[0]:rows_owned[-1] + 1, cols_owned[0]:cols_owned[-1] + 1].all())
    report = {
        "tile_cells": n_tile_cells, "owned_per_tile": [list(x) for x in sizes],
        "context_cells": ctx, "context_km_min": min(ctx.values()) * cell_px * pixel_m(grid) / 1000.0,
        "tiles": tshape[0] * tshape[1], "cells_shape": list(n_target.shape), "cells": int(n_target.size),
        "cells_owned": int(owned.sum()), "cells_owned_expected": tshape[0] * tshape[1] * (stride_px // cell_px) ** 2,
        "cells_not_owned": int((~owned).sum()),
        "owned_cell_rows": [int(rows_owned[0]), int(rows_owned[-1])], "owned_cell_cols": [int(cols_owned[0]), int(cols_owned[-1])],
        "check_no_cell_owned_twice": twice == 0, "check_owned_block_contiguous": contiguous,
        "check_equal_split_cells_npz": bool(np.array_equal(rebuilt, stored)),
        "check_equal_cell_owner": bool(np.array_equal(rebuilt, cell_owner(n_target.shape, tshape, tile_px, stride_px, cell_px))),
        "target_craters": int(n_target.sum()), "target_craters_in_owned_cells": int(n_target[owned].sum()),
        "target_craters_in_cells_not_owned": int(n_target[~owned].sum()),
        "positive_cells_not_owned": int((lab["cls"][~owned] == POSITIVE).sum()),
    }
    report["pass"] = bool(report["check_no_cell_owned_twice"] and contiguous and report["check_equal_split_cells_npz"]
                          and report["check_equal_cell_owner"] and report["cells_owned"] == report["cells_owned_expected"]
                          and sizes == {(stride_px // cell_px,) * 2})
    save_report(cfg, "QC7.2", report)
    return report


# ---------------------------------------------------------------------------
# QC7.3 - Tile reading
# ---------------------------------------------------------------------------


class TileReader:
    """Tiles of R read as windows of the memory-mapped rasters of the stage (QC7.3), masked and
    filled (QC3.1) and normalised for the backbone (QC5.1)."""

    MASK_LAYERS = ("wac", "dem")          # QC3.1: valid WAC AND valid DEM

    def __init__(self, cfg: dict, inputs: tuple[str, ...] | None = None):
        from qc_preprocess import load_normalisation, raster_path

        self.tile_px, self.stride_px, self.cell_px = (int(cfg["tiling"][k]) for k in ("tile_px", "stride_px", "cell_px"))
        self.norm = load_normalisation(cfg)
        self.inputs = {k: v for k, v in self.norm["backbone_inputs"].items() if inputs is None or k in inputs}
        names = list(dict.fromkeys([n for v in self.inputs.values() for n in v] + list(self.MASK_LAYERS)))
        self.layers = {n: np.load(raster_path(cfg, n), mmap_mode="r") for n in names}
        g = next(iter(self.layers.values())).shape
        self.tiles_shape = tile_grid_shape(g[0], g[1], self.tile_px, self.stride_px)
        validity = np.load(cfg["paths"]["output"] / "tile_validity.npz")
        self.tile_used, self.min_valid = validity["tile_used"], float(validity["min_tile_valid_fraction"])
        self.mean, self.std = self.norm["backbone"]["mean"], self.norm["backbone"]["std"]

    def raw(self, ti: int, tj: int, name: str) -> np.ndarray:
        """Window of one layer in physical units (float32), as stored."""
        return np.asarray(self.layers[name][tile_window(ti, tj, self.tile_px, self.stride_px)], dtype=np.float32)

    def read(self, ti: int, tj: int, unit: bool = False) -> dict:
        """Tile (ti, tj): ``inputs`` {name: (3, H, W) float32 backbone input, or layers in [0, 1] if
        ``unit``}, ``valid`` (H, W) bool (False = filled pixel, never data), ``n_filled``, and
        ``owned_local`` / ``owned_cells`` (QC7.2)."""
        from qc_preprocess import backbone_input, fill_invalid, to_unit, valid_mask

        if not self.tile_used[ti, tj]:
            raise ValueError(f"tile ({ti}, {tj}) has < {self.min_valid} valid pixels (QC3.1)")
        raw = {n: self.raw(ti, tj, n) for n in self.layers}
        valid = valid_mask(*(raw[n] for n in self.MASK_LAYERS))
        u = {n: to_unit(fill_invalid(x, valid), self.norm["layers"][n]) for n, x in raw.items() if n in self.norm["layers"]}
        out = {k: (np.stack([u[n] for n in v]) if unit else backbone_input([u[n] for n in v], self.mean, self.std))
               for k, v in self.inputs.items()}
        local, cells = owned_cells(ti, tj, self.tile_px, self.stride_px, self.cell_px)
        return {"inputs": out, "valid": valid, "n_filled": int((~valid).sum()), "owned_local": local, "owned_cells": cells}


def run_read_check(cfg: dict, n_inverse: int = 20) -> dict:
    """QC7.3: storage of the stage; every tile read once (finite, in range, no fill); channel order
    through the inverses on random tiles; memory allocated while reading."""
    import time
    import tracemalloc

    from qc_preprocess import backbone_to_unit, from_unit, raster_path

    reader = TileReader(cfg)
    storage = {n: {"path": str(raster_path(cfg, n)), "dtype": str(a.dtype), "shape": list(a.shape),
                   "memory_mapped": isinstance(a, np.memmap), "size_gb": raster_path(cfg, n).stat().st_size / 1e9}
               for n, a in reader.layers.items()}
    lo = {k: [(0.0 - m) / s for m, s in zip(reader.mean, reader.std)] for k in reader.inputs}
    hi = {k: [(1.0 - m) / s for m, s in zip(reader.mean, reader.std)] for k in reader.inputs}

    # (1) every used tile once, in row-major order
    n_tiles, n_filled, bad_values, shapes = 0, 0, 0, set()
    tracemalloc.start()
    t0 = time.perf_counter()
    for ti in range(reader.tiles_shape[0]):
        for tj in range(reader.tiles_shape[1]):
            if not reader.tile_used[ti, tj]:
                continue
            t = reader.read(ti, tj)
            n_tiles += 1
            n_filled += t["n_filled"]
            for k, x in t["inputs"].items():
                shapes.add((k, x.shape, str(x.dtype)))
                for c in range(x.shape[0]):
                    xc = x[c]
                    bad_values += int((~np.isfinite(xc)).sum() + (xc < lo[k][c] - 1e-5).sum() + (xc > hi[k][c] + 1e-5).sum())
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # (2) channel order and values: the inverses give back the stored layers where not clipped
    rng = np.random.default_rng(int(cfg["seed"]))
    used = np.argwhere(reader.tile_used)
    max_err = {}
    for ti, tj in used[rng.choice(len(used), n_inverse, replace=False)]:
        t = reader.read(int(ti), int(tj))
        for k, names in reader.inputs.items():
            back = backbone_to_unit(t["inputs"][k], reader.mean, reader.std)
            for c, n in enumerate(names):
                p = reader.norm["layers"][n]
                raw = reader.raw(int(ti), int(tj), n).astype(np.float64)
                inside = (raw >= p["low"]) & (raw <= p["high"])
                err = float((np.abs(from_unit(back[c], p) - raw)[inside]).max() / (p["high"] - p["low"]))
                max_err[f"{k}[{c}]={n}"] = max(err, max_err.get(f"{k}[{c}]={n}", 0.0))
    # (3) a tile not used is refused (none in R: checked on a copy of the mask)
    tiles_used = int(reader.tile_used.sum())
    reader.tile_used = reader.tile_used.copy()
    reader.tile_used[0, 0] = False
    try:
        reader.read(0, 0)
        refused = False
    except ValueError:
        refused = True

    report = {
        "storage": storage, "inputs": reader.inputs,
        "tiles_read": n_tiles, "tiles_used": tiles_used,
        "output_shapes": sorted([list(x) for x in shapes]),
        "filled_pixels": n_filled, "values_not_finite_or_out_of_range": bad_values,
        "seconds": elapsed, "ms_per_tile": 1000 * elapsed / n_tiles,
        "peak_allocated_mb_while_reading": peak / 1e6,
        "stage_rasters_gb": sum(v["size_gb"] for v in storage.values()),
        "check_inverse_tiles": n_inverse, "check_inverse_max_error_fraction_of_range": max_err,
        "check_unused_tile_refused": refused,
    }
    save_report(cfg, "QC7.3", report)
    return report


# ---------------------------------------------------------------------------
# QC8.2 - Counts
# ---------------------------------------------------------------------------


def dataset_cells(tiles, tile_used: np.ndarray, owner: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sub-tiles of the dataset (QC8.2): a sub-tile is in it if its owner tile (QC7.2) is kept by
    the buffer (QB3.2) and valid (QC3.1); its set is the set of the owner.  Returns the used
    tiles (flat, as ``tiles.csv``), the cell mask ``in_data`` and the set name per cell ("" outside)."""
    used_tile = tiles["kept"].to_numpy(bool) & tile_used.ravel()
    own = np.where(owner >= 0, owner, 0)
    in_data = (owner >= 0) & used_tile[own]
    return used_tile, in_data, np.where(in_data, tiles["set"].to_numpy()[own], "")


def run_counts(cfg: dict) -> dict:
    """QC8.2: tiles and owned sub-tiles per set and class after buffer, mask and ownership."""
    import pandas as pd

    tiles = pd.read_csv(cfg["paths"]["output"] / "tiles.csv")
    validity = np.load(cfg["paths"]["output"] / "tile_validity.npz")
    split = np.load(cfg["paths"]["output"] / "split_cells.npz")
    lab = np.load(cfg["paths"]["output"] / "cell_labels.npz")
    cls, band, n_target = lab["cls"], lab["band"], lab["n_target"]
    bands = [float(x) for x in lab["bands_km"]]
    owner, cell_set = split["owner_tile"], split["cell_set"]
    tile_set = tiles["set"].to_numpy()
    valid = validity["tile_used"].ravel()
    kept = tiles["kept"].to_numpy(bool)
    used_tile, in_data, set_of_cell = dataset_cells(tiles, validity["tile_used"], owner)
    flagged = validity["cell_invalid_px"] > 0

    rows = []
    for name in list(SETS) + ["total"]:
        m = in_data & (set_of_cell == name) if name != "total" else in_data
        t = (tile_set == name) if name != "total" else np.ones(len(tiles), bool)
        n_pos, n_neg, n_ign = (int((cls[m] == c).sum()) for c in (POSITIVE, NEGATIVE, IGNORE))
        rows.append({
            "set": name, "tiles_centre_in_set": int(t.sum()), "tiles_kept_buffer": int((t & kept).sum()),
            "tiles_valid": int((t & valid).sum()), "tiles_used": int((t & used_tile).sum()),
            "subtiles": int(m.sum()), "positive": n_pos, "negative": n_neg, "ignore": n_ign,
            "positive_fraction_of_labelled": n_pos / (n_pos + n_neg),
            "negatives_per_positive": n_neg / n_pos,
            **{f"positive_{bands[b]:g}_{bands[b + 1]:g}_km": int((band[m] == b).sum()) for b in range(len(bands) - 1)},
            "target_craters": int(n_target[m].sum()), "subtiles_flagged": int((m & flagged).sum()),
        })
    table = pd.DataFrame(rows)

    # checks: (1) the sub-tiles in the dataset are exactly the used cells of QB3.2 with the same set
    #         (the mask removed no tile, QC3.1); (2) every sub-tile is in one set (sets sum to the
    #         total); (3) every used tile owns 121 sub-tiles; (4) same counts as split_tiles_summary.json
    qb32 = load_report(cfg, "QB3.2")["per_set"]
    set_index = {k: i for i, k in enumerate(SETS)}
    same_cells = bool(np.array_equal(in_data, cell_set >= 0)
                      and all((cell_set[in_data & (set_of_cell == k)] == i).all() for k, i in set_index.items()))
    per = table.set_index("set")
    sums_ok = all(int(per.loc[list(SETS), c].sum()) == int(per.loc["total", c])
                  for c in ("subtiles", "positive", "negative", "ignore", "target_craters", "tiles_used"))
    owned_per_tile = np.bincount(owner[in_data], minlength=len(tiles))[used_tile]
    qb32_equal = all(per.loc[k, "tiles_used"] == qb32[k]["tiles_kept"] and per.loc[k, "subtiles"] == qb32[k]["cells_used"]
                     and per.loc[k, "positive"] == qb32[k]["positive"] and per.loc[k, "negative"] == qb32[k]["negative"]
                     and per.loc[k, "ignore"] == qb32[k]["ignore"] for k in SETS)
    report = {
        "per_set": rows,
        "subtiles_owned_by_no_tile_border_of_R": int((owner < 0).sum()),
        "subtiles_of_tiles_dropped_by_the_buffer": int(((owner >= 0) & ~in_data).sum()),
        "check_subtiles_equal_qb32_used_cells": same_cells, "check_sets_sum_to_total": bool(sums_ok),
        "check_121_subtiles_per_used_tile": bool((owned_per_tile == 121).all()),
        "check_counts_equal_split_tiles_summary": bool(qb32_equal),
    }
    save_report(cfg, "QC8.2", report)
    return report


def print_counts(r: dict) -> None:
    keys = [k for k in r["per_set"][0] if k.startswith("positive_") and k.endswith("_km")]
    print(f"{'set':>9} {'tiles':>6} {'used':>6} {'subtiles':>9} {'positive':>9} {'negative':>10} {'ignore':>7} "
          f"{'% pos':>6} {'neg/pos':>8} " + " ".join(f"{k[9:]:>9}" for k in keys) + f" {'craters':>8}")
    for x in r["per_set"]:
        print(f"{x['set']:>9} {x['tiles_centre_in_set']:>6,} {x['tiles_used']:>6,} {x['subtiles']:>9,} {x['positive']:>9,} "
              f"{x['negative']:>10,} {x['ignore']:>7,} {100 * x['positive_fraction_of_labelled']:>6.2f} "
              f"{x['negatives_per_positive']:>8.1f} " + " ".join(f"{x[k]:>9,}" for k in keys) + f" {x['target_craters']:>8,}")
    print({k: v for k, v in r.items() if k != "per_set"})


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def selftest() -> None:
    assert tile_grid_shape(18194, 27291, 256, 176) == (102, 154)
    assert owned_cell_offset(256, 176, 16) == 2
    # 3 x 2 tiles of 256 px stride 176 on a 784 x 608 px grid -> 49 x 38 cells
    owner = cell_owner((49, 38), (3, 2), 256, 176, 16)
    assert (owner[:2] == -1).all() and (owner[:, :2] == -1).all() and (owner[35:] == -1).all() and (owner[:, 24:] == -1).all()
    assert owner[2, 2] == 0 and owner[12, 12] == 0 and owner[13, 2] == 2 and owner[2, 13] == 1 and owner[34, 23] == 5
    counts = np.bincount(owner[owner >= 0])
    assert counts.tolist() == [121] * 6                              # 11 x 11 cells per tile, each cell once
    # every owned cell is inside the footprint of its tile, at >= 2 cells from its border
    for t in range(6):
        i, j = np.nonzero(owner == t)
        ti, tj = divmod(t, 2)
        assert i.min() - 11 * ti >= 2 and 11 * ti + 16 - 1 - i.max() >= 2 and j.min() - 11 * tj >= 2
    # buffer: 0.1 deg pixels, blocks of 1 deg (10 px), tiles of 4 px stride 3 on a 20 x 30 px grid
    g = CylGrid(1.0, 0.0, 10.0, 20, 30)
    bounds = {"lat_min": -1.0, "lat_max": 1.0, "lon_min": 0.0, "lon_max": 3.0}
    block_set = np.array([0, 0, 2, 0, 0, 3])                       # 2 x 3 blocks; sets differ only in column 2
    tshape = tile_grid_shape(20, 30, 4, 3)
    assert tshape == (6, 9)
    cset, clean = tile_sets(g, tshape, 4, 3, bounds, 1.0, block_set, 3)
    # tile columns: tj = 6 covers px 18-21 (blocks 1 and 2) -> centre at px 20 (block 2) -> set 2/3, dropped
    assert (cset[:, 6] >= 2).all() and not clean[:, 6].any()
    assert clean[:, :6].all() and (cset[:, :6] == 0).all()        # sets 0 across the row border: same set, kept
    assert clean[:3, 7:].all() and (cset[:3, 7:] == 2).all() and (cset[4:, 7:] == 3).all()
    assert not clean[3, 7:].any()                                  # tiles crossing the 2/3 border
    a, b = neighbour_pairs(np.arange(12).reshape(3, 4), 1, -1)
    assert (b - a == 3).all() and a.shape == (2, 3)
    # QC7.1: windows and coverage
    assert tile_window(2, 3, 256, 176) == (slice(352, 608), slice(528, 784))
    cov = tiles_per_pixel(784, 3, 256, 176)                          # tiles at 0, 176, 352
    assert cov[:176].tolist() == [1] * 176 and cov[176:256].tolist() == [2] * 80 and cov[608:].tolist() == [0] * 176
    # QC7.2: owned_cells agrees with cell_owner
    for t in range(6):
        (lr, lc), (gr, gc) = owned_cells(*divmod(t, 2), 256, 176, 16)
        assert (lr, lc) == (slice(2, 13), slice(2, 13)) and (owner[gr, gc] == t).all()
    print("selftest OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["buffer", "grid", "ownership", "read-check", "counts"])
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if args.command is None:
        ap.error("give a command or --selftest")
    cfg = load_config(args.config)
    if args.command == "buffer":
        print_buffer(run_buffer(cfg))
    elif args.command == "grid":
        print(json.dumps(run_grid(cfg), indent=2))
    elif args.command == "ownership":
        print(json.dumps(run_ownership(cfg), indent=2))
    elif args.command == "read-check":
        print(json.dumps(run_read_check(cfg), indent=2))
    elif args.command == "counts":
        print_counts(run_counts(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
