#!/usr/bin/env python3
"""Pre-processing of the region R (plan phase QC, proxy option "1 clean").

The rasters of R are written as a checkpoint ("stage") of the cleaned dataset,
``Dataset/Moon/preprocessing/clean_dataset/<preprocess.stage>/`` (folder from
``data_inventory.clean_dataset_dir`` of the dataset scripts), never over the
original products.  A stage is never modified by a later step: a raster that
already exists is not rewritten unless ``--force`` is given.  Every raster is a
memory-mapped ``.npy`` array on the WAC grid of R with a ``.json`` sidecar
(grid, sources, no-data, statistics).

Commands
--------
wac      QC1.1: WAC mosaic of R in physical values (``qc_data.read_region``:
         no-data sentinels -> NaN, the D15 shift applied to the mosaic of the
         two hemisphere tiles, crop to R), float32, written by row strips.
         -> ``<stage>/LROC/WAC_GLOBAL_100M/wac.npy`` (+ ``.json``), report section QC1.1.

dem      QC1.2: SLDEM2015 height of R (``read_dem_height``): the 4 tiles read as
         one mosaic by ``read_region`` (``read_physical`` adds the label OFFSET,
         giving the radius in km), converted to height in m above the 1,737.4 km
         sphere with ``diag_dem_sources.height_m``.  Light step applied at
         reading, no raster written: the 4 tiles cover R exactly (no crop, no
         resampling), the DEM checkpoint is the resampled DEM of QC2.
         -> report section QC1.2 (checks and statistics).

dem-grid QC2.1: DEM height resampled to the WAC grid of R (``dem_on_grid``, as
         ``diag_coregistration.dem_sample``): mask-aware mean of k x k DEM pixels
         (``diag_lamp_snr.block_mean``, k = round(WAC pixel / DEM pixel) = 2, i.e.
         ~118 m: the anti-aliasing step, it also removes the SLDEM stripes
         < ~470 m) then bilinear sampling at the WAC pixel centres; float32 m.
         -> ``<stage>/<layers.dem>`` (+ ``.json``) report section QC2.1.

coreg    QC2.2: residual WAC-DEM shift in R.  (a) The windows of the dataset
         diagnostics (``diagnostics/B4/coregistration.csv``, WAC_GLOBAL) inside
         R, as delivered and with D15; (b) the same method
         (``diag_coregistration.measure``: image predicted from the terrain,
         phase correlation) re-run on the two rasters of the stage (WAC mosaic
         of QC1.1, DEM on the WAC grid of QC2.1), on 256 px windows centred every
         ``coreg_step_deg`` in R.  Pass if |median shift| < 0.5 px on both axes.
         -> report section QC2.2.

mask     QC3.1: validity mask = valid WAC AND valid DEM (``valid_mask``: both
         layers finite).  Light step: the mask is not written as a raster, it is
         recomputed from the NaN of the layers when a tile is read.  Saved: the
         number of invalid pixels of every cell (16 x 16 px) and the valid fraction
         of every tile (QC7.1 grid, from the cell counts, tile and stride being
         whole numbers of cells); a tile is used only if its valid fraction is
         >= ``masks.min_tile_valid_fraction`` (0.95).  Inside a used tile the
         invalid pixels are filled with the tile median of the layer and flagged
         (``fill_invalid``, applied at tile reading, QC7.3); a cell with an
         invalid pixel is flagged (``cell_invalid_px`` > 0).
         -> ``<output>/tile_validity.npz`` report section QC3.1.

relief   QC4.1 (QD3): local relief of the DEM on the WAC grid of R = height (QC2.1)
         minus its Gaussian-smoothed version (``derivatives.relief_sigma_px`` = 32 px
         = 3.2 km, isotropic in pixels, truncated at ``derivatives.truncate`` sigma),
         so that the km-scale topography does not dominate the dynamic range.
         Computed by row strips with a halo of one kernel radius (``smooth_rows``,
         identical to filtering the whole of R); at the border of R the DEM is
         reflected (scipy ``mode="reflect"``), the band within one kernel radius of
         the border is reported.  float16 m (QC7.3).
         -> ``<stage>/<layers.relief>`` (+ ``.json``) report section QC4.1.

slope    QC4.2 (QD3): slope (degrees) of the DEM on the WAC grid of R from central
         differences with the real pixel size (``dem_gradient``: N-S spacing
         = WAC pixel, E-W spacing scaled by cos lat of each row; one-sided
         differences on the first/last row and column of R).  Computed by row
         strips with a halo of one row (identical to the whole raster).  float16
         degrees (QC7.3).
         -> ``<stage>/<layers.slope>`` (+ ``.json``) report section QC4.2.

hillshade QC4.3 (QD3): Lambertian hillshade of the DEM on the WAC grid of R lit from
         ``derivatives.hillshade_azimuth_deg`` (0 = north, orthogonal to the WAC Sun
         from the west, D14) at ``hillshade_altitude_deg`` (30), shadows clipped to 0;
         gradient from ``dem_gradient`` (QC4.2).  A mean over equally spaced azimuths
         is not used: it reduces to sin(alt) cos(slope), a copy of the slope channel
         (checked here).  float16 in [0, 1] (QC7.3).
         -> ``<stage>/<layers.hillshade>`` (+ ``.json``) report section QC4.3.

norm     QC5.1: normalisation parameters, a light step applied at reading (no
         normalised raster is written).  Per layer (wac, relief, slope, hillshade)
         robust min-max: low / high = ``normalisation.percentiles`` (1st / 99th) of
         the pixels of the S1-train tiles (kept by the buffer and valid, QB3.2 +
         QC3.1: the pixels the backbone sees in training, all inside S1-train
         blocks), every ``normalisation.sample_step``-th row and column;
         ``to_unit`` = clip((x - low) / (high - low), 0, 1), then ``backbone_input``
         stacks 3 layers (``normalisation.backbone_inputs``) and applies the
         mean / std of the pre-trained DINOv2 config (timm) per channel.  Inverses:
         ``from_unit``, ``backbone_to_unit`` (exact except where clipped).
         -> ``<stage>/normalisation.json`` report section QC5.1.

Usage
-----
    python Experiments/qcai2027/qc_preprocess.py wac [--force | --checks-only]
    python Experiments/qcai2027/qc_preprocess.py dem
    python Experiments/qcai2027/qc_preprocess.py dem-grid [--force | --checks-only]
    python Experiments/qcai2027/qc_preprocess.py coreg
    python Experiments/qcai2027/qc_preprocess.py mask
    python Experiments/qcai2027/qc_preprocess.py relief [--force | --checks-only]
    python Experiments/qcai2027/qc_preprocess.py slope [--force | --checks-only]
    python Experiments/qcai2027/qc_preprocess.py hillshade [--force | --checks-only]
    python Experiments/qcai2027/qc_preprocess.py norm [--force | --checks-only]
    python Experiments/qcai2027/qc_preprocess.py --selftest   # synthetic checks of QC3.1, QC4.1-QC4.3 and QC5.1, no data needed
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_data import (CONFIG_PATH, MOON_RADIUS_M, SETS, CylGrid, RegionSource,  # noqa: E402
                     _import_dataset_modules, load_config, read_inventory, read_region, region_layout, save_report, written)
from qc_tiles import tile_grid_shape, tile_window  # noqa: E402

STRIP_ROWS = 1024


def stage_dir(cfg: dict) -> Path:
    """Folder of the stage of the experiment in the cleaned dataset."""
    _import_dataset_modules(cfg)
    from data_inventory import clean_dataset_dir  # noqa: E402
    return clean_dataset_dir(cfg["paths"]["dataset_root"], cfg["preprocess"]["stage"])


def raster_path(cfg: dict, layer: str) -> Path:
    return stage_dir(cfg) / cfg["preprocess"]["layers"][layer]


def write_raster(out: Path, shape: tuple[int, int], read_rows: Callable[[int, int], np.ndarray],
                 dtype=np.float32, strip_rows: int = STRIP_ROWS, force: bool = False) -> np.ndarray:
    """Raster of R written strip by strip (``read_rows(r0, r1)`` gives rows [r0, r1)) into a memory-mapped ``.npy``."""
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists: a stage is not rewritten (use --force)")
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(out, mode="w+", dtype=dtype, shape=shape)
    written(out)
    for r0 in range(0, shape[0], strip_rows):
        r1 = min(shape[0], r0 + strip_rows)
        arr[r0:r1] = read_rows(r0, r1)
    arr.flush()
    return arr


def open_or_write(out: Path, shape: tuple[int, int], read_rows, force: bool, checks_only: bool, **kw) -> np.ndarray:
    """The existing raster (``checks_only``) or a newly written one."""
    if checks_only and out.exists() and not force:
        return np.load(out, mmap_mode="r")
    return write_raster(out, shape, read_rows, force=force, **kw)


def grid_meta(g: CylGrid) -> dict:
    return {"grid": asdict(g), "grid_note": "equirectangular WAC grid of R; pixel (r, c) covers lat "
                                            "[north-(r+1)/ppd, north-r/ppd], lon [west+c/ppd, west+(c+1)/ppd]"}


def raster_stats(arr: np.ndarray, strip_rows: int = STRIP_ROWS) -> dict:
    """No-data per row/column, count of negative values, value statistics (percentiles on every 4th pixel)."""
    nan_rows = np.zeros(arr.shape[0], dtype=np.int64)
    nan_cols = np.zeros(arr.shape[1], dtype=np.int64)
    negative, sample = 0, []
    for r0 in range(0, arr.shape[0], strip_rows):
        block = np.asarray(arr[r0:r0 + strip_rows], dtype=np.float32)
        bad = np.isnan(block)
        nan_rows[r0:r0 + block.shape[0]] = bad.sum(1)
        nan_cols += bad.sum(0)
        negative += int((block < 0).sum())
        sample.append(block[::4, ::4][~bad[::4, ::4]])
    sample = np.concatenate(sample)
    pct = [0.1, 1, 50, 99, 99.9]
    return {
        "pixels": int(arr.size), "nodata_pixels": int(nan_rows.sum()),
        "valid_fraction": float(1 - nan_rows.sum() / arr.size),
        "rows_fully_nodata": np.flatnonzero(nan_rows == arr.shape[1]).tolist(),
        "cols_fully_nodata": np.flatnonzero(nan_cols == arr.shape[0]).tolist(),
        "nodata_outside_full_cols": int(nan_rows.sum() - arr.shape[0] * (nan_cols == arr.shape[0]).sum()),
        "negative_pixels": negative, "negative_fraction": negative / arr.size,
        "min": float(sample.min()), "max": float(sample.max()),
        "percentiles": {str(p): float(v) for p, v in zip(pct, np.percentile(sample, pct))},
    }


def run_wac(cfg: dict, force: bool = False, checks_only: bool = False) -> dict:
    """QC1.1: WAC mosaic of R -> ``<stage>/<layers.wac>``."""
    src = region_layout(cfg, "wac")
    out = raster_path(cfg, "wac")
    g = src.grid
    arr = open_or_write(out, (g.lines, g.samples), lambda r0, r1: read_region(src, slice(r0, r1), slice(0, g.samples)),
                        force, checks_only)
    stats = raster_stats(arr)

    # checks: (1) random windows equal to a direct read_region (strip borders included);
    # (2) the equator rows (seam of the two WAC tiles) are fully valid
    rng = np.random.default_rng(int(cfg["seed"]))
    windows = [(STRIP_ROWS - 3, 1000)] + [(int(rng.integers(0, g.lines - 64)), int(rng.integers(0, g.samples - 64)))
                                         for _ in range(5)]
    same = all(np.array_equal(np.asarray(arr[r:r + 64, c:c + 64]),
                              read_region(src, slice(r, r + 64), slice(c, c + 64)).astype(np.float32), equal_nan=True)
               for r, c in windows)
    seam = next(p.dst_rows.start for p in src.pieces if p.dst_rows.start > 0)
    seam_rows_valid = float(np.isfinite(np.asarray(arr[seam - 1:seam + 1])).mean())

    shift = {src.rasters[p.product_id][0].shift_rc for p in src.pieces}.pop()
    meta = {
        "layer": "wac", "step": "QC1.1", "dtype": "float32", "nodata": "NaN",
        "values": "physical values of the WAC global mosaic as returned by data_quality.read_physical",
        **grid_meta(g),
        "sources": [{"product_id": p.product_id, "src_rows": [p.src_rows.start, p.src_rows.stop],
                     "src_cols": [p.src_cols.start, p.src_cols.stop], "dst_rows": [p.dst_rows.start, p.dst_rows.stop],
                     "dst_cols": [p.dst_cols.start, p.dst_cols.stop]} for p in src.pieces],
        "d15_shift_rc_applied_to_mosaic": list(shift), "stats": stats,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    written(out.with_suffix(".json"))
    report = {"output": str(out), "shape": list(arr.shape), "size_gb": out.stat().st_size / 1e9, **stats,
              "check_windows_equal_read_region": same, "check_windows": windows,
              "equator_seam_rows": [seam - 1, seam], "equator_seam_rows_valid_fraction": seam_rows_valid}
    save_report(cfg, "QC1.1", report)
    return report


def dem_unit(cfg: dict) -> str:
    """Unit of the DEM values in the inventory (the same for all the DEM products of R)."""
    units = {read_inventory(cfg)[pid]["unit"] for pid in cfg["products"]["dem"]}
    if len(units) != 1:
        raise ValueError(f"DEM products with different units {units}")
    return units.pop()


def read_dem_height(src: RegionSource, rows: slice, cols: slice, unit: str) -> np.ndarray:
    """QC1.2: window of the DEM mosaic of R as height (m) above the reference sphere, NaN = no-data."""
    from diag_dem_sources import height_m  # noqa: E402
    return height_m(read_region(src, rows, cols), unit)


def run_dem(cfg: dict, step: int = 8, strip_rows: int = 4096) -> dict:
    """QC1.2 checks: unit and offset of the products, conversion against the stored values,
    validity and statistics of the height of R (every ``step``-th row and column)."""
    _import_dataset_modules(cfg)
    from diag_dem_sources import REF_RADIUS_M  # noqa: E402

    src = region_layout(cfg, "dem")
    unit = dem_unit(cfg)
    g = src.grid
    products = {pid: {"scale": r.scale, "offset": r.offset, "shift_rc": list(r.shift_rc), "dtype": str(r.data.dtype)}
                for pid, (r, _) in src.rasters.items()}

    # (1) conversion: height = (stored * scale + offset) [km] * 1000 - R_ref, on random windows of every product
    rng = np.random.default_rng(int(cfg["seed"]))
    conv_err = 0.0
    for p in src.pieces:
        raster = src.rasters[p.product_id][0]
        for _ in range(3):
            r = int(rng.integers(p.dst_rows.start, p.dst_rows.stop - 64))
            c = int(rng.integers(p.dst_cols.start, p.dst_cols.stop - 64))
            sr, sc = r - p.dst_rows.start + p.src_rows.start, c - p.dst_cols.start + p.src_cols.start
            stored = np.asarray(raster.data[sr:sr + 64, sc:sc + 64], dtype=np.float64)
            expect = (stored * raster.scale + raster.offset) * 1000.0 - REF_RADIUS_M
            conv_err = max(conv_err, float(np.nanmax(np.abs(read_dem_height(src, slice(r, r + 64), slice(c, c + 64), unit) - expect))))
    # (2) the corner where the 4 tiles meet: a window across it is complete
    rc = next(p.dst_rows.start for p in src.pieces if p.dst_rows.start > 0)
    cc = next(p.dst_cols.start for p in src.pieces if p.dst_cols.start > 0)
    corner = read_dem_height(src, slice(rc - 32, rc + 32), slice(cc - 32, cc + 32), unit)

    # (3) validity and statistics on a subsample of R
    nan, total, sample = 0, 0, []
    for r0 in range(0, g.lines, strip_rows):
        h = read_dem_height(src, slice(r0, min(g.lines, r0 + strip_rows), step), slice(0, g.samples, step), unit)
        bad = np.isnan(h)
        nan += int(bad.sum())
        total += h.size
        sample.append(h[~bad])
    sample = np.concatenate(sample)
    pct = [0.1, 1, 50, 99, 99.9]
    report = {
        "unit_in_inventory": unit, "reference_radius_m": REF_RADIUS_M, "products": products,
        "grid": asdict(g), "subsample_step": step, "subsample_pixels": total,
        "subsample_nodata": nan, "subsample_valid_fraction": 1 - nan / total,
        "height_m": {"min": float(sample.min()), "max": float(sample.max()), "mean": float(sample.mean()),
                     "percentiles": {str(q): float(v) for q, v in zip(pct, np.percentile(sample, pct))}},
        "check_conversion_max_abs_error_m": conv_err,
        "check_corner_window": {"rows": [rc - 32, rc + 32], "cols": [cc - 32, cc + 32],
                                "valid_fraction": float(np.isfinite(corner).mean())},
        "seams": "continuity of the DEM seams checked in QB1.1 (report section QB1.1, modality dem)",
    }
    save_report(cfg, "QC1.2", report)
    return report


def pixel_m(g: CylGrid) -> float:
    """N-S pixel size (m) of an equirectangular grid on the reference sphere."""
    return 2 * np.pi * MOON_RADIUS_M / 360.0 / g.ppd


def dem_on_grid(dem: RegionSource, unit: str, target: CylGrid, rows: slice, k: int) -> np.ndarray:
    """QC2.1: DEM height (m) at the centres of the pixels ``rows`` x all columns of ``target``:
    mean of k x k DEM pixels (aligned with the DEM grid), then bilinear interpolation
    (``scipy.ndimage.map_coordinates``, order 1).  The at most ~0.1 block beyond the outer
    block centres at the border of R takes the nearest value."""
    from diag_lamp_snr import block_mean  # noqa: E402
    from scipy.ndimage import map_coordinates

    g = dem.grid
    lat, _ = target.rc_to_latlon(np.arange(target.lines)[rows], 0)
    _, lon = target.rc_to_latlon(0, np.arange(target.samples))
    rf, cf = g.latlon_to_rc(lat, g.west)[0], g.latlon_to_rc(g.north, lon)[1]   # fractional DEM row / col
    r0 = max(0, int(np.floor(rf.min())) - 2 * k)
    r0 -= r0 % k
    r1 = min(g.lines, int(np.ceil(rf.max())) + 2 * k + 1)
    win = block_mean(read_dem_height(dem, slice(r0, r1), slice(0, g.samples), unit), k)
    br, bc = (rf - r0 + 0.5) / k - 0.5, (cf + 0.5) / k - 0.5        # coordinates in the block grid
    rr, cc = np.meshgrid(br, bc, indexing="ij")
    return map_coordinates(win, [rr, cc], order=1, mode="nearest", prefilter=False)


def run_dem_grid(cfg: dict, force: bool = False, checks_only: bool = False, step: int = 16) -> dict:
    """QC2.1: DEM on the WAC grid of R -> ``<stage>/<layers.dem>``, with checks."""
    _import_dataset_modules(cfg)
    dem, unit = region_layout(cfg, "dem"), dem_unit(cfg)
    target = region_layout(cfg, "wac").grid
    k = max(1, int(round(pixel_m(target) / pixel_m(dem.grid))))
    out = raster_path(cfg, "dem")
    shape = (target.lines, target.samples)
    arr = open_or_write(out, shape, lambda r0, r1: dem_on_grid(dem, unit, target, slice(r0, r1), k),
                        force, checks_only, strip_rows=512)
    stats = raster_stats(arr)

    # (1) independent of the strips: windows recomputed alone (one across a strip border)
    rng = np.random.default_rng(int(cfg["seed"]))
    starts = [512 - 3] + [int(rng.integers(0, target.lines - 32)) for _ in range(3)]
    same = all(np.array_equal(np.asarray(arr[r:r + 32]), dem_on_grid(dem, unit, target, slice(r, r + 32), k).astype(np.float32),
                              equal_nan=True) for r in starts)
    # (2) against the native DEM (nearest DEM pixel) at every step-th WAC pixel: the block mean
    #     smooths (spread of the difference) but must not shift or bias the heights
    rows = np.arange(0, target.lines, step)
    cols = np.arange(0, target.samples, step)
    lat, _ = target.rc_to_latlon(rows, 0)
    _, lon = target.rc_to_latlon(0, cols)
    dr = np.clip(np.round(dem.grid.latlon_to_rc(lat, 0)[0]).astype(int), 0, dem.grid.lines - 1)
    dc = np.clip(np.round(dem.grid.latlon_to_rc(0, lon)[1]).astype(int), 0, dem.grid.samples - 1)
    diff = []
    for i, r in enumerate(rows):
        native = read_dem_height(dem, slice(dr[i], dr[i] + 1), slice(0, dem.grid.samples), unit)[0, dc]
        diff.append(np.asarray(arr[r, cols], dtype=np.float64) - native)
    diff = np.concatenate(diff)
    diff = diff[np.isfinite(diff)]
    pct = [1, 5, 50, 95, 99]
    # (3) pixels whose block coordinate is beyond the outer block centres (nearest value used)
    nb_r, nb_c = (dem.grid.lines // k), (dem.grid.samples // k)
    lat_all, _ = target.rc_to_latlon(np.arange(target.lines), 0)
    _, lon_all = target.rc_to_latlon(0, np.arange(target.samples))
    br = (dem.grid.latlon_to_rc(lat_all, 0)[0] + 0.5) / k - 0.5
    bc = (dem.grid.latlon_to_rc(0, lon_all)[1] + 0.5) / k - 0.5
    out_r, out_c = (br < 0) | (br > nb_r - 1), (bc < 0) | (bc > nb_c - 1)

    meta = {
        "layer": "dem", "step": "QC2.1", "dtype": "float32", "nodata": "NaN", "unit": "m",
        "values": "SLDEM2015 height above the 1,737.4 km sphere, mean of k x k DEM pixels, bilinear at the WAC pixel centres",
        **grid_meta(target), "block_k": k, "effective_pixel_m": k * pixel_m(dem.grid),
        "sources": list(dem.rasters), "stats": stats,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    written(out.with_suffix(".json"))
    report = {
        "output": str(out), "shape": list(shape), "size_gb": out.stat().st_size / 1e9,
        "wac_pixel_m": pixel_m(target), "dem_pixel_m": pixel_m(dem.grid), "block_k": k,
        "effective_pixel_m": k * pixel_m(dem.grid), **stats,
        "check_windows_equal_direct": same, "check_window_rows": starts,
        "check_vs_native_nearest": {"step": step, "n": int(diff.size), "mean_m": float(diff.mean()),
                                    "std_m": float(diff.std()), "mean_abs_m": float(np.abs(diff).mean()),
                                    "percentiles_m": {str(q): float(v) for q, v in zip(pct, np.percentile(diff, pct))}},
        "border_nearest": {"rows": np.flatnonzero(out_r).tolist(), "cols": np.flatnonzero(out_c).tolist(),
                           "max_block_distance": float(max(np.maximum(-br, br - (nb_r - 1)).max(),
                                                           np.maximum(-bc, bc - (nb_c - 1)).max()))},
    }
    save_report(cfg, "QC2.1", report)
    return report


def shift_summary(e_px: np.ndarray, n_px: np.ndarray, valid: np.ndarray, fit_r: np.ndarray) -> dict:
    """Median and robust sigma (1.4826 MAD) of the valid shifts, in WAC pixels."""
    mad = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else None
    e, n = e_px[valid], n_px[valid]
    return {"windows": int(valid.size), "valid": int(valid.sum()), "median_fit_r": float(np.median(fit_r)),
            "median_shift_e_px": float(np.median(e)) if e.size else None,
            "median_shift_n_px": float(np.median(n)) if n.size else None,
            "robust_sigma_e_px": mad(e), "robust_sigma_n_px": mad(n)}


def run_coreg(cfg: dict, step_deg: float = 3.0, win: int = 256, tol_px: float = 0.5) -> dict:
    """QC2.2: residual shift of the WAC mosaic relative to the DEM inside R."""
    import math

    import pandas as pd

    _import_dataset_modules(cfg)
    from diag_coregistration import measure  # noqa: E402

    reg = cfg["region"]
    # (a) windows of the dataset diagnostics inside R
    diag_csv = cfg["paths"]["source_root"] / "preprocessing/diagnostics/B4/coregistration.csv"
    d = pd.read_csv(diag_csv)
    d = d[(d["target"] == "WAC_GLOBAL") & d["product_id"].isin(cfg["products"]["wac"])
          & d["lat"].between(reg["lat_min"], reg["lat_max"]) & d["lon"].between(reg["lon_min"], reg["lon_max"])]
    diag = {}
    for reading, g in d.groupby("reading"):
        px = g["pixel_m"].to_numpy()
        lat_cos = np.cos(np.radians(g["lat"].to_numpy()))
        diag[reading] = shift_summary(g["shift_e_m"].to_numpy() / (px * lat_cos), g["shift_n_m"].to_numpy() / px,
                                      g["valid"].to_numpy(bool), g["fit_r"].to_numpy())
        diag[reading]["per_product"] = {pid: shift_summary(gg["shift_e_px"].to_numpy(), gg["shift_n_px"].to_numpy(),
                                                           gg["valid"].to_numpy(bool), gg["fit_r"].to_numpy())
                                        for pid, gg in g.groupby("product_id")}

    # (b) re-measured on the rasters of the stage
    grid = region_layout(cfg, "wac").grid
    wac = np.load(raster_path(cfg, "wac"), mmap_mode="r")
    dem = np.load(raster_path(cfg, "dem"), mmap_mode="r")
    px_m = pixel_m(grid)
    rows = []
    for lat in np.arange(reg["lat_min"] + step_deg / 2, reg["lat_max"], step_deg):
        for lon in np.arange(reg["lon_min"] + step_deg / 2, reg["lon_max"], step_deg):
            r, c = (int(round(x)) for x in grid.latlon_to_rc(lat, lon))
            r0, c0 = r - win // 2, c - win // 2
            if r0 < 0 or c0 < 0 or r0 + win > grid.lines or c0 + win > grid.samples:
                continue
            img = np.asarray(wac[r0:r0 + win, c0:c0 + win], dtype=np.float64)
            z = np.asarray(dem[r0:r0 + win, c0:c0 + win], dtype=np.float64)
            m = measure(img, z, px_m * math.cos(math.radians(lat)), px_m, 0.0)
            if m is not None:
                rows.append({"lat": float(lat), "lon": float(lon), "row0": r0, "col0": c0, **m})
    w = pd.DataFrame(rows)
    stage = shift_summary(w["shift_e_px"].to_numpy(), w["shift_n_px"].to_numpy(), w["valid"].to_numpy(bool),
                          w["fit_r"].to_numpy())
    hemi = {name: shift_summary(g["shift_e_px"].to_numpy(), g["shift_n_px"].to_numpy(), g["valid"].to_numpy(bool),
                                g["fit_r"].to_numpy()) for name, g in w.groupby(np.where(w["lat"] > 0, "north", "south"))}
    passed = all(abs(stage[k]) < tol_px for k in ("median_shift_e_px", "median_shift_n_px"))
    report = {
        "tolerance_px": tol_px, "diagnostics_csv": str(diag_csv), "diagnostics_in_R": diag,
        "stage_rasters": {"step_deg": step_deg, "window_px": win, **stage, "per_hemisphere": hemi},
        "pass": bool(passed),
    }
    save_report(cfg, "QC2.2", report)
    return report


# ---------------------------------------------------------------------------
# QC3.1 - Validity mask
# ---------------------------------------------------------------------------


def valid_mask(*layers: np.ndarray) -> np.ndarray:
    """QC3.1: True where every layer is valid (finite), i.e. valid WAC AND valid DEM."""
    return np.logical_and.reduce([np.isfinite(np.asarray(x)) for x in layers])


def fill_invalid(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """QC3.1: float32 copy of a tile layer with the invalid pixels set to the median of its
    valid pixels.  The caller keeps ``valid`` as the flag of the filled pixels (never data)."""
    out = np.array(x, dtype=np.float32, copy=True)
    if not valid.all():
        if not valid.any():
            raise ValueError("tile without valid pixels")
        out[~valid] = np.median(out[valid])
    return out


def cell_counts(bad: np.ndarray, cell_px: int, n_cell_cols: int) -> np.ndarray:
    """Number of True pixels of ``bad`` (rows a multiple of ``cell_px``) in each of the
    first ``n_cell_cols`` cells of every cell row."""
    h = bad.shape[0] // cell_px
    b = bad[:h * cell_px, :n_cell_cols * cell_px].reshape(h, cell_px, n_cell_cols, cell_px)
    return b.sum(axis=(1, 3))


def tile_sums(per_cell: np.ndarray, tiles_shape: tuple[int, int], tile_cells: int, stride_cells: int) -> np.ndarray:
    """Sum of ``per_cell`` over the tile_cells x tile_cells cells of every tile (origins at
    multiples of ``stride_cells``), with a 2-D cumulative sum."""
    c = np.zeros((per_cell.shape[0] + 1, per_cell.shape[1] + 1), dtype=np.int64)
    c[1:, 1:] = per_cell.cumsum(0).cumsum(1)
    r0 = np.arange(tiles_shape[0]) * stride_cells
    c0 = np.arange(tiles_shape[1]) * stride_cells
    r1, c1 = r0 + tile_cells, c0 + tile_cells
    return c[r1][:, c1] - c[r0][:, c1] - c[r1][:, c0] + c[r0][:, c0]


def run_mask(cfg: dict, strip_rows: int = STRIP_ROWS) -> dict:
    """QC3.1: invalid pixels per layer, per cell and per tile; tiles used (valid fraction >= threshold)."""
    import pandas as pd

    tile_px, stride_px, cell_px = (int(cfg["tiling"][k]) for k in ("tile_px", "stride_px", "cell_px"))
    thr = float(cfg["masks"]["min_tile_valid_fraction"])
    layers = {name: np.load(raster_path(cfg, name), mmap_mode="r") for name in ("wac", "dem")}
    lines, samples = layers["wac"].shape
    if layers["dem"].shape != (lines, samples) or strip_rows % cell_px:
        raise ValueError("layers on different grids or strip not a whole number of cells")
    cells_shape = (lines // cell_px, samples // cell_px)
    tshape = tile_grid_shape(lines, samples, tile_px, stride_px)

    # one pass over R: invalid pixels per layer, per row/column and per cell
    invalid_layer = {name: 0 for name in layers}
    bad_rows = np.zeros(lines, dtype=np.int64)
    bad_cols = np.zeros(samples, dtype=np.int64)
    cell_bad = np.zeros(cells_shape, dtype=np.int64)
    for r0 in range(0, lines, strip_rows):
        blocks = {name: np.asarray(a[r0:r0 + strip_rows]) for name, a in layers.items()}
        for name, b in blocks.items():
            invalid_layer[name] += int((~np.isfinite(b)).sum())
        bad = ~valid_mask(*blocks.values())
        bad_rows[r0:r0 + bad.shape[0]] = bad.sum(1)
        bad_cols += bad.sum(0)
        i0 = r0 // cell_px
        counts = cell_counts(bad, cell_px, cells_shape[1])
        cell_bad[i0:i0 + counts.shape[0]] = counts
    tile_bad = tile_sums(cell_bad, tshape, tile_px // cell_px, stride_px // cell_px)
    valid_fraction = 1.0 - tile_bad / tile_px ** 2
    tile_used = valid_fraction >= thr

    # checks: (1) tile fractions recomputed directly from the rasters (random tiles and the tiles
    # at the corners of the tile grid, the east column being the closest to the no-data column);
    # (2) fill_invalid on the real window of 256 px at the north-east corner of R (it contains
    # the no-data column of the WAC, outside every tile): no invalid pixel left, filled with the median
    rng = np.random.default_rng(int(cfg["seed"]))
    probe = [(0, 0), (0, tshape[1] - 1), (tshape[0] - 1, 0), (tshape[0] - 1, tshape[1] - 1)] + \
            [(int(rng.integers(tshape[0])), int(rng.integers(tshape[1]))) for _ in range(6)]
    direct_ok = True
    for ti, tj in probe:
        win = tile_window(ti, tj, tile_px, stride_px)
        direct_ok &= bool(np.isclose(valid_mask(*(a[win] for a in layers.values())).mean(), valid_fraction[ti, tj]))
    win = (slice(0, tile_px), slice(samples - tile_px, samples))
    v = valid_mask(*(a[win] for a in layers.values()))
    filled = {name: fill_invalid(a[win], v) for name, a in layers.items()}
    fill_check = {"rows": [0, tile_px], "cols": [samples - tile_px, samples], "invalid_pixels": int((~v).sum()),
                  "nan_left": int(sum(np.isnan(f).sum() for f in filled.values())),
                  "filled_equal_median": bool(all(np.allclose(f[~v], np.median(f[v])) for f in filled.values()))}

    # effect on the split of QB3.2: tiles kept by the buffer that the mask drops, flagged cells used
    tiles = pd.read_csv(cfg["paths"]["output"] / "tiles.csv")
    if len(tiles) != tile_used.size:
        raise ValueError("tile grid differs from tiles.csv (QB3.2)")
    split_cells = np.load(cfg["paths"]["output"] / "split_cells.npz")
    cell_set, owner = split_cells["cell_set"], split_cells["owner_tile"]
    if cell_set.shape != cells_shape:
        raise ValueError("cell grid differs from split_cells.npz (QB3.2)")
    used_flat = tile_used.ravel()
    cell_ok = (owner >= 0) & used_flat[np.where(owner >= 0, owner, 0)]
    per_set = {}
    for k, name in enumerate(SETS):
        t = tiles["set"].to_numpy() == name
        kept = t & tiles["kept"].to_numpy(bool)
        m = cell_set == k
        per_set[name] = {"tiles_kept_by_buffer": int(kept.sum()), "tiles_kept_and_valid": int((kept & used_flat).sum()),
                         "cells_used_by_buffer": int(m.sum()), "cells_used_after_mask": int((m & cell_ok).sum()),
                         "cells_used_flagged": int((m & cell_ok & (cell_bad > 0)).sum())}

    out = cfg["paths"]["output"] / "tile_validity.npz"
    np.savez_compressed(out, valid_fraction=valid_fraction.astype(np.float32), tile_used=tile_used,
                        cell_invalid_px=cell_bad.astype(np.int16), min_tile_valid_fraction=thr)
    written(out)
    n_bad = int(bad_rows.sum())
    full_cols = np.flatnonzero(bad_cols == lines)
    report = {
        "output": str(out), "min_tile_valid_fraction": thr, "grid_shape": [lines, samples],
        "invalid_pixels_per_layer": invalid_layer, "invalid_pixels_mask": n_bad, "invalid_fraction": n_bad / (lines * samples),
        "cols_fully_invalid": full_cols.tolist(), "rows_fully_invalid": np.flatnonzero(bad_rows == samples).tolist(),
        "invalid_outside_full_cols": int(n_bad - lines * full_cols.size),
        "invalid_pixels_in_full_cells": int(cell_bad.sum()),
        "invalid_pixels_in_tiles": int(cell_bad[:(tshape[0] - 1) * stride_px // cell_px + tile_px // cell_px,
                                                :(tshape[1] - 1) * stride_px // cell_px + tile_px // cell_px].sum()),
        "cells_shape": list(cells_shape), "cells_flagged": int((cell_bad > 0).sum()),
        "tiles_shape": list(tshape), "tiles": int(tile_used.size), "tiles_with_invalid_pixels": int((tile_bad > 0).sum()),
        "tiles_used": int(tile_used.sum()), "tiles_not_used": int((~tile_used).sum()),
        "tile_valid_fraction_min": float(valid_fraction.min()),
        "per_set": per_set,
        "check_tile_fraction_equal_direct": bool(direct_ok), "check_tiles": len(probe), "check_fill_invalid": fill_check,
    }
    save_report(cfg, "QC3.1", report)
    return report


# ---------------------------------------------------------------------------
# QC4 - DEM derivatives
# ---------------------------------------------------------------------------


def smooth_rows(z: np.ndarray, rows: slice, sigma: float, truncate: float) -> np.ndarray:
    """Rows ``rows`` of the Gaussian-smoothed raster ``z`` (``scipy.ndimage.gaussian_filter``,
    ``mode="reflect"`` at the border of ``z``), reading only the rows within one kernel radius:
    the missing rows beyond the first/last row of ``z`` are reflected as scipy does
    (numpy ``symmetric``), so the result equals filtering the whole raster."""
    from scipy.ndimage import gaussian_filter

    radius = int(truncate * sigma + 0.5)
    r0, r1 = rows.start, rows.stop
    a0, a1 = max(0, r0 - radius), min(z.shape[0], r1 + radius)
    win = np.asarray(z[a0:a1], dtype=np.float64)
    if np.isnan(win).any():
        raise ValueError("no-data in the DEM: the Gaussian smoothing is not mask-aware")
    pad = (radius - (r0 - a0), radius - (a1 - r1))
    if any(pad):
        win = np.pad(win, (pad, (0, 0)), mode="symmetric")
    sm = gaussian_filter(win, sigma, mode="reflect", truncate=truncate)
    return sm[radius:radius + (r1 - r0)]


def local_relief(z: np.ndarray, rows: slice, sigma: float, truncate: float) -> np.ndarray:
    """QC4.1: height minus its Gaussian-smoothed version, rows ``rows`` of ``z``."""
    return np.asarray(z[rows], dtype=np.float64) - smooth_rows(z, rows, sigma, truncate)


def run_relief(cfg: dict, force: bool = False, checks_only: bool = False, step: int = 4) -> dict:
    """QC4.1: local relief of the DEM on the WAC grid of R -> ``<stage>/<layers.relief>``, with checks."""
    sigma, truncate = float(cfg["derivatives"]["relief_sigma_px"]), float(cfg["derivatives"]["truncate"])
    radius = int(truncate * sigma + 0.5)
    dem = np.load(raster_path(cfg, "dem"), mmap_mode="r")
    out = raster_path(cfg, "relief")
    arr = open_or_write(out, dem.shape, lambda r0, r1: local_relief(dem, slice(r0, r1), sigma, truncate),
                        force, checks_only, dtype=np.float16)
    stats = raster_stats(arr)
    lines, samples = dem.shape

    # (1) independent of the strips: windows recomputed alone, in float64 (one across a strip
    #     border, the first and last rows of R where the reflection is used, random ones);
    #     the difference is the float16 rounding only
    rng = np.random.default_rng(int(cfg["seed"]))
    starts = [STRIP_ROWS - 3, 0, lines - 32] + [int(rng.integers(0, lines - 32)) for _ in range(3)]
    err, rel_err = 0.0, 0.0
    for r in starts:
        ref = local_relief(dem, slice(r, r + 32), sigma, truncate)
        d = np.abs(np.asarray(arr[r:r + 32], dtype=np.float64) - ref)
        err = max(err, float(d.max()))
        rel_err = max(rel_err, float((d / np.maximum(np.abs(ref), 1.0)).max()))
    # (2) the long wavelengths are removed: spread of the height vs of the relief, and the mean
    #     of the relief over 16 x 16 cells (should be ~0 against the height variations)
    h = np.asarray(dem[::step, ::step], dtype=np.float64)
    rl = np.asarray(arr[::step, ::step], dtype=np.float64)
    pct = [0.1, 1, 5, 50, 95, 99, 99.9]
    meta = {
        "layer": "relief", "step": "QC4.1", "dtype": "float16", "nodata": "NaN", "unit": "m",
        "values": "DEM height on the WAC grid of R (QC2.1) minus its Gaussian-smoothed version",
        **grid_meta(region_layout(cfg, "wac").grid), "sigma_px": sigma, "truncate": truncate, "kernel_radius_px": radius,
        "border": "scipy gaussian_filter mode='reflect' at the border of R", "source": str(raster_path(cfg, "dem")),
        "stats": stats,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    written(out.with_suffix(".json"))
    report = {
        "output": str(out), "shape": list(arr.shape), "size_gb": out.stat().st_size / 1e9,
        "sigma_px": sigma, "sigma_km_ns": sigma * pixel_m(region_layout(cfg, "wac").grid) / 1000.0,
        "truncate": truncate, "kernel_radius_px": radius, **stats,
        "subsample_step": step,
        "height_m_std": float(h.std()), "relief_m_std": float(rl.std()), "relief_m_mean": float(rl.mean()),
        "relief_m_percentiles": {str(q): float(v) for q, v in zip(pct, np.percentile(rl, pct))},
        "height_m_percentiles": {str(q): float(v) for q, v in zip(pct, np.percentile(h, pct))},
        "check_windows_rows": starts, "check_windows_max_abs_error_m": err,
        "check_windows_max_rel_error": rel_err,
        **border_effect(cfg, radius, lines, samples),
    }
    save_report(cfg, "QC4.1", report)
    return report


def dem_gradient(z: np.ndarray, rows: slice, grid: CylGrid) -> tuple[np.ndarray, np.ndarray]:
    """Gradient (m/m) of the height ``z`` on the equirectangular ``grid``, rows ``rows``:
    (dz/dx towards east, dz/dy towards north), central differences (``np.gradient``,
    one-sided on the first/last row and column of ``z``), N-S spacing = pixel size,
    E-W spacing = pixel size * cos(lat) of the row.  Reads one row of halo."""
    r0, r1 = rows.start, rows.stop
    a0, a1 = max(0, r0 - 1), min(z.shape[0], r1 + 1)
    win = np.asarray(z[a0:a1], dtype=np.float64)
    if np.isnan(win).any():
        raise ValueError("no-data in the DEM: the gradient is not mask-aware")
    d_row, d_col = np.gradient(win)                                   # per pixel, rows towards south
    d_row, d_col = d_row[r0 - a0:r0 - a0 + (r1 - r0)], d_col[r0 - a0:r0 - a0 + (r1 - r0)]
    lat, _ = grid.rc_to_latlon(np.arange(r0, r1), 0)
    px = pixel_m(grid)
    return d_col / (px * np.cos(np.radians(lat)))[:, None], -d_row / px


def slope_deg(z: np.ndarray, rows: slice, grid: CylGrid) -> np.ndarray:
    """QC4.2: slope in degrees, rows ``rows`` of ``z``."""
    gx, gy = dem_gradient(z, rows, grid)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def border_effect(cfg: dict, band_px: int, lines: int, samples: int) -> dict:
    """Pixels, used tiles (QB3.2 + QC3.1) and used cells within ``band_px`` of the border of R."""
    import pandas as pd

    tile_px, cell_px = int(cfg["tiling"]["tile_px"]), int(cfg["tiling"]["cell_px"])
    tiles = pd.read_csv(cfg["paths"]["output"] / "tiles.csv")
    r0t, c0t = tiles["row0"].to_numpy(), tiles["col0"].to_numpy()
    touch = (r0t < band_px) | (c0t < band_px) | (r0t + tile_px > lines - band_px) | (c0t + tile_px > samples - band_px)
    kept = tiles["kept"].to_numpy(bool) & np.load(cfg["paths"]["output"] / "tile_validity.npz")["tile_used"].ravel()
    cell_set = np.load(cfg["paths"]["output"] / "split_cells.npz")["cell_set"]
    ci = np.arange(cell_set.shape[0]) * cell_px
    cj = np.arange(cell_set.shape[1]) * cell_px
    near = ((ci < band_px) | (ci + cell_px > lines - band_px))[:, None] | \
           ((cj < band_px) | (cj + cell_px > samples - band_px))[None, :]
    band_pixels = lines * samples - max(0, lines - 2 * band_px) * max(0, samples - 2 * band_px)
    return {"border_band_px": band_px, "border_band_pixels": int(band_pixels),
            "border_band_fraction_of_R": band_pixels / (lines * samples),
            "tiles_touching_border_band": int(touch.sum()), "tiles_used_touching_border_band": int((touch & kept).sum()),
            "tiles_used": int(kept.sum()), "cells_used_in_border_band": int(((cell_set >= 0) & near).sum()),
            "cells_used": int((cell_set >= 0).sum())}


def run_slope(cfg: dict, force: bool = False, checks_only: bool = False, step: int = 4) -> dict:
    """QC4.2: slope of the DEM on the WAC grid of R -> ``<stage>/<layers.slope>``, with checks."""
    grid = region_layout(cfg, "wac").grid
    dem = np.load(raster_path(cfg, "dem"), mmap_mode="r")
    out = raster_path(cfg, "slope")
    arr = open_or_write(out, dem.shape, lambda r0, r1: slope_deg(dem, slice(r0, r1), grid),
                        force, checks_only, dtype=np.float16)
    stats = raster_stats(arr)
    lines, samples = dem.shape

    # (1) independent of the strips: windows recomputed alone in float64 (across a strip border,
    #     first and last rows of R, random); the difference is the float16 rounding only
    rng = np.random.default_rng(int(cfg["seed"]))
    starts = [STRIP_ROWS - 3, 0, lines - 32] + [int(rng.integers(0, lines - 32)) for _ in range(3)]
    err = max(float(np.abs(np.asarray(arr[r:r + 32], dtype=np.float64) - slope_deg(dem, slice(r, r + 32), grid)).max())
              for r in starts)
    # (2) effect of cos lat: E-W spacing at the centre and at the edges of R, and the slope that
    #     a spacing not scaled by cos lat would give (every step-th pixel of a row band at the edge)
    lat, _ = grid.rc_to_latlon(np.array([0, lines // 2, lines - 1]), 0)
    r_edge = slice(0, 64)
    gx, gy = dem_gradient(dem, r_edge, grid)
    cosl = np.cos(np.radians(grid.rc_to_latlon(np.arange(64), 0)[0]))[:, None]
    s_true = np.degrees(np.arctan(np.hypot(gx, gy)))
    s_nocos = np.degrees(np.arctan(np.hypot(gx * cosl, gy)))
    # (3) statistics on every step-th pixel
    sample = np.asarray(arr[::step, ::step], dtype=np.float64).ravel()
    pct = [1, 5, 25, 50, 75, 95, 99, 99.9]

    meta = {
        "layer": "slope", "step": "QC4.2", "dtype": "float16", "nodata": "NaN", "unit": "degree",
        "values": "slope of the DEM on the WAC grid of R (QC2.1), central differences, E-W spacing scaled by cos lat",
        **grid_meta(grid), "pixel_m_ns": pixel_m(grid), "border": "one-sided differences on the first/last row and column of R",
        "source": str(raster_path(cfg, "dem")), "stats": stats,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    written(out.with_suffix(".json"))
    report = {
        "output": str(out), "shape": list(arr.shape), "size_gb": out.stat().st_size / 1e9, **stats,
        "pixel_m_ns": pixel_m(grid),
        "pixel_m_ew": {f"{float(la):.4f}": pixel_m(grid) * float(np.cos(np.radians(la))) for la in lat},
        "subsample_step": step, "slope_deg_mean": float(sample.mean()),
        "slope_deg_percentiles": {str(q): float(v) for q, v in zip(pct, np.percentile(sample, pct))},
        "fraction_above_deg": {str(t): float((sample > t).mean()) for t in (5, 10, 20, 30)},
        "check_windows_rows": starts, "check_windows_max_abs_error_deg": err,
        "cos_lat_effect_first_64_rows": {"mean_slope_deg": float(s_true.mean()),
                                         "mean_slope_deg_without_cos_lat": float(s_nocos.mean())},
        **border_effect(cfg, 1, lines, samples),
    }
    save_report(cfg, "QC4.2", report)
    return report


def hillshade(gx: np.ndarray, gy: np.ndarray, azimuth_deg: float, altitude_deg: float, clip: bool = True) -> np.ndarray:
    """QC4.3: Lambertian shading (cosine of the angle between the surface normal and the Sun) of a
    surface with gradient (gx east, gy north), Sun at ``azimuth_deg`` (clockwise from north) and
    ``altitude_deg`` above the horizon; shadows (negative values) clipped to 0 if ``clip``."""
    az, alt = np.radians(azimuth_deg), np.radians(altitude_deg)
    sx, sy, sz = np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)
    hs = (sz - gx * sx - gy * sy) / np.sqrt(1.0 + gx * gx + gy * gy)
    return np.maximum(hs, 0.0) if clip else hs


def run_hillshade(cfg: dict, force: bool = False, checks_only: bool = False, n_windows: int = 24,
                  win_rows: int = 32) -> dict:
    """QC4.3: hillshade of the DEM on the WAC grid of R -> ``<stage>/<layers.hillshade>``, with checks."""
    az = float(cfg["derivatives"]["hillshade_azimuth_deg"])
    alt = float(cfg["derivatives"]["hillshade_altitude_deg"])
    grid = region_layout(cfg, "wac").grid
    dem = np.load(raster_path(cfg, "dem"), mmap_mode="r")
    out = raster_path(cfg, "hillshade")

    def rows(r0: int, r1: int) -> np.ndarray:
        return hillshade(*dem_gradient(dem, slice(r0, r1), grid), az, alt)

    arr = open_or_write(out, dem.shape, rows, force, checks_only, dtype=np.float16)
    stats = raster_stats(arr)
    lines, samples = dem.shape

    # on n_windows windows of win_rows full rows, evenly spaced in R (the first one across a
    # strip border): (1) equal to a direct computation in float64 (float16 rounding only);
    # (2) why one azimuth: the mean of 4 equally spaced azimuths vs sin(alt) cos(slope);
    # (3) Pearson correlation with the WAC of the hillshade lit from each of the 4 directions
    #     and of the 4-azimuth mean (the WAC shading follows its Sun, from the west, D14)
    wac = np.load(raster_path(cfg, "wac"), mmap_mode="r")
    starts = [STRIP_ROWS - 3] + np.linspace(0, lines - win_rows, n_windows - 1).astype(int).tolist()
    azimuths = (0.0, 90.0, 180.0, 270.0)
    err, n_px, n_shadow4, max_dev4 = 0.0, 0, 0, 0.0
    sums = {k: np.zeros(5) for k in list(azimuths) + ["mean4"]}       # n, sx, sy, sxx, syy -> corr via sxy below
    sxy = {k: 0.0 for k in sums}
    for r in starts:
        sl = slice(r, r + win_rows)
        gx, gy = dem_gradient(dem, sl, grid)
        hs = hillshade(gx, gy, az, alt)
        err = max(err, float(np.abs(np.asarray(arr[sl], dtype=np.float64) - hs).max()))
        lit = {a: hillshade(gx, gy, a, alt) for a in azimuths}
        lit["mean4"] = sum(lit[a] for a in azimuths) / len(azimuths)
        expect = np.sin(np.radians(alt)) / np.sqrt(1.0 + gx * gx + gy * gy)     # sin(alt) cos(slope)
        dev = np.abs(lit["mean4"] - expect)
        n_px += dev.size
        n_shadow4 += int((dev > 1e-9).sum())
        max_dev4 = max(max_dev4, float(dev.max()))
        w = np.asarray(wac[sl], dtype=np.float64)
        ok = np.isfinite(w)
        for k, v in lit.items():
            x, y = v[ok], w[ok]
            sums[k] += [x.size, x.sum(), y.sum(), (x * x).sum(), (y * y).sum()]
            sxy[k] += float((x * y).sum())

    def corr(k: str) -> float:
        n, sx, sy, sxx, syy = sums[k]
        return float((sxy[k] - sx * sy / n) / np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n)))

    sample = np.asarray(arr[::4, ::4], dtype=np.float64).ravel()
    pct = [0.1, 1, 5, 50, 95, 99, 99.9]
    meta = {
        "layer": "hillshade", "step": "QC4.3", "dtype": "float16", "nodata": "NaN", "unit": "cosine of the Sun-normal angle, [0, 1]",
        "values": "Lambertian hillshade of the DEM on the WAC grid of R (QC2.1), shadows = 0",
        **grid_meta(grid), "azimuth_deg": az, "altitude_deg": alt, "gradient": "dem_gradient (QC4.2)",
        "source": str(raster_path(cfg, "dem")), "stats": stats,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    written(out.with_suffix(".json"))
    report = {
        "output": str(out), "shape": list(arr.shape), "size_gb": out.stat().st_size / 1e9,
        "azimuth_deg": az, "altitude_deg": alt, **stats,
        "subsample_step": 4, "hillshade_mean": float(sample.mean()), "hillshade_std": float(sample.std()),
        "hillshade_percentiles": {str(q): float(v) for q, v in zip(pct, np.percentile(sample, pct))},
        "shadow_fraction": float((sample == 0).mean()),
        "check_windows_rows": starts, "check_window_rows_each": win_rows,
        "check_windows_max_abs_error": err,
        "mean_of_4_azimuths": {"azimuths_deg": list(azimuths), "pixels": n_px,
                               "fraction_not_equal_sin_alt_cos_slope": n_shadow4 / n_px,
                               "max_abs_difference": max_dev4},
        "correlation_with_wac": {("mean4" if k == "mean4" else f"az_{k:g}"): corr(k) for k in sums},
        **border_effect(cfg, 1, lines, samples),
    }
    save_report(cfg, "QC4.3", report)
    return report



# ---------------------------------------------------------------------------
# QC5.1 - Normalisation
# ---------------------------------------------------------------------------


def to_unit(x: np.ndarray, p: dict) -> np.ndarray:
    """QC5.1: robust min-max of a layer, clip((x - low) / (high - low), 0, 1), float32."""
    return np.clip((np.asarray(x, dtype=np.float32) - p["low"]) / (p["high"] - p["low"]), 0.0, 1.0)


def from_unit(u: np.ndarray, p: dict) -> np.ndarray:
    """Inverse of ``to_unit`` (exact where the value was not clipped)."""
    return p["low"] + np.asarray(u, dtype=np.float64) * (p["high"] - p["low"])


def backbone_input(channels: list[np.ndarray], mean, std) -> np.ndarray:
    """QC5.1: 3 layers in [0, 1] -> (3, H, W) float32 input of the backbone, (u - mean[c]) / std[c]."""
    if len(channels) != len(mean):
        raise ValueError(f"{len(channels)} channels for {len(mean)} mean values")
    return np.stack([(np.asarray(u, dtype=np.float32) - m) / s for u, m, s in zip(channels, mean, std)])


def backbone_to_unit(x: np.ndarray, mean, std) -> np.ndarray:
    """Inverse of ``backbone_input``: (3, H, W) -> layers in [0, 1]."""
    return np.stack([x[c] * s + m for c, (m, s) in enumerate(zip(mean, std))])


def load_normalisation(cfg: dict) -> dict:
    """Parameters of QC5.1 (``<stage>/normalisation.json``)."""
    return json.loads((stage_dir(cfg) / "normalisation.json").read_text())


def tile_footprint_cells(tiles, use: np.ndarray, cells_shape: tuple[int, int], tile_px: int, cell_px: int) -> np.ndarray:
    """Cells covered by the footprint of the tiles selected by ``use`` (tile origins at whole cells)."""
    n = tile_px // cell_px
    cov = np.zeros(cells_shape, dtype=bool)
    for r0, c0 in zip(tiles["row0"].to_numpy()[use] // cell_px, tiles["col0"].to_numpy()[use] // cell_px):
        cov[r0:r0 + n, c0:c0 + n] = True
    return cov


def run_norm(cfg: dict, force: bool = False, checks_only: bool = False) -> dict:
    """QC5.1: percentiles of every layer on the S1-train tiles, clipped fractions per set, checks."""
    import pandas as pd
    import timm

    from qc_checks import DINO_MODEL

    ncfg = cfg["normalisation"]
    q_low, q_high = (float(q) for q in ncfg["percentiles"])
    step = int(ncfg["sample_step"])
    out = stage_dir(cfg) / "normalisation.json"
    if out.exists() and not (force or checks_only):
        raise FileExistsError(f"{out} exists: a stage is not rewritten (use --force or --checks-only)")
    tile_px, cell_px = int(cfg["tiling"]["tile_px"]), int(cfg["tiling"]["cell_px"])
    if cell_px % (2 * step) or STRIP_ROWS % cell_px:
        raise ValueError("sample step must divide half a cell")
    names = list(dict.fromkeys(n for inp in ncfg["backbone_inputs"].values() for n in inp))
    layers = {n: np.load(raster_path(cfg, n), mmap_mode="r") for n in names}
    lines, samples = layers[names[0]].shape
    cells_shape = (lines // cell_px, samples // cell_px)

    # pixels of every set = footprints of its tiles kept by the buffer (QB3.2) and valid (QC3.1)
    tiles = pd.read_csv(cfg["paths"]["output"] / "tiles.csv")
    used = tiles["kept"].to_numpy(bool) & np.load(cfg["paths"]["output"] / "tile_validity.npz")["tile_used"].ravel()
    cover = {k: tile_footprint_cells(tiles, used & (tiles["set"].to_numpy() == k), cells_shape, tile_px, cell_px)
             for k in SETS}

    # one pass: every step-th row/column of the footprints of every set (and every 2*step-th of S1-train)
    samples_of = {k: {n: [] for n in names} for k in SETS}
    coarse = {n: [] for n in names}
    for r0 in range(0, cells_shape[0] * cell_px, STRIP_ROWS):
        r1 = min(r0 + STRIP_ROWS, cells_shape[0] * cell_px)
        i0, i1 = r0 // cell_px, r1 // cell_px
        sub = {n: np.asarray(a[r0:r1:step, :cells_shape[1] * cell_px:step], dtype=np.float32) for n, a in layers.items()}
        rep_ = cell_px // step
        for k in SETS:
            sel = np.repeat(np.repeat(cover[k][i0:i1], rep_, 0), rep_, 1)
            if not sel.any():
                continue
            for n in names:
                ok = sel & np.isfinite(sub[n])
                samples_of[k][n].append(sub[n][ok])
                if k == "s1_train":
                    coarse[n].append(sub[n][::2, ::2][ok[::2, ::2]])
    samples_of = {k: {n: np.concatenate(v) for n, v in d.items()} for k, d in samples_of.items()}
    coarse = {n: np.concatenate(v) for n, v in coarse.items()}

    params, per_set, stability = {}, {}, {}
    for n in names:
        lo, hi = np.percentile(samples_of["s1_train"][n].astype(np.float64), [q_low, q_high])
        lo2, hi2 = np.percentile(coarse[n].astype(np.float64), [q_low, q_high])
        params[n] = {"low": float(lo), "high": float(hi), "raster": cfg["preprocess"]["layers"][n],
                     "unit": json.loads(raster_path(cfg, n).with_suffix(".json").read_text()).get("unit")}
        stability[n] = {"low_step_2x": float(lo2), "high_step_2x": float(hi2),
                        "relative_change": float(max(abs(lo2 - lo), abs(hi2 - hi)) / (hi - lo))}
    for k in SETS:
        per_set[k] = {"pixels_sampled": int(samples_of[k][names[0]].size),
                      **{n: {"below_low": float((samples_of[k][n] < params[n]["low"]).mean()),
                             "above_high": float((samples_of[k][n] > params[n]["high"]).mean()),
                             "median_unit": float(np.median(to_unit(samples_of[k][n], params[n])))} for n in names}}

    pre = timm.models.get_pretrained_cfg(DINO_MODEL)
    mean, std = [float(v) for v in pre.mean], [float(v) for v in pre.std]
    norm = {
        "step": "QC5.1", "stage": cfg["preprocess"]["stage"],
        "statistics_from": f"pixels of the S1-train tiles kept by the buffer (QB3.2) and valid (QC3.1), "
                           f"every {step}th row and column",
        "percentiles": [q_low, q_high], "layers": params,
        "to_unit": "u = clip((x - low) / (high - low), 0, 1)", "from_unit": "x = low + u * (high - low) (exact where not clipped)",
        "backbone": {"model": DINO_MODEL, "mean": mean, "std": std,
                     "input": "x_c = (u_c - mean[c]) / std[c] for the 3 layers of backbone_inputs, in this order",
                     "inverse": "u_c = x_c * std[c] + mean[c]"},
        "backbone_inputs": {k: list(v) for k, v in ncfg["backbone_inputs"].items()},
    }
    if checks_only and out.exists() and not force:
        # the stage keeps its parameters: they must equal the ones computed now
        if json.loads(out.read_text()) != json.loads(json.dumps(norm)):
            raise AssertionError(f"{out} differs from the parameters computed now")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(norm, indent=2))
        written(out)

    # checks on a real window of every layer (first tile of S1-train): round trip of to_unit /
    # backbone_input, values in [0, 1] and the backbone input range
    t = tiles[used & (tiles["set"].to_numpy() == "s1_train")].iloc[0]
    win = (slice(int(t.row0), int(t.row0) + tile_px), slice(int(t.col0), int(t.col0) + tile_px))
    rt_err, unit_ok, ranges = 0.0, True, {}
    for k, inp in ncfg["backbone_inputs"].items():
        raw = [np.asarray(layers[n][win], dtype=np.float64) for n in inp]
        u = [to_unit(x, params[n]) for x, n in zip(raw, inp)]
        unit_ok &= all(float(v.min()) >= 0.0 and float(v.max()) <= 1.0 for v in u)
        x = backbone_input(u, mean, std)
        ranges[k] = [float(x.min()), float(x.max())]
        back = backbone_to_unit(x, mean, std)
        for c, (r, n) in enumerate(zip(raw, inp)):
            inside = (r >= params[n]["low"]) & (r <= params[n]["high"])
            rt = from_unit(back[c], params[n])
            rt_err = max(rt_err, float((np.abs(rt - r)[inside] / (params[n]["high"] - params[n]["low"])).max()))
    report = {
        "output": str(out), "parameters": params, "backbone_mean": mean, "backbone_std": std,
        "backbone_inputs": norm["backbone_inputs"], "sample_step": step, "per_set": per_set,
        "check_percentiles_with_step_2x": stability,
        "check_window": {"tile": int(t.tile), "rows": [win[0].start, win[0].stop], "cols": [win[1].start, win[1].stop],
                         "unit_values_in_0_1": bool(unit_ok), "backbone_input_range": ranges,
                         "round_trip_max_error_fraction_of_range": rt_err},
    }
    save_report(cfg, "QC5.1", report)
    return report


def selftest() -> None:
    """Synthetic checks of the QC3.1 functions."""
    wac = np.arange(36, dtype=np.float32).reshape(6, 6)
    dem = np.ones((6, 6), dtype=np.float32)
    wac[0, 0], dem[5, 5] = np.nan, np.nan
    v = valid_mask(wac, dem)
    assert v.sum() == 34 and not v[0, 0] and not v[5, 5]
    f = fill_invalid(wac, v)
    assert np.isfinite(f).all() and f[0, 0] == f[5, 5] == np.median(wac[v]) and np.array_equal(f[v], wac[v])
    assert np.array_equal(fill_invalid(dem, np.ones_like(v)), dem, equal_nan=True)   # nothing to fill: unchanged
    try:
        fill_invalid(wac, np.zeros_like(v))
        raise AssertionError("empty tile not refused")
    except ValueError:
        pass
    # cell counts and tile sums against brute force: 4 px cells, tiles of 4 cells, stride 3 cells
    rng = np.random.default_rng(0)
    bad = rng.random((44, 58)) < 0.1
    cc = cell_counts(bad, 4, 14)
    assert cc.shape == (11, 14) and all(cc[i, j] == bad[4 * i:4 * i + 4, 4 * j:4 * j + 4].sum()
                                        for i in range(11) for j in range(14))
    ts = tile_sums(cc, tile_grid_shape(44, 58, 16, 12), 4, 3)
    assert ts.shape == (3, 4) and all(ts[i, j] == bad[12 * i:12 * i + 16, 12 * j:12 * j + 16].sum()
                                      for i in range(3) for j in range(4))
    # QC4.1: smoothing by strips (any strip, borders included) equals filtering the whole raster
    from scipy.ndimage import gaussian_filter
    z = rng.normal(size=(70, 45)).cumsum(0).cumsum(1)
    whole = gaussian_filter(z, 3.0, mode="reflect", truncate=4.0)
    for a, b in ((0, 5), (5, 30), (30, 70), (0, 70), (60, 70)):
        assert np.allclose(smooth_rows(z, slice(a, b), 3.0, 4.0), whole[a:b], atol=1e-10)
    assert np.allclose(local_relief(z, slice(0, 70), 3.0, 4.0), z - whole, atol=1e-10)
    # QC4.2: a plane z = a * east_m + b * north_m on a grid at 20-30 deg N (E-W spacing shrinks
    # with cos lat): exact gradient and slope everywhere, strips equal to the whole raster
    g = CylGrid(30.0, 0.0, 10.0, 100, 80)
    lat, lon = g.rc_to_latlon(np.arange(100), np.arange(80))
    px = pixel_m(g)
    a, b = 0.05, -0.02
    east = (np.arange(80) * px)[None, :] * np.cos(np.radians(lat))[:, None]
    north = (lat[:, None] - lat[-1]) / (1 / g.ppd) * px + 0 * east
    zp = a * east + b * north
    gx, gy = dem_gradient(zp, slice(0, 100), g)
    assert np.allclose(gx, a) and np.allclose(gy[:, 0], b)        # gy exact (north linear in rows)
    assert np.allclose(slope_deg(zp, slice(0, 100), g)[:, 0], np.degrees(np.arctan(np.hypot(a, b))))
    whole_s = slope_deg(z, slice(0, 70), CylGrid(1.0, 0.0, 10.0, 70, 45))
    for r0, r1 in ((0, 1), (1, 30), (30, 70), (69, 70)):
        assert np.allclose(slope_deg(z, slice(r0, r1), CylGrid(1.0, 0.0, 10.0, 70, 45)), whole_s[r0:r1])
    # QC4.3: flat ground -> sin(alt); a slope facing the Sun at 90 - alt -> 1; facing away -> shadow;
    # the mean of 4 equally spaced azimuths = sin(alt) cos(slope) where nothing is in shadow
    assert np.isclose(hillshade(np.array(0.0), np.array(0.0), 0.0, 30.0), 0.5)
    t = np.tan(np.radians(60.0))
    assert np.isclose(hillshade(np.array(0.0), np.array(-t), 0.0, 30.0), 1.0)      # descends northwards = faces north
    assert hillshade(np.array(0.0), np.array(t), 0.0, 30.0) == 0.0
    assert np.isclose(hillshade(np.array(-t), np.array(0.0), 90.0, 30.0), 1.0)     # faces east, Sun in the east
    gx, gy = rng.normal(scale=0.2, size=(2, 50))
    m4 = sum(hillshade(gx, gy, a, 30.0, clip=False) for a in (0, 90, 180, 270)) / 4
    assert np.allclose(m4, 0.5 / np.sqrt(1 + gx ** 2 + gy ** 2))
    # QC5.1: robust min-max, clipping and inverses
    pr = {"low": -100.0, "high": 300.0}
    x = np.array([-500.0, -100.0, 100.0, 300.0, 900.0])
    u = to_unit(x, pr)
    assert np.allclose(u, [0, 0, 0.5, 1, 1]) and np.allclose(from_unit(u[1:4], pr), x[1:4])
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ch = [rng.random((4, 5)) for _ in range(3)]
    xb = backbone_input(ch, mean, std)
    assert xb.shape == (3, 4, 5) and np.isclose(xb[1, 0, 0], (ch[1][0, 0] - 0.456) / 0.224)
    assert np.allclose(backbone_to_unit(xb, mean, std), np.stack(ch), atol=1e-6)
    print("selftest OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["wac", "dem", "dem-grid", "coreg", "mask", "relief", "slope", "hillshade", "norm"])
    ap.add_argument("--force", action="store_true", help="rewrite a raster that already exists")
    ap.add_argument("--checks-only", action="store_true", help="recompute the checks of an existing raster")
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if args.command is None:
        ap.error("give a command or --selftest")
    cfg = load_config(args.config)
    if args.command == "wac":
        print(json.dumps(run_wac(cfg, force=args.force, checks_only=args.checks_only), indent=2))
    elif args.command == "dem":
        print(json.dumps(run_dem(cfg), indent=2))
    elif args.command == "dem-grid":
        print(json.dumps(run_dem_grid(cfg, force=args.force, checks_only=args.checks_only), indent=2))
    elif args.command == "coreg":
        print(json.dumps(run_coreg(cfg), indent=2))
    elif args.command == "mask":
        print(json.dumps(run_mask(cfg), indent=2))
    elif args.command == "relief":
        print(json.dumps(run_relief(cfg, force=args.force, checks_only=args.checks_only), indent=2))
    elif args.command == "slope":
        print(json.dumps(run_slope(cfg, force=args.force, checks_only=args.checks_only), indent=2))
    elif args.command == "hillshade":
        print(json.dumps(run_hillshade(cfg, force=args.force, checks_only=args.checks_only), indent=2))
    elif args.command == "norm":
        print(json.dumps(run_norm(cfg, force=args.force, checks_only=args.checks_only), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
