#!/usr/bin/env python3
"""Agreement between the DEM sources: height (plan step B3.1) and horizontal shift (B3.2).

The DEM layer will be merged from two products (D2.2): SLDEM2015 (LOLA laser
+ Kaguya stereo images) below 60 deg, the LOLA polar DEM ``LDEM_45N/S_100M``
(laser only) above 60 deg and in the polar grid margin.  If one is offset
from the other, the merged DEM shows a step along the 60 deg parallel.  This
script measures, where two sources cover the same ground:

* SLDEM (512 and 256 ppd) vs the LOLA polar DEM in the 45-60 deg overlap
  (the 60 deg junction);
* SLDEM vs the LOLA cylindrical DEM ``LDEM_256`` (118 m) at every latitude
  of SLDEM (0-60 deg): the general agreement of the two processings.

Method
------
1. Heights: both products store the planetary radius (SLDEM in km, LOLA in
   m); they are converted to height in metres above the 1737.4 km sphere,
   the reference of D2.1.
2. Common scale: the two sources have different pixel sizes, and part of
   their pixel-by-pixel difference would only be the finer detail of one of
   them.  Both are therefore averaged to about ``COMPARE_M`` m: SLDEM in
   b x b blocks of its cylindrical grid; the polar DEM in 2 x 2 blocks
   aligned to the pole, then sampled (bilinear) at the centres of the SLDEM
   blocks; LDEM_256 shares the 256 px/deg grid of SLDEM, so SLDEM 512 is
   averaged in 2 x 2 blocks and compared pixel by pixel (no interpolation).
3. Statistics of the difference (SLDEM - LOLA), area-weighted (cos lat),
   per hemisphere and latitude band: mean (= vertical bias, the value D2.1
   corrects), median, RMSE, standard deviation (RMSE once the bias is
   removed), robust sigma (1.4826 x median absolute deviation: the spread
   of most pixels, insensitive to a few local artefacts) and the 1st / 99th
   percentiles.  The per-pixel noise of the two DEMs
   (``data_quality_metrics.csv``) is reported for comparison.

4. Horizontal shift (B3.2): windows of ``WIN`` x ``WIN`` cells of ~118 m
   on a regular grid of both comparisons; both DEMs are turned into a
   hillshade (the slopes lit from the north-west, i.e. an image of the
   relief) and the shift of LOLA relative to SLDEM is found by phase
   correlation (``phase_shift``), refined to 1/``UPSAMPLE`` px.  A window
   is valid if its correlation peak is >= ``PEAK_MIN``.

Output in ``<root>/preprocessing/diagnostics/B3/``: ``dem_sources.csv``
(B3.1, one row per comparison, hemisphere and latitude band) and
``dem_shifts.csv`` (B3.2, one row per window).

Usage
-----
    python Dataset/scripts/B_diagnostics/diag_dem_sources.py              # B3.1 and B3.2
    python Dataset/scripts/B_diagnostics/diag_dem_sources.py --step B3.2  # shifts only
    python Dataset/scripts/B_diagnostics/diag_dem_sources.py --selftest   # shift method on synthetic DEMs
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

_SCRIPTS = Path(__file__).resolve().parents[1]   # Dataset/scripts: one folder per plan chapter
sys.path[:0] = [str(_SCRIPTS / "A_sources"), str(Path(__file__).resolve().parent)]

from data_quality import DEFAULT_ROOT, inventory_path, metrics_path, diagnostics_dir, open_image, read_physical, robust_sigma  # noqa: E402
from diag_lamp_snr import block_mean  # noqa: E402

REF_RADIUS_M = 1737400.0
COMPARE_M = 200.0              # target scale of the comparison with the polar DEM
POLAR_BLOCK = 2                # polar DEM 100 m -> 200 m
POLAR_BAND = (45.0, 60.0)      # |lat| overlap of SLDEM and the polar DEM
LAT_BANDS = ((0, 15), (15, 30), (30, 45), (45, 50), (50, 55), (55, 60))
CHUNK_DEG = 0.5                # latitude rows read at once
LON_CHUNK_DEG = 15.0           # longitude span sampled at once in the polar DEM
SAMPLE_CAP = 2_000_000         # differences kept per group for the percentiles
# B3.2
SHIFT_M = 118.0                # cell of the shift windows (SLDEM 512 2x2, SLDEM 256 and LDEM_256 native)
WIN = 256                      # window side in cells (~30 km)
SHIFT_STEP_POLAR = (3.0, 8.0)  # window spacing (deg lat, deg lon) in the 45-60 deg overlap
SHIFT_STEP_LDEM = (5.0, 10.0)  # window spacing vs LDEM_256 (0-60 deg)
SUN_AZ, SUN_ALT = 315.0, 30.0  # hillshade illumination (from the north-west, 30 deg above the horizon)
PEAK_MIN = 0.05               # phase-correlation peak below this: no reliable match (flat window)
UPSAMPLE = 20                  # sub-pixel refinement of the shift: 1/20 px

COLUMNS = [
    "comparison", "source_a", "source_b", "hemisphere", "lat_band", "compare_m", "pixels",
    "bias_mean", "median", "rmse", "std", "robust_sigma", "p1", "p99", "noise_a", "noise_b",
]
SHIFT_COLUMNS = [
    "comparison", "source_a", "source_b", "hemisphere", "lat", "lon", "pixel_m", "shift_e_px", "shift_n_px",
    "shift_e_m", "shift_n_m", "peak", "relief_m", "valid",
]


def height_m(x: np.ndarray, unit: str) -> np.ndarray:
    """Planetary radius (km or m) -> height in m above the reference sphere."""
    return x * (1000.0 if unit.strip().lower() == "km" else 1.0) - REF_RADIUS_M


@dataclass
class Acc:
    """Area-weighted running statistics of a difference, plus a capped sample for the quantiles."""
    w: float = 0.0
    wd: float = 0.0
    wd2: float = 0.0
    n: int = 0
    sample: list = field(default_factory=list)
    kept: int = 0

    def add(self, d: np.ndarray, w: np.ndarray) -> None:
        ok = np.isfinite(d)
        d, w = d[ok], w[ok]
        if d.size == 0:
            return
        self.w += float(w.sum())
        self.wd += float((w * d).sum())
        self.wd2 += float((w * d * d).sum())
        self.n += d.size
        step = max(1, self.n // SAMPLE_CAP)   # thin the sample as the count grows
        self.sample.append(d[::step].astype(np.float32))
        self.kept += self.sample[-1].size
        if self.kept > 2 * SAMPLE_CAP:
            s = np.concatenate(self.sample)[::2]
            self.sample, self.kept = [s], s.size

    def stats(self) -> dict[str, float]:
        if self.n < 100:
            return {"pixels": self.n}
        mean = self.wd / self.w
        rmse = math.sqrt(self.wd2 / self.w)
        s = np.concatenate(self.sample).astype(np.float64)
        return dict(pixels=self.n, bias_mean=mean, median=float(np.median(s)), rmse=rmse,
                    std=math.sqrt(max(rmse * rmse - mean * mean, 0.0)), robust_sigma=robust_sigma(s),
                    p1=float(np.percentile(s, 1)), p99=float(np.percentile(s, 99)))


def band_of(abs_lat: float) -> str | None:
    for lo, hi in LAT_BANDS:
        if lo <= abs_lat < hi:
            return f"{lo}-{hi}"
    return None


def sldem_chunks(row: dict[str, str], root: Path, b: int, abs_lo: float, abs_hi: float):
    """SLDEM tile in chunks of rows within lo <= |lat| <= hi, as heights averaged in b x b blocks.

    Yields (height blocks, block-centre latitudes, block-centre longitudes)."""
    raster, _, sentinels = open_image(row, root)
    lat0, lat1 = float(row["bbox_min_lat_deg"]), float(row["bbox_max_lat_deg"])
    lon0, lon1 = float(row["bbox_min_lon_deg"]), float(row["bbox_max_lon_deg"])
    ppd = raster.lines / (lat1 - lat0)
    lon_ppd = raster.samples / (lon1 - lon0)
    # rows whose latitude is inside the band (row 0 = lat1)
    top = max(min(lat1, abs_hi if lat1 > 0 else -abs_lo), lat0)
    bottom = min(max(lat0, abs_lo if lat1 > 0 else -abs_hi), lat1)
    r_start, r_stop = int(round((lat1 - top) * ppd)), int(round((lat1 - bottom) * ppd))
    r_start -= r_start % b
    chunk = max(b, int(CHUNK_DEG * ppd) // b * b)
    lon = lon0 + (np.arange(raster.samples // b) * b + b / 2.0) / lon_ppd
    for s in range(r_start, r_stop, chunk):
        e = min(s + chunk, r_stop)
        e -= (e - s) % b
        if e <= s:
            continue
        x = height_m(read_physical(raster, sentinels, slice(s, e), slice(0, raster.samples // b * b)), row["unit"])
        lat = lat1 - (np.arange(s, e, b) + b / 2.0) / ppd
        yield block_mean(x, b), lat, lon


def polar_sampler(row: dict[str, str], root: Path, k: int = POLAR_BLOCK):
    """Function (lat 2-D, lon 2-D) -> polar DEM height averaged in k x k blocks, bilinear."""
    raster, geom, sentinels = open_image(row, root)
    unit = row["unit"]

    def sample(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        rf, cf = geom.polar_index(lat, lon)               # fine pixel-centre indices
        r0 = max(0, int(np.floor(rf.min())) - 2 * k)
        c0 = max(0, int(np.floor(cf.min())) - 2 * k)
        r0 -= (r0 - int(round(geom.line_off))) % k        # blocks aligned to the pole
        c0 -= (c0 - int(round(geom.samp_off))) % k
        r0, c0 = max(r0, 0), max(c0, 0)
        r1 = min(raster.lines, int(np.ceil(rf.max())) + 2 * k + 1)
        c1 = min(raster.samples, int(np.ceil(cf.max())) + 2 * k + 1)
        win = block_mean(height_m(read_physical(raster, sentinels, slice(r0, r1), slice(c0, c1)), unit), k)
        # fine centre index i -> coarse index (i - start + 0.5) / k - 0.5
        rc = (rf - r0 + 0.5) / k - 0.5
        cc = (cf - c0 + 0.5) / k - 0.5
        return map_coordinates(win, [rc, cc], order=1, cval=np.nan, prefilter=False)

    return sample


def compare_polar(sldem: list[dict[str, str]], polar: dict[str, dict[str, str]], root: Path,
                  accs: dict) -> None:
    for row in sldem:
        north = float(row["bbox_max_lat_deg"]) > 0
        p = polar.get("N" if north else "S")
        if p is None:
            continue
        b = max(1, int(round(COMPARE_M / float(row["native_resolution_m"]))))
        sample = polar_sampler(p, root)
        lon_span = float(row["bbox_max_lon_deg"]) - float(row["bbox_min_lon_deg"])
        for h, lat, lon in sldem_chunks(row, root, b, *POLAR_BAND):
            n_lon = max(1, int(math.ceil(lon_span / LON_CHUNK_DEG)))
            for cols in np.array_split(np.arange(lon.size), n_lon):
                la, lo = np.meshgrid(lat, lon[cols], indexing="ij")
                d = h[:, cols] - sample(la, lo)
                w = np.cos(np.radians(la))
                for i, lt in enumerate(lat):
                    band = band_of(abs(lt))
                    if band:
                        key = (row["_product"], p["product_id"], "N" if north else "S", band)
                        accs.setdefault(key, Acc()).add(d[i], w[i])
        print(f"  {row['product_id']} vs {p['product_id']} done", flush=True)


def compare_ldem256(sldem: list[dict[str, str]], ldem: list[dict[str, str]], root: Path, accs: dict) -> None:
    for row in sldem:
        lat0, lat1 = float(row["bbox_min_lat_deg"]), float(row["bbox_max_lat_deg"])
        lon0, lon1 = float(row["bbox_min_lon_deg"]), float(row["bbox_max_lon_deg"])
        ref = next((r for r in ldem if float(r["bbox_min_lat_deg"]) <= lat0 and float(r["bbox_max_lat_deg"]) >= lat1
                    and float(r["bbox_min_lon_deg"]) <= lon0 and float(r["bbox_max_lon_deg"]) >= lon1), None)
        if ref is None:
            continue
        rr, _, rs = open_image(ref, root)
        ref_ppd = rr.lines / (float(ref["bbox_max_lat_deg"]) - float(ref["bbox_min_lat_deg"]))
        sl_raster, _, _ = open_image(row, root)
        b = int(round(sl_raster.lines / (lat1 - lat0) / ref_ppd))       # 512 -> 2, 256 -> 1
        r_off = int(round((float(ref["bbox_max_lat_deg"]) - lat1) * ref_ppd))
        c_off = int(round((lon0 - float(ref["bbox_min_lon_deg"])) * ref_ppd))
        north = lat1 > 0
        for h, lat, lon in sldem_chunks(row, root, b, 0.0, 60.0):
            rows = np.round((float(ref["bbox_max_lat_deg"]) - lat) * ref_ppd - 0.5).astype(int)
            g = height_m(read_physical(rr, rs, slice(rows[0], rows[-1] + 1),
                                       slice(c_off, c_off + h.shape[1])), ref["unit"])
            d = h - g
            w = np.cos(np.radians(lat))[:, None] * np.ones((1, d.shape[1]))
            for i, lt in enumerate(lat):
                band = band_of(abs(lt))
                if band:
                    key = (row["_product"], "LDEM_256", "N" if north else "S", band)
                    accs.setdefault(key, Acc()).add(d[i], w[i])
        print(f"  {row['product_id']} vs {ref['product_id']} done", flush=True)


# ---------------------------------------------------------------------------
# B3.2: horizontal shift (phase correlation of hillshades)
# ---------------------------------------------------------------------------


def hillshade(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Lambertian shading of a DEM (rows southwards, cols eastwards) lit from SUN_AZ / SUN_ALT."""
    gy_rows, gx = np.gradient(z, dy, dx)
    gn = -gy_rows                                     # slope towards north
    az, alt = math.radians(SUN_AZ), math.radians(SUN_ALT)
    sx, sy, sz = math.cos(alt) * math.sin(az), math.cos(alt) * math.cos(az), math.sin(alt)
    return (-gx * sx - gn * sy + sz) / np.sqrt(1.0 + gx * gx + gn * gn)


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Shift (rows, cols) in pixels with b(y, x) ~ a(y - dy, x - dx), and the correlation peak height.

    Phase correlation: the cross-power spectrum normalised to unit amplitude keeps only the phase
    difference of the two images, whose inverse transform is a sharp peak at the shift.  The
    images are tapered with a Hann window (no edge jump).  The integer peak is refined to
    1/UPSAMPLE px by evaluating the inverse transform on a fine grid around it only (matrix-
    multiply DFT, Guizar-Sicairos et al. 2008, the method of scikit-image): a parabola through
    the peak was tried and discarded, it underestimates fractional shifts (0.25 px -> 0.10 px).
    """
    n, m = a.shape
    taper = np.outer(np.hanning(n), np.hanning(m))
    fa = np.fft.fft2((a - a.mean()) * taper)
    fb = np.fft.fft2((b - b.mean()) * taper)
    cross = np.conj(fa) * fb
    cross /= np.abs(cross) + 1e-12
    r = np.real(np.fft.ifft2(cross))
    i, j = np.unravel_index(int(np.argmax(r)), r.shape)
    peak = float(r[i, j])
    shift = np.array([(i + n / 2) % n - n / 2, (j + m / 2) % m - m / 2], dtype=float)
    # fine grid of UPSAMPLE points per pixel, 1.5 px wide, centred on the integer peak
    region = int(math.ceil(UPSAMPLE * 1.5))
    centre = region // 2
    offset = centre - shift * UPSAMPLE
    k_row = np.exp(2j * np.pi * (np.arange(region)[:, None] - offset[0]) * np.fft.fftfreq(n, UPSAMPLE)[None, :])
    k_col = np.exp(2j * np.pi * np.fft.fftfreq(m, UPSAMPLE)[:, None] * (np.arange(region)[None, :] - offset[1]))
    fine = np.real(k_row @ cross @ k_col)
    fi, fj = np.unravel_index(int(np.argmax(fine)), fine.shape)
    dy, dx = shift + (np.array([fi, fj]) - centre) / UPSAMPLE
    return float(dy), float(dx), peak


class Products:
    """Opened rasters, cached (opening an image samples it to find the fill values)."""

    def __init__(self, root: Path):
        self.root, self.cache = root, {}

    def get(self, row: dict[str, str]):
        pid = row["product_id"]
        if pid not in self.cache:
            self.cache[pid] = open_image(row, self.root)
        return self.cache[pid]


def sldem_window(prod: Products, row: dict[str, str], b: int, lat_c: float, lon_c: float):
    """WIN x WIN window of SLDEM heights averaged in b x b blocks around (lat_c, lon_c), or None."""
    raster, _, sentinels = prod.get(row)
    lat0, lat1 = float(row["bbox_min_lat_deg"]), float(row["bbox_max_lat_deg"])
    lon0, lon1 = float(row["bbox_min_lon_deg"]), float(row["bbox_max_lon_deg"])
    ppd, lon_ppd = raster.lines / (lat1 - lat0), raster.samples / (lon1 - lon0)
    size = WIN * b
    r0 = int(round((lat1 - lat_c) * ppd - size / 2))
    c0 = int(round((lon_c - lon0) * lon_ppd - size / 2))
    r0, c0 = r0 - r0 % b, c0 - c0 % b
    if r0 < 0 or c0 < 0 or r0 + size > raster.lines or c0 + size > raster.samples:
        return None
    z = block_mean(height_m(read_physical(raster, sentinels, slice(r0, r0 + size), slice(c0, c0 + size)),
                            row["unit"]), b)
    lat = lat1 - (r0 + np.arange(WIN) * b + b / 2.0) / ppd
    lon = lon0 + (c0 + np.arange(WIN) * b + b / 2.0) / lon_ppd
    return z, lat, lon


def ldem_window(prod: Products, ref: dict[str, str], lat: np.ndarray, lon: np.ndarray) -> np.ndarray | None:
    """LDEM_256 pixels at the given cell-centre latitudes / longitudes (same 256 px/deg grid)."""
    raster, _, sentinels = prod.get(ref)
    ppd = raster.lines / (float(ref["bbox_max_lat_deg"]) - float(ref["bbox_min_lat_deg"]))
    r0 = int(round((float(ref["bbox_max_lat_deg"]) - lat[0]) * ppd - 0.5))
    c0 = int(round((lon[0] - float(ref["bbox_min_lon_deg"])) * ppd - 0.5))
    if r0 < 0 or c0 < 0 or r0 + lat.size > raster.lines or c0 + lon.size > raster.samples:
        return None
    return height_m(read_physical(raster, sentinels, slice(r0, r0 + lat.size), slice(c0, c0 + lon.size)), ref["unit"])


def measure_shift(z_a: np.ndarray, z_b: np.ndarray, lat_c: float, cell_deg: float) -> dict[str, float] | None:
    """Shift of DEM b relative to DEM a (metres east / north) from their hillshades."""
    if np.isnan(z_a).mean() > 0.01 or np.isnan(z_b).mean() > 0.01:
        return None
    z_a = np.where(np.isfinite(z_a), z_a, np.nanmean(z_a))
    z_b = np.where(np.isfinite(z_b), z_b, np.nanmean(z_b))
    dy_m = math.radians(cell_deg) * REF_RADIUS_M
    dx_m = dy_m * math.cos(math.radians(lat_c))
    ha, hb = hillshade(z_a, dx_m, dy_m), hillshade(z_b, dx_m, dy_m)
    sy, sx, peak = phase_shift(ha, hb)
    return dict(shift_e_m=sx * dx_m, shift_n_m=-sy * dy_m, shift_e_px=sx, shift_n_px=-sy, peak=peak,
                relief_m=robust_sigma(z_a), pixel_m=dy_m,
                valid=bool(peak >= PEAK_MIN and abs(sx) < WIN / 4 and abs(sy) < WIN / 4))


def window_centres(row: dict[str, str], abs_lo: float, abs_hi: float, lat_step: float, lon_step: float):
    lat0, lat1 = float(row["bbox_min_lat_deg"]), float(row["bbox_max_lat_deg"])
    lon0, lon1 = float(row["bbox_min_lon_deg"]), float(row["bbox_max_lon_deg"])
    lo, hi = max(lat0, abs_lo if lat1 > 0 else -abs_hi), min(lat1, abs_hi if lat1 > 0 else -abs_lo)
    for lat in np.arange(lo + lat_step / 2, hi, lat_step):
        for lon in np.arange(lon0 + lon_step / 2, lon1, lon_step):
            yield float(lat), float(lon)


def shifts(sldem: list[dict[str, str]], polar: dict[str, dict[str, str]], ldem: list[dict[str, str]],
           root: Path) -> list[dict[str, object]]:
    prod = Products(root)
    out = []
    samplers = {}
    for row in sldem:
        north = float(row["bbox_max_lat_deg"]) > 0
        hemi = "N" if north else "S"
        res = float(row["native_resolution_m"])
        b = max(1, int(round(SHIFT_M / res)))                       # 512 -> 2, 256 -> 1: 118 m cells
        cell_deg = b / (prod.get(row)[0].lines / (float(row["bbox_max_lat_deg"]) - float(row["bbox_min_lat_deg"])))
        jobs = []
        p = polar.get(hemi)
        if p is not None:
            if p["product_id"] not in samplers:
                samplers[p["product_id"]] = polar_sampler(p, root, k=1)
            jobs.append(("LOLA polar DEM", p["product_id"], POLAR_BAND, SHIFT_STEP_POLAR,
                         lambda lat, lon, s=samplers[p["product_id"]]: s(*np.meshgrid(lat, lon, indexing="ij"))))
        lat0, lat1 = float(row["bbox_min_lat_deg"]), float(row["bbox_max_lat_deg"])
        lon0, lon1 = float(row["bbox_min_lon_deg"]), float(row["bbox_max_lon_deg"])
        ref = next((r for r in ldem if float(r["bbox_min_lat_deg"]) <= lat0 and float(r["bbox_max_lat_deg"]) >= lat1
                    and float(r["bbox_min_lon_deg"]) <= lon0 and float(r["bbox_max_lon_deg"]) >= lon1), None)
        if ref is not None:
            jobs.append(("LDEM_256", ref["product_id"], (0.0, 60.0), SHIFT_STEP_LDEM,
                         lambda lat, lon, r=ref: ldem_window(prod, r, lat, lon)))
        for name, ref_id, (lo, hi), (lat_step, lon_step), reference in jobs:
            for lat_c, lon_c in window_centres(row, lo, hi, lat_step, lon_step):
                w = sldem_window(prod, row, b, lat_c, lon_c)
                if w is None:
                    continue
                z, lat, lon = w
                z_ref = reference(lat, lon)
                if z_ref is None:
                    continue
                m = measure_shift(z, z_ref, lat_c, cell_deg)
                if m is None:
                    continue
                out.append(dict(comparison=f"{row['_product']} vs {name}", source_a=row["product_id"],
                                source_b=ref_id, hemisphere=hemi, lat=lat_c, lon=lon_c, **m))
            print(f"  {row['product_id']} vs {ref_id}: {sum(o['source_a'] == row['product_id'] and o['source_b'] == ref_id for o in out)} windows",
                  flush=True)
    return out


def shift_selftest() -> int:
    """A synthetic DEM shifted by a known sub-pixel amount must be recovered."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from diag_seams import _synthetic_terrain
    rng = np.random.default_rng(3)
    ok = True
    for true_e, true_n in ((0.0, 0.0), (1.3, -0.4), (-2.7, 0.8), (0.25, 0.25)):
        z = 300.0 * _synthetic_terrain(rng, WIN, WIN)
        ky = np.fft.fftfreq(WIN)[:, None]
        kx = np.fft.fftfreq(WIN)[None, :]
        # shift east by true_e px (cols +) and north by true_n px (rows -)
        moved = np.real(np.fft.ifft2(np.fft.fft2(z) * np.exp(-2j * np.pi * (kx * true_e + ky * (-true_n)))))
        moved += rng.normal(0, 2.0, moved.shape)
        m = measure_shift(z, moved, 45.0, 100.0 / REF_RADIUS_M * 180 / math.pi)
        good = abs(m["shift_e_px"] - true_e) < 0.15 and abs(m["shift_n_px"] - true_n) < 0.15 and m["valid"]
        ok &= good
        print(f"true E {true_e:+.2f} N {true_n:+.2f} px -> measured E {m['shift_e_px']:+.2f} N {m['shift_n_px']:+.2f} "
              f"(peak {m['peak']:.2f})  {'OK' if good else 'FAIL'}")
    return 0 if ok else 1


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (("" if not math.isfinite(v) else f"{v:.6g}") if isinstance(v, float) else v)
                         for k, v in ((k, r.get(k, "")) for k in columns)})
    print(f"written {path}")


def vertical(sldem, polar, ldem, noise, root: Path, skip_ldem256: bool) -> list[dict[str, object]]:
    """B3.1 rows: one per comparison, hemisphere and latitude band (plus "all")."""
    accs: dict = {}
    print("SLDEM vs LOLA polar DEM (45-60 deg)")
    compare_polar(sldem, polar, root, accs)
    if not skip_ldem256:
        print("SLDEM vs LDEM_256 (0-60 deg)")
        compare_ldem256(sldem, ldem, root, accs)

    def noise_of(prefix: str) -> float:
        v = [n for pid, n in noise.items() if pid.startswith(prefix)]
        return float(np.median(v)) if v else float("nan")

    # noise_sigma is in the stored unit: km for SLDEM (x 1000 below), m for LOLA
    out = []
    groups: dict = {}
    for (a, b, hemi, band), acc in accs.items():
        g = groups.setdefault((a, b, hemi, "all"), Acc())
        for k in ("w", "wd", "wd2", "n"):
            setattr(g, k, getattr(g, k) + getattr(acc, k))
        g.sample += acc.sample
    for (a, b, hemi, band), acc in sorted({**accs, **groups}.items(),
                                          key=lambda kv: (kv[0][1] == "LDEM_256", kv[0][0], kv[0][2], kv[0][3])):
        st = acc.stats()
        polar_cmp = b.startswith("LDEM_45")
        res_a = next(float(r["native_resolution_m"]) for r in sldem if r["_product"] == a)
        # comparison scale: SLDEM blocks of ~COMPARE_M (polar), or the LDEM_256 pixel
        compare_m = res_a * max(1, round(COMPARE_M / res_a)) if polar_cmp else 118.45
        out.append(dict(comparison=f"{a} vs {'LOLA polar DEM' if polar_cmp else b}", source_a=a, source_b=b,
                        hemisphere=hemi, lat_band=band, compare_m=round(compare_m), **st,
                        noise_a=1000.0 * noise_of(a), noise_b=noise_of(b if polar_cmp else "LDEM_256")))
        print(f"{a:14s} vs {b:14s} {hemi} {band:6s} n={st.get('pixels', 0):>10} "
              + (f"bias {st['bias_mean']:+7.2f} m  median {st['median']:+7.2f}  rmse {st['rmse']:6.2f}  "
                 f"robust sigma {st['robust_sigma']:6.2f}" if "rmse" in st else ""), flush=True)
    return out


def print_shift_summary(rows: list[dict[str, object]]) -> None:
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["comparison"], r["hemisphere"]), []).append(r)
    for (comp, hemi), rs in sorted(groups.items()):
        v = [r for r in rs if r["valid"]]
        if not v:
            print(f"{comp:32s} {hemi}: no valid window of {len(rs)}")
            continue
        e = np.array([r["shift_e_m"] for r in v])
        n = np.array([r["shift_n_m"] for r in v])
        print(f"{comp:32s} {hemi}: {len(v):3d}/{len(rs):3d} windows  shift E {np.median(e):+6.1f} m "
              f"(robust sigma {robust_sigma(e):5.1f})  N {np.median(n):+6.1f} m (robust sigma {robust_sigma(n):5.1f})  "
              f"|shift| p90 {np.percentile(np.hypot(e, n), 90):5.1f} m  (pixel {v[0]['pixel_m']:.0f} m)")


def main(argv: list[str] | None = None) -> int:
    dataset_root = DEFAULT_ROOT
    parser = argparse.ArgumentParser(description="Agreement between the DEM sources: height (B3.1), shift (B3.2).")
    parser.add_argument("--root", default=str(dataset_root))
    parser.add_argument("--out-dir", default=None, help="default: <root>/preprocessing/diagnostics/B3")
    parser.add_argument("--step", choices=("B3.1", "B3.2", "all"), default="all")
    parser.add_argument("--skip-ldem256", action="store_true", help="B3.1: only the 45-60 deg comparison with the polar DEM")
    parser.add_argument("--selftest", action="store_true", help="B3.2: recover a known shift on synthetic DEMs")
    args = parser.parse_args(argv)
    if args.selftest:
        return shift_selftest()
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else diagnostics_dir(root, "B3")
    with (inventory_path(root)).open(encoding="utf-8-sig", newline="") as fh:
        inv = list(csv.DictReader(fh))
    with (metrics_path(root)).open(encoding="utf-8-sig", newline="") as fh:
        noise = {r["product_id"]: float(r["noise_sigma"]) for r in csv.DictReader(fh) if r["noise_sigma"]}

    sldem = []
    for r in inv:
        if r["product_id"].startswith("SLDEM2015_"):
            r["_product"] = "SLDEM2015_" + r["product_id"].split("_")[1]
            sldem.append(r)
    polar = {("N" if float(r["bbox_max_lat_deg"]) > 0 else "S"): r for r in inv
             if r["product_id"].startswith("LDEM_45")}
    ldem = [r for r in inv if r["product_id"].startswith("LDEM_256_")]

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.step in ("B3.1", "all"):
        write_csv(out_dir / "dem_sources.csv", COLUMNS, vertical(sldem, polar, ldem, noise, root, args.skip_ldem256))
    if args.step in ("B3.2", "all"):
        print("B3.2: horizontal shift (phase correlation of hillshades)")
        rows = shifts(sldem, polar, ldem, root)
        print_shift_summary(rows)
        write_csv(out_dir / "dem_shifts.csv", SHIFT_COLUMNS, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
