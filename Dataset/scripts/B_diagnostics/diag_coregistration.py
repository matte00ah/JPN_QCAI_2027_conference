#!/usr/bin/env python3
"""Co-registration of every modality with the DEM (plan step B4.1).

All layers will be stacked on common grids, so a crater must fall on the same
pixels in the optical, radar, far-UV and elevation layers.  The DEM is the
geometric reference (its sources are co-registered with each other, B3.2):
this script measures the shift of each modality relative to it.

Method (per window of ``win`` x ``win`` pixels of the image, on a regular grid)
------------------------------------------------------------------------------
1. The DEM is averaged to about the image pixel (``block_mean``) and sampled
   (bilinear) at the image pixel centres: cylindrical image -> SLDEM 512
   (the dataset DEM below 60 deg, B3); polar image -> LOLA polar DEM.
2. Shape of the terrain from the DEM: slope towards east, slope towards
   north, steepness and local relief (height minus its running mean).
3. What the image "should look like" given the terrain: least-squares fit of
   the image on these four quantities.  For WAC this is the shading of the
   relief under the illumination of the mosaic (a hillshade with the WAC
   geometry, whose Sun azimuth is not in the labels and so is fitted); for
   Mini-RF the dependence of the backscatter on the slope facing the radar;
   for LAMP whatever topographic signal there is.  ``fit_r`` is how much of
   the image the terrain explains.
4. Shift of the image relative to the DEM: phase correlation of the fitted
   image with the real one (``phase_shift`` of B3.2, 1/20 px).  Noisy images
   (Mini-RF speckle, LAMP) are smoothed by a Gaussian of ``smooth`` px
   first, both sides alike.  A window is valid when ``fit_r`` >= ``FIT_MIN``
   (the terrain is visible in the image) and the correlation peak >=
   ``PEAK_MIN``.

Products: WAC_GLOBAL equirectangular and polar (100 m), WAC_HAPKE 643 nm
(400 m), Mini-RF S1 (237 m, log10), LAMP Lyman-alpha albedo and brightness
(240 m, poles).

Output: ``<root>/preprocessing/diagnostics/B4/coregistration.csv`` (one row
per window).

Usage
-----
    python Dataset/scripts/B_diagnostics/diag_coregistration.py
    python Dataset/scripts/B_diagnostics/diag_coregistration.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, uniform_filter

_SCRIPTS = Path(__file__).resolve().parents[1]   # Dataset/scripts: one folder per plan chapter
sys.path[:0] = [str(_SCRIPTS / "A_sources"), str(Path(__file__).resolve().parent)]

import data_quality  # noqa: E402
from data_quality import DEFAULT_ROOT, inventory_path, diagnostics_dir, open_image, read_physical, robust_sigma  # noqa: E402
from diag_dem_sources import REF_RADIUS_M, height_m, phase_shift  # noqa: E402
from diag_lamp_snr import block_mean  # noqa: E402

WIN = 256
FIT_MIN = 0.3            # correlation of the terrain-predicted image with the real one
PEAK_MIN = 0.05
RELIEF_WINDOW = 15       # px: running mean removed to get the local relief
MAX_NAN = 0.02           # windows with more missing pixels are skipped
ITERATIONS = 6           # refit + re-measure of the shift (see measure)


@dataclass
class Target:
    label: str
    pattern: str          # product_id prefix
    dem: str              # "cyl" (SLDEM 512) or "polar" (LOLA polar DEM)
    lat_step: float       # window spacing (deg)
    lon_step: float
    smooth: float = 0.0   # Gaussian sigma (px) applied to image and model
    log10: bool = False
    min_abs_lat: float = 0.0
    win: int = WIN        # window side (px)
    max_nan: float = MAX_NAN


TARGETS = [
    Target("WAC_GLOBAL", "WAC_GLOBAL_E", "cyl", 6.0, 12.0),
    Target("WAC_GLOBAL polar", "WAC_GLOBAL_P", "polar", 4.0, 30.0, min_abs_lat=62.0),
    Target("WAC_HAPKE 643 nm", "WAC_HAPKE_643NM", "cyl", 8.0, 16.0),
    # the Mini-RF mosaic has many gaps between strips: smaller windows, denser grid
    Target("Mini-RF S1", "GLOBAL_S1_", "cyl", 1.5, 3.0, smooth=1.5, log10=True, win=128, max_nan=0.05),
    Target("LAMP Lyman-a albedo", "LAMP_80", "polar", 2.5, 45.0, smooth=1.0, min_abs_lat=81.0),
]
LAMP_LAYERS = ("_lya_albed_", "_lya_brite_")

COLUMNS = ["target", "reading", "product_id", "dem", "lat", "lon", "pixel_m", "fit_r", "peak", "shift_e_px", "shift_n_px",
           "shift_e_m", "shift_n_m", "valid"]


class Products:
    def __init__(self, root: Path):
        self.root, self.cache = root, {}

    def get(self, row):
        pid = row["product_id"]
        if pid not in self.cache:
            self.cache[pid] = open_image(row, self.root)
        return self.cache[pid]


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------


def window_latlon(row, geom, r0: int, c0: int, n: int = WIN, step: int = 1):
    """Lat / lon of the pixel centres of an n x n window starting at (r0, c0); with ``step`` > 1,
    of the centres of its n x n blocks of step x step pixels."""
    centres = np.arange(n, dtype=float) * step + (step - 1) / 2.0
    rows = (r0 + centres)[:, None] * np.ones((1, n))
    cols = np.ones((n, 1)) * (c0 + centres)[None, :]
    if geom.polar:
        return geom.polar_latlon(rows, cols)
    lat0, lat1 = float(row["bbox_min_lat_deg"]), float(row["bbox_max_lat_deg"])
    lines, samples = int(row["lines"]), int(row["samples"])
    return lat1 - (rows + 0.5) * (lat1 - lat0) / lines, geom.column_lon(cols, samples)


def dem_sample(prod: Products, dems: list[dict[str, str]], lat: np.ndarray, lon: np.ndarray, pixel_m: float):
    """(DEM heights averaged to ~pixel_m and sampled at (lat, lon), DEM id); (None, None) if not covered."""
    for row in dems:
        raster, geom, sentinels = prod.get(row)
        k = max(1, int(round(pixel_m / float(row["native_resolution_m"]))))
        if geom.polar:
            if np.any(np.abs(lat) < geom.polar_min_abs_lat + 0.5) or (geom.max_lat > 0) != (lat.mean() > 0):
                continue
            rf, cf = geom.polar_index(lat, lon)
        else:
            lat0, lat1 = float(row["bbox_min_lat_deg"]), float(row["bbox_max_lat_deg"])
            rel = (lon - geom.west_lon) % 360.0
            if lat.min() < lat0 or lat.max() > lat1 or rel.max() > geom.lon_span or rel.max() - rel.min() > 180:
                continue
            rf = (lat1 - lat) * raster.lines / (lat1 - lat0) - 0.5
            cf = rel * raster.samples / geom.lon_span - 0.5
        r0 = max(0, int(np.floor(rf.min())) - 2 * k)
        c0 = max(0, int(np.floor(cf.min())) - 2 * k)
        r0, c0 = r0 - r0 % k, c0 - c0 % k
        r1 = min(raster.lines, int(np.ceil(rf.max())) + 2 * k + 1)
        c1 = min(raster.samples, int(np.ceil(cf.max())) + 2 * k + 1)
        win = block_mean(height_m(read_physical(raster, sentinels, slice(r0, r1), slice(c0, c1)), row["unit"]), k)
        z = map_coordinates(win, [(rf - r0 + 0.5) / k - 0.5, (cf - c0 + 0.5) / k - 0.5],
                            order=1, cval=np.nan, prefilter=False)
        return z, row["product_id"]
    return None, None


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def terrain_features(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """(4, H, W): slope east, slope north, steepness, local relief."""
    gy_rows, gx = np.gradient(z, dy, dx)
    gn = -gy_rows
    relief = z - uniform_filter(z, RELIEF_WINDOW)
    return np.stack([gx, gn, np.hypot(gx, gn), relief / max(dx, dy)])


def measure(image: np.ndarray, z: np.ndarray, dx: float, dy: float, smooth: float,
            max_nan: float = MAX_NAN) -> dict[str, float] | None:
    """Shift of the image relative to the DEM, from the terrain-predicted image."""
    if np.isnan(image).mean() > max_nan or np.isnan(z).mean() > max_nan:
        return None
    lim = image.shape[0] / 4
    image = np.where(np.isfinite(image), image, np.nanmedian(image))
    z = np.where(np.isfinite(z), z, np.nanmedian(z))
    feats = terrain_features(z, dx, dy)
    if smooth:
        image = gaussian_filter(image, smooth)
        feats = np.stack([gaussian_filter(f, smooth) for f in feats])
    # The fit on misaligned features is biased towards a smaller shift, so the estimate is
    # iterated: the features are moved by the current shift (Fourier shift), refitted, and the
    # residual shift is added, until it is below 0.01 px.
    ky, kx = np.fft.fftfreq(image.shape[0])[:, None], np.fft.fftfreq(image.shape[1])[None, :]
    spec = np.fft.fft2(feats, axes=(1, 2))
    sy = sx = 0.0
    for _ in range(ITERATIONS):
        moved = np.real(np.fft.ifft2(spec * np.exp(-2j * np.pi * (ky * sy + kx * sx)), axes=(1, 2)))
        a = np.column_stack([f.ravel() for f in moved] + [np.ones(image.size)])
        coef, *_ = np.linalg.lstsq(a, image.ravel(), rcond=None)
        model = (a @ coef).reshape(image.shape)
        dy, dx, peak = phase_shift(model, image)
        sy, sx = sy + dy, sx + dx
        if abs(dy) < 0.01 and abs(dx) < 0.01 or abs(sy) > lim or abs(sx) > lim:
            break
    fit_r = float(np.corrcoef(model.ravel(), image.ravel())[0, 1]) if model.std() > 0 else 0.0
    return dict(fit_r=fit_r, peak=peak, shift_e_px=sx, shift_n_px=-sy,
                valid=bool(fit_r >= FIT_MIN and peak >= PEAK_MIN and abs(sx) < lim and abs(sy) < lim))


def windows(row, geom, t: Target):
    """Start (r0, c0) of the windows of an image, on a lat/lon grid of centres."""
    lines, samples = int(row["lines"]), int(row["samples"])
    if geom.polar:
        north = geom.max_lat > 0
        for alat in np.arange(max(t.min_abs_lat, geom.polar_min_abs_lat) + t.lat_step / 2, 90.0, t.lat_step):
            for lon in np.arange(t.lon_step / 2, 360.0, t.lon_step):
                r, c = geom.polar_index(np.array([alat if north else -alat]), np.array([lon]))
                yield int(r[0]) - t.win // 2, int(c[0]) - t.win // 2
        return
    lat0, lat1 = float(row["bbox_min_lat_deg"]), float(row["bbox_max_lat_deg"])
    for lat in np.arange(-60 + t.lat_step / 2, 60, t.lat_step):
        if not lat0 <= lat <= lat1:
            continue
        for lon in np.arange(t.lon_step / 2, 360, t.lon_step):
            rel = (lon - geom.west_lon) % 360.0
            if rel <= geom.lon_span:
                yield (int((lat1 - lat) * lines / (lat1 - lat0)) - t.win // 2,
                       int(rel * samples / geom.lon_span) - t.win // 2)


def local_axis(geom, lat: float, lon: float, dlat: float, dlon: float) -> np.ndarray:
    """Unit vector (d_row, d_col) of a polar map pointing to (lat + dlat, lon + dlon) from (lat, lon)."""
    r0, c0 = geom.polar_index(np.array([lat]), np.array([lon]))
    r1, c1 = geom.polar_index(np.array([lat + dlat]), np.array([lon + dlon]))
    v = np.array([r1[0] - r0[0], c1[0] - c0[0]])
    return v / np.linalg.norm(v)


def run_target(t: Target, inv: list[dict[str, str]], prod: Products) -> list[dict[str, object]]:
    rows = [r for r in inv if r["product_id"].startswith(t.pattern)]
    if t.label.startswith("LAMP"):
        rows = [r for r in rows if any(k in r["product_id"] for k in LAMP_LAYERS)]
    # DEM below 60 deg: SLDEM 512, else SLDEM 256 where 512 is missing locally (the two agree, B3)
    dems = ([r for r in inv if r["product_id"].startswith("SLDEM2015_512")]
            + [r for r in inv if r["product_id"].startswith("SLDEM2015_256")] if t.dem == "cyl"
            else [r for r in inv if r["product_id"].startswith("LDEM_45")])
    out = []
    for row in rows:
        raster, geom, sentinels = prod.get(row)
        pixel_m = float(row["native_resolution_m"])
        for r0, c0 in windows(row, geom, t):
            n = t.win
            if r0 < 0 or c0 < 0 or r0 + n > raster.lines or c0 + n > raster.samples:
                continue
            img = read_physical(raster, sentinels, slice(r0, r0 + n), slice(c0, c0 + n))
            if t.log10:
                img = np.log10(np.where(img > 0, img, np.nan))
            lat, lon = window_latlon(row, geom, r0, c0, n)
            z, dem_id = dem_sample(prod, dems, lat, lon, pixel_m)
            if z is None:
                continue
            lat_c = float(lat[n // 2, n // 2])
            dy = pixel_m
            dx = pixel_m * (1.0 if geom.polar else math.cos(math.radians(lat_c)))
            m = measure(img, z, dx, dy, t.smooth, t.max_nan)
            if m is None:
                continue
            if geom.polar:   # polar map axes are not east / north: project on the local directions
                la, lo = float(lat[n // 2, n // 2]), float(lon[n // 2, n // 2])
                d = np.array([-m["shift_n_px"], m["shift_e_px"]])          # (d_row, d_col) of the shift
                m["shift_e_px"], m["shift_n_px"] = (float(d @ local_axis(geom, la, lo, 0.0, 1e-3)),
                                                    float(d @ local_axis(geom, la, lo, 1e-3, 0.0)))
            out.append(dict(target=t.label, product_id=row["product_id"], dem=dem_id,
                            lat=lat_c, lon=float(lon[n // 2, n // 2]), pixel_m=pixel_m,
                            shift_e_m=m["shift_e_px"] * dx, shift_n_m=m["shift_n_px"] * dy, **m))
        print(f"  {row['product_id']}: {sum(o['product_id'] == row['product_id'] for o in out)} windows", flush=True)
    return out


def summary(rows: list[dict[str, object]]) -> None:
    for label, reading in dict.fromkeys((r["target"], r.get("reading", "")) for r in rows):
        rs = [r for r in rows if r["target"] == label and r.get("reading", "") == reading]
        label = f"{label} [{reading}]" if reading else label
        v = [r for r in rs if r["valid"]]
        fit = np.median([r["fit_r"] for r in rs])
        if not v:
            print(f"{label:40s} {0:4d}/{len(rs):4d} valid (median fit r {fit:.2f}): terrain not visible enough")
            continue
        e = np.array([r["shift_e_m"] for r in v])
        n = np.array([r["shift_n_m"] for r in v])
        print(f"{label:40s} {len(v):4d}/{len(rs):4d} valid (median fit r {fit:.2f})  shift E {np.median(e):+7.1f} m "
              f"(robust sigma {robust_sigma(e):6.1f})  N {np.median(n):+7.1f} m (robust sigma {robust_sigma(n):6.1f})"
              f"  = {np.median(e) / v[0]['pixel_m']:+.2f} / {np.median(n) / v[0]['pixel_m']:+.2f} px")


def selftest() -> int:
    """An image made from a DEM with an unknown linear shading and noise, shifted by a known amount."""
    from diag_seams import _synthetic_terrain
    rng = np.random.default_rng(5)
    ok = True
    for true_e, true_n, noise in ((0.0, 0.0, 0.2), (1.4, -0.6, 0.2), (-0.3, 2.2, 1.0)):
        z = 400.0 * _synthetic_terrain(rng, WIN, WIN)
        f = terrain_features(z, 100.0, 100.0)
        img = 0.6 * f[0] - 0.3 * f[1] + 0.1 * f[2]
        img = img / img.std() + rng.normal(0, noise, img.shape)
        ky, kx = np.fft.fftfreq(WIN)[:, None], np.fft.fftfreq(WIN)[None, :]
        img = np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * (kx * true_e + ky * (-true_n)))))
        m = measure(img, z, 100.0, 100.0, smooth=1.0 if noise > 0.5 else 0.0)
        good = abs(m["shift_e_px"] - true_e) < 0.15 and abs(m["shift_n_px"] - true_n) < 0.15 and m["valid"]
        ok &= good
        print(f"true E {true_e:+.2f} N {true_n:+.2f} px (noise {noise}) -> E {m['shift_e_px']:+.2f} "
              f"N {m['shift_n_px']:+.2f}  fit r {m['fit_r']:.2f}  {'OK' if good else 'FAIL'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    dataset_root = DEFAULT_ROOT
    parser = argparse.ArgumentParser(description="Co-registration of every modality with the DEM (B4.1).")
    parser.add_argument("--root", default=str(dataset_root))
    parser.add_argument("--out-dir", default=None, help="default: <root>/preprocessing/diagnostics/B4")
    parser.add_argument("--only", nargs="+", metavar="TARGET", help="e.g. 'Mini-RF S1' (print only)")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else diagnostics_dir(root, "B4")
    with (inventory_path(root)).open(encoding="utf-8-sig", newline="") as fh:
        inv = list(csv.DictReader(fh))
    rows = []
    shifts = dict(data_quality.PIXEL_SHIFTS)
    for t in TARGETS:
        if args.only and t.label not in args.only:
            continue
        corrected = any(pid.startswith(t.pattern) for pid in shifts)
        # products with a georeferencing correction (D15) are measured as delivered and after it
        for reading in (("as delivered", "corrected (D15)") if corrected else ("as delivered",)):
            data_quality.PIXEL_SHIFTS.clear()
            if reading != "as delivered":
                data_quality.PIXEL_SHIFTS.update(shifts)
            print(f"{t.label} ({reading})", flush=True)
            for r in run_target(t, inv, Products(root)):
                r["reading"] = reading
                rows.append(r)
    data_quality.PIXEL_SHIFTS.update(shifts)
    summary(rows)
    if args.only:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "coregistration.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in ((k, r.get(k, "")) for k in COLUMNS)})
    print(f"written {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
