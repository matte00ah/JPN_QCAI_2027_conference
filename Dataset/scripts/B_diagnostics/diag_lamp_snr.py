#!/usr/bin/env python3
"""LAMP signal-to-noise maps and validity threshold (plan step B1.5).

Every LAMP band (Lyman-alpha, long, on-band, off-band) and the off/on ratio
has an "error" map whose label says: "Each sample represents the
signal-to-noise (SNR) ratio".  This script measures, for each hemisphere
and band, what a pixel-level SNR threshold would do and how the noise
behaves when the pyramid averages the pixels.

Measurements (area-weighted, inside the polar footprint)
--------------------------------------------------------
1. Pixel SNR distribution: percentiles 10/50/90 and area with SNR >= 1, 2, 3.
2. Physical check of the SNR: albedo cannot be negative, so the fraction of
   negative albedo values in the bins SNR < 1, 1-2, >= 2 tells where the
   values are dominated by the noise.
3. Selection bias of a threshold: median albedo of all pixels vs median of
   the pixels kept by SNR >= 1.  A pixel threshold keeps the brightest
   pixels, so it biases the layer towards high values.
4. Noise after averaging: the native albedo is averaged in b x b blocks (b =
   level pixel / 240 m: L1 1 (L1 is an upsampling), L2 2, L3 7, L4 27,
   mean of the valid pixels if at least half are valid) and the fraction of
   negative values left is measured.  Albedo is positive, so this fraction
   is an unambiguous indicator of noise-dominated values that does not rely
   on the definition of the SNR maps (not documented beyond the label
   sentence).  A noise estimate from second differences was tried and
   discarded: it decreases from 1 to 2x2 blocks, i.e. the noise is
   correlated between neighbouring pixels (smoothed maps) and that
   estimator is not valid.
5. Off/on ratio: the stored ratio map (mean of pixel ratios once averaged)
   is compared with the ratio of the averaged off-band and on-band albedo
   (ratio of means) at each level: median and fraction of negative values.

Output: ``<root>/preprocessing/diagnostics/B1/lamp_snr.csv`` (one row per
hemisphere and band).

Usage
-----
    python Dataset/scripts/B_diagnostics/diag_lamp_snr.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1]   # Dataset/scripts: one folder per plan chapter
sys.path[:0] = [str(_SCRIPTS / "A_sources"), str(Path(__file__).resolve().parent)]

from data_quality import DEFAULT_ROOT, inventory_path, diagnostics_dir, open_image, read_physical  # noqa: E402

BANDS = ("lya", "long", "on", "off", "ratio")
LEVEL_BLOCKS = {"L1": 1, "L2": 2, "L3": 7, "L4": 27}   # level pixel / 240 m, rounded
SNR_BINS = ((0.0, 1.0), (1.0, 2.0), (2.0, np.inf))
COLUMNS = [
    "hemisphere", "band", "snr_p10", "snr_p50", "snr_p90",
    "area_snr_ge1", "area_snr_ge2", "area_snr_ge3",
    "neg_frac_snr_lt1", "neg_frac_snr_1_2", "neg_frac_snr_ge2",
    "median_all", "median_snr_ge1",
    *[f"neg_frac_{lv}" for lv in LEVEL_BLOCKS], *[f"median_{lv}" for lv in LEVEL_BLOCKS],
    *[f"ratio_of_means_median_{lv}" for lv in LEVEL_BLOCKS],
    *[f"ratio_of_means_neg_frac_{lv}" for lv in LEVEL_BLOCKS],
]


def load(row: dict[str, str], root: Path):
    raster, geom, sentinels = open_image(row, root)
    x = read_physical(raster, sentinels, slice(None), slice(None))
    r, c = np.indices(x.shape)
    abs_lat = geom._polar_lat(r + 0.5, c + 0.5)
    x[abs_lat < geom.polar_min_abs_lat - 1e-9] = np.nan
    weight = ((1.0 + np.sin(np.radians(abs_lat))) / 2.0) ** 2
    return x, weight


def wmedian(x: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cw, 0.5 * cw[-1])]) if x.size else float("nan")


def wpct(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cw, q * cw[-1])]) if x.size else float("nan")


def block_mean(x: np.ndarray, b: int | tuple[int, int]) -> np.ndarray:
    """Mask-aware mean of b x b blocks (or br x bc if b is a pair): NaN if < half the pixels are valid."""
    br, bc = (b, b) if isinstance(b, int) else b
    if br == bc == 1:
        return x
    h, w = (x.shape[0] // br) * br, (x.shape[1] // bc) * bc
    blocks = x[:h, :w].reshape(h // br, br, w // bc, bc)
    n = np.isfinite(blocks).sum(axis=(1, 3))
    with np.errstate(invalid="ignore"):
        m = np.nansum(blocks, axis=(1, 3)) / np.maximum(n, 1)
    return np.where(n >= 0.5 * br * bc, m, np.nan)


def level_stats(x: np.ndarray, w: np.ndarray, b: int) -> tuple[float, float]:
    """(area-weighted median, fraction of negative values) of the b x b block-averaged map."""
    m, wb = block_mean(x, b), block_mean(w, b)
    ok = np.isfinite(m)
    if ok.sum() < 100:
        return float("nan"), float("nan")
    return wmedian(m[ok], wb[ok]), float(wb[ok & (m < 0)].sum() / wb[ok].sum())


def main(argv: list[str] | None = None) -> int:
    dataset_root = DEFAULT_ROOT
    parser = argparse.ArgumentParser(description="LAMP SNR maps and validity threshold (B1.5).")
    parser.add_argument("--root", default=str(dataset_root))
    parser.add_argument("--out-dir", default=None, help="default: <root>/preprocessing/diagnostics/B1")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else diagnostics_dir(root, "B1")
    with (inventory_path(root)).open(encoding="utf-8-sig", newline="") as handle:
        inv = {r["product_id"]: r for r in csv.DictReader(handle)}

    records = []
    for hemi in ("80n", "80s"):
        for band in BANDS:
            value_id = f"LAMP_{hemi}_240mpp_{band}_albed_01"
            snr_id = f"LAMP_{hemi}_240mpp_{band}_error_01"
            if value_id not in inv or snr_id not in inv:
                continue
            a, w = load(inv[value_id], root)
            s, _ = load(inv[snr_id], root)
            ok = np.isfinite(a) & np.isfinite(s)
            av, sv, wv = a[ok], s[ok], w[ok]
            tot = wv.sum()
            rec = dict(hemisphere=hemi, band=band,
                       snr_p10=wpct(sv, wv, 0.10), snr_p50=wpct(sv, wv, 0.50), snr_p90=wpct(sv, wv, 0.90),
                       area_snr_ge1=wv[sv >= 1].sum() / tot, area_snr_ge2=wv[sv >= 2].sum() / tot,
                       area_snr_ge3=wv[sv >= 3].sum() / tot,
                       median_all=wmedian(av, wv), median_snr_ge1=wmedian(av[sv >= 1], wv[sv >= 1]))
            for (lo, hi), key in zip(SNR_BINS, ("neg_frac_snr_lt1", "neg_frac_snr_1_2", "neg_frac_snr_ge2")):
                m = (sv >= lo) & (sv < hi)
                rec[key] = wv[m & (av < 0)].sum() / wv[m].sum() if m.any() else float("nan")
            for lv, b in LEVEL_BLOCKS.items():
                rec[f"median_{lv}"], rec[f"neg_frac_{lv}"] = level_stats(a, w, b)
            if band == "ratio":
                off, _ = load(inv[f"LAMP_{hemi}_240mpp_off_albed_01"], root)
                on, _ = load(inv[f"LAMP_{hemi}_240mpp_on_albed_01"], root)
                for lv, b in LEVEL_BLOCKS.items():
                    mo, mn, wb = block_mean(off, b), block_mean(on, b), block_mean(w, b)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        r = np.where(np.isfinite(mo) & np.isfinite(mn) & (mn != 0), mo / mn, np.nan)
                    good = np.isfinite(r)
                    rec[f"ratio_of_means_median_{lv}"] = wmedian(r[good], wb[good])
                    rec[f"ratio_of_means_neg_frac_{lv}"] = float(wb[good & (r < 0)].sum() / wb[good].sum())
            records.append(rec)
            print(f"{hemi} {band:5s} SNR p50={rec['snr_p50']:.2f} area>=1/2={rec['area_snr_ge1']:.2f}/"
                  f"{rec['area_snr_ge2']:.2f} neg<1/1-2/>=2={rec['neg_frac_snr_lt1']:.2f}/"
                  f"{rec['neg_frac_snr_1_2']:.2f}/{rec['neg_frac_snr_ge2']:.2f} "
                  f"median all/kept={rec['median_all']:.4g}/{rec['median_snr_ge1']:.4g} "
                  "median " + " ".join(f"{lv}={rec[f'median_{lv}']:.3g}" for lv in LEVEL_BLOCKS)
                  + " neg " + " ".join(f"{lv}={rec[f'neg_frac_{lv}']:.2f}" for lv in LEVEL_BLOCKS), flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "lamp_snr.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for rec in records:
            writer.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in rec.items()})
    print(f"Output: {out_dir / 'lamp_snr.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
