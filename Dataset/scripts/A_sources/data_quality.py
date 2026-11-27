#!/usr/bin/env python3
"""Compute a per-image quality score for every product of the dataset.

An image of the dataset is a PDS3 label (``.lbl``, or the label embedded at
the start of LROC ``.IMG`` files) plus its raster (``.img``).  The list of
images is read from ``data_inventory.csv`` (columns ``label_file`` and
``data_file``) and the ``quality_score`` column of that file is overwritten
with the score computed here.

Score
-----
The score is a weighted sum of five sub-scores, each in [0, 1] and anchored
to physical/absolute references (never to dataset-wide min-max), so that
images of different subsets and missions are directly comparable::

    Q = 0.30 * S_res + 0.30 * S_snr + 0.20 * S_stripe
        + 0.10 * S_dropout + 0.10 * S_clipquant

S_res (effective resolution)
    Mean of two terms:

    * spectral resolution ratio r = f_cut / f_Nyquist, from the 1-D power
      spectral density along each image axis (worst axis kept).  f_cut is
      the first frequency where the PSD either reaches twice the white-noise
      floor (detail no longer resolved above the noise / quantisation) or
      falls one decade below the power law fitted on mid frequencies
      (wavelengths 16-100 px; roll-off of noise-free interpolated or
      blurred products).  r ~ 1 means the image really carries detail up to
      the declared pixel size; r << 1 means an over-sampled, interpolated
      or blurred product.
    * 1 - fraction of "interpolated" 8x8 patches, i.e. patches whose
      residual after a quadratic fit is much smaller (< 0.25x) than the
      sensor noise of the image.  Real smooth terrain (maria) still carries
      sensor noise and is not flagged; linearly/bicubically interpolated or
      nearest-neighbour filled patches are.

S_snr (noise)
    Noise sigma_n is estimated robustly from second differences
    (MAD / 0.6745 / sqrt(6)) taken at the effective sampling step
    round(1 / r), so that over-sampled or interpolated images (whose
    pixel-to-pixel differences are artificially smooth) do not get a
    spuriously high SNR; the signal sigma_s
    as the robust spread (IQR / 1.349) of the plane-detrended tile.  The
    tile SNR in dB is mapped linearly: <= -10 dB -> 0, >= 30 dB -> 1.
    Radar power and power-ratio products (SC, OC, S1, CPR, LPR) are analysed
    in log10 domain because speckle is multiplicative.

S_stripe (striping / orbit seams)
    Fraction of high-frequency spectral energy lying on the narrow wedge
    expected for artefacts aligned with the orbit tracks (and the cross-track
    direction), divided by the fraction expected for an isotropic texture.
    Ratio R <= 2 -> 1, R >= 20 -> 0 (log scale in between).

S_dropout (drop-outs and invalid values)
    Fraction of footprint pixels holding valid-looking values outside the
    hard physical range of the quantity (e.g. slope > 90 deg, visibility > 1,
    negative radar power).  0 % -> 1, >= 2 % -> 0.

    No-data pixels are NOT penalised, whatever their size: in these gridded
    products even small isolated holes are bins never observed (the LAMP hole
    pattern is identical in the albedo, brightness and error maps of a band;
    the LDSM_85N_40M holes grow with the distance between LOLA tracks), i.e.
    coverage, not image degradation.  Small holes (no-data components of at
    most ``HOLE_MAX_PX`` pixels not touching the tile border nor the
    footprint edge) are still reported as ``small_holes_per_mpx`` for
    information only.

S_clipquant (clipping and quantisation)
    Product of (1 - excess fraction of pixels stuck at the min/max value,
    5 % -> 0) and (1 - fraction of empty histogram bins in the central
    5-95 % range, a posterisation indicator).  For ``fraction`` products the
    physical bounds 0 and 1 are legitimate and not treated as clipping.

If a sub-score cannot be computed (e.g. no tile with enough valid pixels for
the spectral analysis) the remaining weights are renormalised and the case is
reported in the metrics file.  Binary masks (``value_kind == binary_mask``,
LOLA PSR maps) are excluded: their ``quality_score`` is left empty.

Polar vs cylindrical images
---------------------------
The sub-scores are the same; only the geometry used to compute them differs:

* footprint: polar stereographic images use the latitude limits of the label
  (corners outside the minimum latitude are ignored); cylindrical images use
  the whole raster;
* area weights: tile statistics are weighted by the true ground area of the
  pixels, cos(lat) in cylindrical, ((1 + sin|lat|) / 2)^2 in polar
  stereographic (true scale at the pole);
* resolution / noise direction: in cylindrical images the longitude
  direction is over-sampled by 1/cos(lat), so noise is computed along the
  latitude direction (axis 0) only, and the longitude-axis resolution ratio
  is divided by cos(lat) and used only for |lat| <= 60 deg; in polar
  stereographic images (conformal) both axes are used and the resolution
  ratio is multiplied by the local scale factor k = 2 / (1 + sin|lat|);
* striping direction: orbit tracks are N-S (image columns) in cylindrical
  maps and radial (towards the pole) in polar maps.

Large images are analysed on randomly sampled tiles (seed derived from the
product id, so results are reproducible) and statistics are aggregated with
area-weighted medians.

Undeclared no-data
------------------
Besides the label missing constant and non-finite values, values with
|v| >= 1e30 (ISIS special pixels, used by LROC and Mini-RF) and a raw value
equal to 0 or to the data-type extreme covering more than 1 % of the image
(e.g. the 0 fill of LDSM_85N_40M) are treated as no-data.

Outputs
-------
In a full run (no ``--only``) every finished image is appended at once to
``data_quality_metrics.csv`` (checkpoint), so an interrupted run loses nothing
and is continued with ``--resume`` (images already in the file are skipped,
those in error are analysed again).  At the end the metrics file is rewritten
in inventory order and the ``quality_score`` column of the inventory is
updated once, from all the results (both files are replaced atomically).

Usage
-----
    python Dataset/scripts/A_sources/data_quality.py
    python Dataset/scripts/A_sources/data_quality.py --jobs 4
    python Dataset/scripts/A_sources/data_quality.py --jobs 4 --resume    # continue an interrupted run
    python Dataset/scripts/A_sources/data_quality.py --only LDSM_16 LAMP_80n_240mpp_on_albed_01 --dry-run

Requires numpy and scipy.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import struct
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_inventory import (  # noqa: E402  (path set up above)
    DEFAULT_ROOT,
    HEX_VALUE_RE,
    PdsLabel,
    find,
    inventory_path,
    metrics_path,
    preprocessing_dir,
    sources_dir,
    lookup,
    number,
    read_embedded_label,
    text_value,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WEIGHTS = {
    "S_res": 0.30,
    "S_snr": 0.30,
    "S_stripe": 0.20,
    "S_dropout": 0.10,
    "S_clipquant": 0.10,
}

TILE = 512                  # tile side in pixels
MAX_TEXTURE_TILES = 32      # tiles used for spectral / noise / stripe metrics
MAX_READ_TILES = 128        # max tiles read per image
MIN_TEXTURE_TILES = 4       # below this, accept tiles with more no-data
TEXTURE_VALID_STRICT = 0.95
TEXTURE_VALID_RELAXED = 0.60
HOLE_MAX_PX = 25            # max area of a "small hole" (reported only, not scored)
SENTINEL_MIN_FRACTION = 0.01
SPECIAL_ABS = 1e30          # ISIS special pixels
POOL_PER_TILE = 20_000      # values per tile kept for histogram metrics

# PSD fit band (cycles/pixel) and drop threshold (decades)
PSD_FIT_BAND = (0.01, 0.06)
PSD_DROP_DECADES = 1.0
PATCH = 8
INTERP_RESID_FACTOR = 0.25

SNR_DB_ZERO, SNR_DB_ONE = -10.0, 30.0
STRIPE_WEDGE_DEG = 3.0
STRIPE_KMIN = 16 / TILE     # ignore large scales in stripe analysis
STRIPE_R_OK, STRIPE_R_BAD = 2.0, 20.0
DROPOUT_BAD_FRACTION = 0.02
CLIP_BAD_FRACTION = 0.05

# Quantities analysed in log10 domain (multiplicative speckle).
LOG_QUANTITIES = {
    "same_sense_circular_power", "opposite_sense_circular_power", "stokes_s1",
    "circular_polarization_ratio", "linear_polarization_ratio",
}

# Hard physical ranges (on physical values).  Only bounds guaranteed by the
# processing are listed: small negative values of noisy reflectance/albedo
# are noise, not invalid data, and are handled by the SNR sub-score.
HARD_RANGES = {
    "slope": (0.0, 90.0),
    "roughness": (0.0, math.inf),
    "solar_visibility": (0.0, 1.0),
    "earth_visibility": (0.0, 1.0),
    "degree_of_polarization": (0.0, 1.0),
    "degree_of_linear_polarization": (0.0, 1.0),
    "degree_of_circular_polarization": (-1.0, 1.0),
    "circular_polarization_ratio": (0.0, math.inf),
    "linear_polarization_ratio": (0.0, math.inf),
    "same_sense_circular_power": (0.0, math.inf),
    "opposite_sense_circular_power": (0.0, math.inf),
    "stokes_s1": (0.0, math.inf),
    "uv_snr": (0.0, math.inf),
}
RANGE_TOL = 1e-6


def out_of_range(phys: np.ndarray, quantity: str) -> np.ndarray:
    """Pixels outside the hard physical range of ``quantity`` (False for NaN and for
    quantities without a range).  Shared by the quality score and the cleaning (C1.3)."""
    bounds = HARD_RANGES.get(quantity)
    if bounds is None:
        return np.zeros(np.shape(phys), dtype=bool)
    lo, hi = bounds
    tol = RANGE_TOL * max(1.0, abs(lo), abs(hi) if math.isfinite(hi) else 1.0)
    with np.errstate(invalid="ignore"):
        return (phys < lo - tol) | (phys > hi + tol)

EXCLUDED_VALUE_KINDS = {"binary_mask"}

METRIC_COLUMNS = [
    "product_id", "mission", "subset", "projection_class", "status",
    "tiles_read", "tiles_texture", "sentinels",
    "resolution_ratio", "interp_patch_fraction", "snr_db", "noise_sigma",
    "stripe_ratio", "dropout_fraction", "small_holes_per_mpx", "out_of_range_fraction",
    "clip_excess_fraction", "hist_gap_fraction",
    "S_res", "S_snr", "S_stripe", "S_dropout", "S_clipquant", "quality_score",
]

# ---------------------------------------------------------------------------
# Raster access
# ---------------------------------------------------------------------------

SAMPLE_TYPES = {
    "PC_REAL": "<f", "REAL": ">f", "IEEE_REAL": ">f", "MAC_REAL": ">f", "SUN_REAL": ">f",
    "LSB_INTEGER": "<i", "PC_INTEGER": "<i", "VAX_INTEGER": "<i",
    "MSB_INTEGER": ">i", "INTEGER": ">i", "SUN_INTEGER": ">i", "MAC_INTEGER": ">i",
    "LSB_UNSIGNED_INTEGER": "<u", "PC_UNSIGNED_INTEGER": "<u", "VAX_UNSIGNED_INTEGER": "<u",
    "MSB_UNSIGNED_INTEGER": ">u", "UNSIGNED_INTEGER": ">u", "SUN_UNSIGNED_INTEGER": ">u",
    "MAC_UNSIGNED_INTEGER": ">u",
}


@dataclass
class Raster:
    data: np.memmap
    lines: int
    samples: int
    scale: float
    offset: float
    is_integer: bool
    missing: list[float]
    dtype: np.dtype
    # integer georeferencing correction (rows, cols): value at (r, c) = stored value at (r + dr, c + dc)
    shift_rc: tuple[int, int] = (0, 0)


# Georeferencing corrections measured in B4.1 and decided in D15: in these WAC_GLOBAL tiles every
# structure is stored one row south and one column east of its position given by the label, so the
# corrected value at (r, c) is the stored value at (r + 1, c + 1); the last row and column become
# no-data.  To be re-measured on the remote machine with the full SLDEM 512 (plan, D15).
PIXEL_SHIFTS: dict[str, tuple[int, int]] = {
    f"WAC_GLOBAL_E300{h}{lon}_100M": (1, 1) for h in "NS" for lon in ("0450", "2250", "3150")
}


def _image_offset(label: PdsLabel, record_bytes: int) -> int:
    """Byte offset of the image from the ``^IMAGE`` pointer."""

    value = find(label, "^IMAGE")
    if value is None:
        return 0
    text = value.text.strip()
    if text.startswith("(") and text.endswith(")"):
        parts = [p.strip().strip('"') for p in text[1:-1].split(",")]
        text = parts[1] if len(parts) > 1 else "1"
    if value.unit and value.unit.upper().startswith("BYTE"):
        return int(float(text)) - 1
    try:
        return (int(float(text)) - 1) * record_bytes
    except ValueError:
        return 0  # pointer to a detached file: image starts at byte 0


def _missing_values(label: PdsLabel, sample_type: str, bits: int) -> list[float]:
    out: list[float] = []
    for key in ("MISSING_CONSTANT", "CORE_NULL", "NULL_VALUE", "MISSING_VALUE",
                "NODATA_VALUE", "INVALID_CONSTANT"):
        value = lookup(label.image, key) or lookup(label.root, key)
        text = text_value(value)
        if text is None:
            continue
        match = HEX_VALUE_RE.fullmatch(text)
        if match and "REAL" in sample_type and bits in (32, 64):
            raw = int(match.group(2), int(match.group(1)))
            fmt = ">f" if bits == 32 else ">d"
            out.append(struct.unpack(fmt, raw.to_bytes(bits // 8, "big"))[0])
            continue
        num = number(value)
        if num is not None:
            out.append(float(num))
    return out


def _image_key(label: PdsLabel, key: str):
    """Keyword from the IMAGE object, falling back to the label root."""
    return lookup(label.image, key) or lookup(label.root, key)


def open_raster(label: PdsLabel, image_path: Path) -> Raster:
    image = label.image
    sample_type = (text_value(_image_key(label, "SAMPLE_TYPE")) or "").upper()
    bits = int(number(_image_key(label, "SAMPLE_BITS")) or 0)
    lines = int(number(_image_key(label, "LINES")))
    samples = int(number(_image_key(label, "LINE_SAMPLES")))
    bands = int(number(lookup(image, "BANDS")) or 1)
    prefix = int(number(lookup(image, "LINE_PREFIX_BYTES")) or 0)
    suffix = int(number(lookup(image, "LINE_SUFFIX_BYTES")) or 0)
    if bands != 1 or prefix or suffix:
        raise ValueError(f"unsupported layout (bands={bands}, prefix={prefix}, suffix={suffix})")
    if sample_type not in SAMPLE_TYPES:
        raise ValueError(f"unsupported SAMPLE_TYPE {sample_type!r}")

    code = SAMPLE_TYPES[sample_type]  # e.g. "<f": byte order + kind
    nbytes = bits // 8
    if nbytes == 1:
        dtype = np.dtype("u1" if code[1] == "u" else "i1")  # no byte order for 8-bit
    else:
        dtype = np.dtype(f"{code}{nbytes}")
    record_bytes = int(number(find(label, "RECORD_BYTES")) or samples * nbytes)
    offset_bytes = _image_offset(label, record_bytes)
    data = np.memmap(image_path, dtype=dtype, mode="r", offset=offset_bytes, shape=(lines, samples))

    scale = number(_image_key(label, "SCALING_FACTOR"))
    offset = number(_image_key(label, "OFFSET"))
    return Raster(
        data=data, lines=lines, samples=samples,
        scale=float(scale) if scale is not None else 1.0,
        offset=float(offset) if offset is not None else 0.0,
        is_integer=dtype.kind in "iu",
        missing=_missing_values(label, sample_type, bits),
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass
class Geometry:
    polar: bool
    min_lat: float
    max_lat: float
    res_m: float
    radius_m: float = 1737400.0
    line_off: float = 0.0
    samp_off: float = 0.0
    # cylindrical maps: longitude of the west edge of the first column and span, from the
    # label (the inventory bbox is normalised to 0-360 and does not say where the image starts:
    # the Mini-RF mosaics start at -180)
    west_lon: float = 0.0
    lon_span: float = 360.0

    def column_lon(self, cols: np.ndarray, samples: int) -> np.ndarray:
        """Longitude (0-360) of cylindrical-map pixel centres given by column indices."""
        return (self.west_lon + (cols + 0.5) * self.lon_span / samples) % 360.0

    @property
    def polar_min_abs_lat(self) -> float:
        """Latitude limit of a polar footprint (|lat| grows towards the pole)."""
        return min(abs(self.min_lat), abs(self.max_lat))

    def tile_center_lat(self, r0: int, c0: int, h: int, w: int, lines: int) -> float:
        rc, cc = r0 + h / 2.0, c0 + w / 2.0
        if self.polar:
            return float(self._polar_lat(np.array([rc]), np.array([cc]))[0])
        return self.max_lat - rc * (self.max_lat - self.min_lat) / lines

    def _polar_lat(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        rho = np.hypot(rows - self.line_off, cols - self.samp_off) * self.res_m
        return 90.0 - 2.0 * np.degrees(np.arctan(rho / (2.0 * self.radius_m)))  # |lat|

    def polar_index(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(row, col) array indices (pixel centres at integers) of (lat, lon) in a polar map.

        PDS polar stereographic, centre longitude 0, east-positive longitudes:
        longitude 0 points down from the north pole and up from the south pole
        (checked on WAC_GLOBAL: the detail continues across the 60 deg junction).
        """
        rho = 2.0 * self.radius_m * np.tan(np.radians(90.0 - np.abs(lat)) / 2.0) / self.res_m
        lam = np.radians(lon)
        north = self.max_lat > 0
        rows = self.line_off + (1.0 if north else -1.0) * rho * np.cos(lam) - 0.5
        cols = self.samp_off + rho * np.sin(lam) - 0.5
        return rows, cols

    def polar_latlon(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(lat, lon) of polar-map pixels given by array indices (pixel centres at integers);
        inverse of ``polar_index``."""
        north = self.max_lat > 0
        dr = rows + 0.5 - self.line_off
        dc = cols + 0.5 - self.samp_off
        lon = np.degrees(np.arctan2(dc, dr if north else -dr)) % 360.0
        lat = self._polar_lat(rows + 0.5, cols + 0.5)
        return (lat if north else -lat), lon

    def footprint(self, r0: int, c0: int, h: int, w: int) -> np.ndarray:
        if not self.polar:
            return np.ones((h, w), dtype=bool)
        rows = np.arange(r0, r0 + h, dtype=np.float64)[:, None] + 0.5
        cols = np.arange(c0, c0 + w, dtype=np.float64)[None, :] + 0.5
        return self._polar_lat(rows, cols) >= self.polar_min_abs_lat - 1e-9

    def area_weight(self, abs_lat: float) -> float:
        if self.polar:
            return ((1.0 + math.sin(math.radians(abs_lat))) / 2.0) ** 2
        return max(math.cos(math.radians(abs_lat)), 1e-3)

    def scale_factor(self, abs_lat: float) -> float:
        """Map scale factor along the analysed axes (1 along meridians in cylindrical)."""
        if self.polar:
            return 2.0 / (1.0 + math.sin(math.radians(abs_lat)))
        return 1.0

    def track_angle(self, r0: int, c0: int, h: int, w: int) -> float:
        """Direction (radians, image coords row/col) of the orbit tracks."""
        if not self.polar:
            return 0.0  # along axis 0 (N-S)
        dr = r0 + h / 2.0 - self.line_off
        dc = c0 + w / 2.0 - self.samp_off
        return math.atan2(dc, dr)


def build_geometry(label: PdsLabel, row: dict[str, str]) -> Geometry:
    proj = label.projection
    projection = (row.get("projection") or "").upper()
    polar = "POLAR" in projection or "STEREO" in projection
    min_lat = float(row["bbox_min_lat_deg"])
    max_lat = float(row["bbox_max_lat_deg"])
    res_m = float(row["native_resolution_m"])
    radius = number(lookup(proj, "A_AXIS_RADIUS"))
    radius_m = float(radius) * 1000.0 if radius is not None and radius < 1e5 else float(radius or 1737400.0)
    geom = Geometry(polar=polar, min_lat=min_lat, max_lat=max_lat, res_m=res_m, radius_m=radius_m)
    if polar:
        # PDS projection offsets are the (line, sample) of the pole in
        # 1-based pixel-centre coordinates; convert to 0-based edges.
        geom.line_off = float(number(lookup(proj, "LINE_PROJECTION_OFFSET")) or 0.0) + 0.5
        geom.samp_off = float(number(lookup(proj, "SAMPLE_PROJECTION_OFFSET")) or 0.0) + 0.5
    else:
        west, east = number(lookup(proj, "WESTERNMOST_LONGITUDE")), number(lookup(proj, "EASTERNMOST_LONGITUDE"))
        if west is not None and east is not None:
            geom.west_lon, geom.lon_span = float(west), float(east) - float(west)
        else:
            geom.west_lon = float(row["bbox_min_lon_deg"])
            geom.lon_span = float(row["bbox_max_lon_deg"]) - geom.west_lon
    return geom


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------


def detect_sentinels(raster: Raster, fraction_kind: bool) -> list[float]:
    """Undeclared fill values: raw 0 or dtype extremes covering > 1 % of the image."""

    step_r = max(1, raster.lines // 400)
    step_c = max(1, raster.samples // 800)
    sub = np.asarray(raster.data[::step_r, ::step_c]).astype(np.float64).ravel()
    sub = sub[np.isfinite(sub) & (np.abs(sub) < SPECIAL_ABS)]
    if sub.size == 0:
        return []
    candidates = [0.0]
    if raster.is_integer:
        info = np.iinfo(raster.dtype)
        candidates += [float(info.min), float(info.max)]
    found = []
    for cand in candidates:
        if any(cand == m for m in raster.missing):
            continue
        frac = float(np.mean(sub == cand))
        if frac <= SENTINEL_MIN_FRACTION:
            continue
        physical = cand * raster.scale + raster.offset
        if fraction_kind and (abs(physical) < 1e-9 or abs(physical - 1.0) < 1e-9):
            continue  # legitimate physical bound (never/always visible)
        found.append(cand)
    return found


def nodata_mask(raw: np.ndarray, raster: Raster, sentinels: list[float]) -> np.ndarray:
    raw64 = raw.astype(np.float64, copy=False)
    bad = ~np.isfinite(raw64) | (np.abs(raw64) >= SPECIAL_ABS)
    for m in [*raster.missing, *sentinels]:
        if abs(m) >= SPECIAL_ABS:
            continue  # already covered by the special-pixel test
        if raster.is_integer or m == 0.0:
            bad |= raw64 == m
        else:
            bad |= np.isclose(raw64, m, rtol=1e-7, atol=0.0)
    return bad


def diagnostics_dir(root: Path, step: str) -> Path:
    """Output folder of the phase-B diagnostics of one macro-step (e.g. "B1", "B2")."""
    return preprocessing_dir(root, "diagnostics", step)


def open_image(row: dict[str, str], root: Path) -> tuple[Raster, Geometry, list[float]]:
    """Open one inventory image: raster (memory-mapped), geometry and undeclared fill values.

    Handles both detached labels (``.lbl``) and labels embedded in the ``.IMG``
    file (LROC).  Shared by all the scripts that read the products.
    """
    label_path = root / row["label_file"]
    text = read_embedded_label(label_path) if label_path.suffix.lower() == ".img" else None
    label = PdsLabel(label_path, text=text)
    raster = open_raster(label, root / row["data_file"])
    geom = build_geometry(label, row)
    if is_cleaned(label):
        # chapter-C output: the D15 shift is already in the pixels and no-data is NaN, so
        # nothing is moved and no value (e.g. a legitimate 0 of a mask) is taken as a fill value
        return raster, geom, []
    raster.shift_rc = PIXEL_SHIFTS.get(row["product_id"], (0, 0))
    sentinels = detect_sentinels(raster, row.get("value_kind") == "fraction")
    return raster, geom, sentinels


CLEANED_KEY = "CLEANING_STEPS"   # root keyword written in the labels of the cleaned images (chapter C)


def is_cleaned(label: PdsLabel) -> bool:
    """True for an image written by the chapter-C cleaning (float32 physical units, NaN no-data)."""
    return lookup(label.root, CLEANED_KEY) is not None


def read_physical(raster: Raster, sentinels: list[float], rows: slice, cols: slice) -> np.ndarray:
    """Window of the image in physical units (float64), no-data as NaN.

    The georeferencing correction ``raster.shift_rc`` (D15) is applied here, so that every
    script reads the corrected image; pixels moved in from outside the image are NaN.
    """
    if raster.shift_rc == (0, 0):
        raw = np.asarray(raster.data[rows, cols])
        phys = raw.astype(np.float64) * raster.scale + raster.offset
        phys[nodata_mask(raw, raster, sentinels)] = np.nan
        return phys
    r_idx = np.arange(raster.lines)[rows] + raster.shift_rc[0]
    c_idx = np.arange(raster.samples)[cols] + raster.shift_rc[1]
    ok_r = (r_idx >= 0) & (r_idx < raster.lines)
    ok_c = (c_idx >= 0) & (c_idx < raster.samples)
    phys = np.full((r_idx.size, c_idx.size), np.nan)
    if ok_r.any() and ok_c.any():
        r_sel, c_sel = r_idx[ok_r], c_idx[ok_c]
        step_r = int(r_sel[1] - r_sel[0]) if r_sel.size > 1 else 1
        step_c = int(c_sel[1] - c_sel[0]) if c_sel.size > 1 else 1
        raw = np.asarray(raster.data[r_sel[0]:r_sel[-1] + 1:step_r, c_sel[0]:c_sel[-1] + 1:step_c])
        sub = raw.astype(np.float64) * raster.scale + raster.offset
        sub[nodata_mask(raw, raster, sentinels)] = np.nan
        phys[np.ix_(ok_r, ok_c)] = sub
    return phys


def sample_physical(raster: Raster, sentinels: list[float], rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Bilinear samples at fractional (row, col) indices, read pixel by pixel from the memory map.

    No-data neighbours are left out and the weights of the valid ones
    renormalised (so the samples next to a footprint edge stay valid); NaN if
    no neighbour is valid or the point is outside the image.  Only the needed
    pixels are read, so it works on images too large to load whole.
    """
    r0, c0 = np.floor(rows).astype(np.int64), np.floor(cols).astype(np.int64)
    fr, fc = rows - r0, cols - c0
    r0, c0 = r0 + raster.shift_rc[0], c0 + raster.shift_rc[1]      # georeferencing correction (D15)
    inside = (r0 >= 0) & (c0 >= 0) & (r0 + 1 < raster.lines) & (c0 + 1 < raster.samples)
    out = np.full(rows.shape, np.nan)
    ri, ci = r0[inside], c0[inside]
    acc, wsum = np.zeros(ri.shape), np.zeros(ri.shape)
    for dr, dc, wgt in ((0, 0, (1 - fr) * (1 - fc)), (0, 1, (1 - fr) * fc), (1, 0, fr * (1 - fc)), (1, 1, fr * fc)):
        raw = np.asarray(raster.data[ri + dr, ci + dc])
        phys = raw.astype(np.float64) * raster.scale + raster.offset
        good = ~nodata_mask(raw, raster, sentinels)
        acc += np.where(good, wgt[inside] * phys, 0.0)
        wsum += np.where(good, wgt[inside], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[inside] = np.where(wsum > 0, acc / wsum, np.nan)
    return out


# ---------------------------------------------------------------------------
# Per-tile metrics
# ---------------------------------------------------------------------------


def detrend_plane(tile: np.ndarray) -> np.ndarray:
    h, w = tile.shape
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.column_stack([np.ones(h * w), yy.ravel(), xx.ravel()])
    coef, *_ = np.linalg.lstsq(a, tile.ravel(), rcond=None)
    return tile - (a @ coef).reshape(h, w)


def robust_sigma(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def noise_sigma(tile: np.ndarray, valid: np.ndarray, axes: tuple[int, ...]) -> float:
    """Robust white-noise sigma from second differences along *axes*."""
    diffs = []
    for ax in axes:
        t = np.moveaxis(tile, ax, 0)
        v = np.moveaxis(valid, ax, 0)
        d2 = t[:-2] - 2.0 * t[1:-1] + t[2:]
        ok = v[:-2] & v[1:-1] & v[2:]
        diffs.append(d2[ok])
    d = np.concatenate(diffs) if diffs else np.empty(0)
    if d.size < 100:
        return float("nan")
    return robust_sigma(d) / math.sqrt(6.0)


def psd_resolution_ratio(tile: np.ndarray, axis: int, sigma_floor: float) -> float:
    """f_cut / Nyquist from the 1-D PSD along *axis* (cycles/pixel).

    f_cut is the lowest frequency where either
    * the PSD reaches twice the white-noise floor (sigma_floor^2 * sum(w^2)),
      i.e. the signal is no longer resolved above the noise, or
    * the PSD falls one decade below the power law fitted on mid frequencies
      (roll-off of interpolated / blurred products without noise).
    """
    t = np.moveaxis(tile, axis, 0)
    n = t.shape[0]
    win = np.hanning(n)
    t = t - t.mean(axis=0, keepdims=True)
    psd = (np.abs(np.fft.rfft(t * win[:, None], axis=0)) ** 2).mean(axis=1)
    floor = (sigma_floor ** 2) * float((win ** 2).sum()) if math.isfinite(sigma_floor) else 0.0
    f = np.fft.rfftfreq(n)
    keep = f > 0
    f, psd = f[keep], psd[keep]
    if np.all(psd <= 0):
        return float("nan")
    psd = np.maximum(psd, psd[psd > 0].min() * 1e-6)  # avoid log10(0)

    # log-spaced smoothing
    edges = np.logspace(np.log10(f[0]), np.log10(0.5), 40)
    centers, logp = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (f >= lo) & (f < hi) if hi < 0.5 else (f >= lo) & (f <= hi)
        if sel.any():
            centers.append(math.sqrt(lo * hi))
            logp.append(np.log10(psd[sel].mean()))
    centers = np.array(centers)
    logp = np.array(logp)
    fit = (centers >= PSD_FIT_BAND[0]) & (centers <= PSD_FIT_BAND[1])
    if fit.sum() < 3:
        return float("nan")
    slope, intercept = np.polyfit(np.log10(centers[fit]), logp[fit], 1)
    model = intercept + slope * np.log10(centers)
    beyond = centers > PSD_FIT_BAND[1]
    below = beyond & (logp < model - PSD_DROP_DECADES)
    if floor > 0:
        below |= logp <= math.log10(2.0 * floor)
    if not below.any():
        return 1.0
    f_cut = centers[np.argmax(below)]
    return float(min(1.0, f_cut / 0.5))


_PATCH_PROJ = None


def _patch_projector() -> np.ndarray:
    global _PATCH_PROJ
    if _PATCH_PROJ is None:
        yy, xx = np.mgrid[0:PATCH, 0:PATCH]
        y, x = yy.ravel() - PATCH / 2, xx.ravel() - PATCH / 2
        a = np.column_stack([np.ones_like(y), y, x, y * y, x * x, x * y]).astype(np.float64)
        _PATCH_PROJ = np.eye(PATCH * PATCH) - a @ np.linalg.pinv(a)
    return _PATCH_PROJ


def patch_residuals(tile: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Residual std after a quadratic fit for every fully-valid 8x8 patch."""
    h, w = (tile.shape[0] // PATCH) * PATCH, (tile.shape[1] // PATCH) * PATCH
    t = tile[:h, :w].reshape(h // PATCH, PATCH, w // PATCH, PATCH).swapaxes(1, 2).reshape(-1, PATCH * PATCH)
    v = valid[:h, :w].reshape(h // PATCH, PATCH, w // PATCH, PATCH).swapaxes(1, 2).reshape(-1, PATCH * PATCH)
    t = t[v.all(axis=1)]
    if t.size == 0:
        return np.empty(0)
    resid = t @ _patch_projector().T
    return np.sqrt((resid ** 2).sum(axis=1) / (PATCH * PATCH - 6))


def stripe_ratio(tile: np.ndarray, track_angle: float) -> float:
    """Energy on the track-aligned wedges / isotropic expectation."""
    n0, n1 = tile.shape
    win = np.outer(np.hanning(n0), np.hanning(n1))
    spec = np.abs(np.fft.fft2((tile - tile.mean()) * win)) ** 2
    ky = np.fft.fftfreq(n0)[:, None]
    kx = np.fft.fftfreq(n1)[None, :]
    kr = np.hypot(ky, kx)
    ring = (kr >= STRIPE_KMIN) & (kr <= 0.5)
    total = spec[ring].sum()
    if total <= 0:
        return float("nan")
    sin_d = math.sin(math.radians(STRIPE_WEDGE_DEG))
    best = 0.0
    for ang in (track_angle, track_angle + math.pi / 2):
        u_r, u_c = math.cos(ang), math.sin(ang)  # feature direction (row, col)
        # Features constant along u put their energy where k . u == 0.
        dot = np.abs(ky * u_r + kx * u_c) / np.where(kr > 0, kr, 1.0)
        wedge = ring & (dot < sin_d)
        expected = wedge.sum() / ring.sum()
        if expected <= 0:
            continue
        best = max(best, (spec[wedge].sum() / total) / expected)
    return float(best)


def small_holes(nodata: np.ndarray, footprint: np.ndarray) -> tuple[int, int]:
    """(pixels, components) of small isolated no-data holes inside the footprint."""
    invalid = nodata | ~footprint
    if not invalid.any():
        return 0, 0
    lab, n = ndimage.label(invalid, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    border = np.zeros(n + 1, dtype=bool)
    for edge in (lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]):
        border[np.unique(edge)] = True
    outside = np.zeros(n + 1, dtype=bool)
    outside[np.unique(lab[~footprint])] = True
    ok = (sizes <= HOLE_MAX_PX) & ~border & ~outside
    ok[0] = False
    return int(sizes[ok].sum()), int(ok.sum())


# ---------------------------------------------------------------------------
# Image-level analysis
# ---------------------------------------------------------------------------


def weighted_median(values: list[float], weights: list[float]) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(v) & (w > 0)
    if not ok.any():
        return float("nan")
    v, w = v[ok], w[ok]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


@dataclass
class TileResult:
    weight: float
    fp_px: int
    hole_px: int
    hole_n: int
    oor_px: int
    texture: bool = False
    res_ratio: float = float("nan")
    sigma_n: float = float("nan")
    snr_db: float = float("nan")
    stripe: float = float("nan")
    patch_resid: np.ndarray = field(default_factory=lambda: np.empty(0))


def _empty_result(row: dict[str, str]) -> dict[str, object]:
    out: dict[str, object] = {k: "" for k in METRIC_COLUMNS}
    out.update(product_id=row["product_id"], mission=row["mission"], subset=row["subset"])
    return out


def _candidate_tiles(geom: Geometry, raster: Raster, th: int, tw: int) -> list[tuple[int, int, float]]:
    """Non-overlapping tiles (row, col, |lat| of centre) inside the footprint."""
    candidates = []
    for r0 in range(0, raster.lines - th + 1, th):
        for c0 in range(0, raster.samples - tw + 1, tw):
            lat = geom.tile_center_lat(r0, c0, th, tw, raster.lines)
            if geom.polar and lat < geom.polar_min_abs_lat:
                continue
            candidates.append((r0, c0, abs(lat)))
    return candidates


def _texture_subscores(tex: list[TileResult], q_floor: float) -> tuple[dict[str, str], dict[str, float]]:
    """Aggregate the texture tiles into (reported metrics, S_res/S_snr/S_stripe)."""
    weights = [t.weight for t in tex]
    res_ratio = weighted_median([t.res_ratio for t in tex], weights)
    sigma_ref = weighted_median([t.sigma_n for t in tex], weights)
    if math.isfinite(sigma_ref):
        sigma_ref = max(sigma_ref, q_floor)

    # Interpolated patches: residual well below the image noise.
    resid = [(t.patch_resid, t.weight) for t in tex if t.patch_resid.size]
    interp = float("nan")
    if resid and math.isfinite(sigma_ref) and sigma_ref > 0:
        num = sum(w * np.mean(r < INTERP_RESID_FACTOR * sigma_ref) for r, w in resid)
        interp = float(num / sum(w for _, w in resid))
    snr_db = weighted_median([t.snr_db for t in tex], weights)
    stripe = weighted_median([t.stripe for t in tex], weights)

    metrics = dict(resolution_ratio=_fmt(res_ratio), interp_patch_fraction=_fmt(interp),
                   snr_db=_fmt(snr_db), noise_sigma=_fmt(sigma_ref), stripe_ratio=_fmt(stripe))
    sub: dict[str, float] = {}
    res_terms = []
    if math.isfinite(res_ratio):
        res_terms.append(res_ratio)
    if math.isfinite(interp):
        res_terms.append(1.0 - interp)
    if res_terms:
        sub["S_res"] = clip01(float(np.mean(res_terms)))
    if math.isfinite(snr_db):
        sub["S_snr"] = clip01((snr_db - SNR_DB_ZERO) / (SNR_DB_ONE - SNR_DB_ZERO))
    if math.isfinite(stripe):
        sub["S_stripe"] = clip01(1.0 - math.log10(max(stripe, STRIPE_R_OK) / STRIPE_R_OK)
                                 / math.log10(STRIPE_R_BAD / STRIPE_R_OK))
    return metrics, sub


def analyse_image(row: dict[str, str], root: Path) -> dict[str, object]:
    out = _empty_result(row)

    if row.get("value_kind") in EXCLUDED_VALUE_KINDS:
        out["status"] = "excluded (binary mask)"
        return out

    raster, geom, sentinels = open_image(row, root)
    out["projection_class"] = "polar" if geom.polar else "cylindrical"

    quantity = row.get("quantity", "")
    fraction_kind = row.get("value_kind") == "fraction"
    log_domain = quantity in LOG_QUANTITIES
    out["sentinels"] = ";".join(f"{s:g}" for s in sentinels)

    axes = (0, 1) if geom.polar else (0,)
    q_floor = abs(raster.scale) / math.sqrt(12.0) if raster.is_integer else 0.0

    # ---- tile sampling ---------------------------------------------------
    th, tw = min(TILE, raster.lines), min(TILE, raster.samples)
    candidates = _candidate_tiles(geom, raster, th, tw)
    rng = np.random.default_rng(zlib.crc32(row["product_id"].encode()))
    order = rng.permutation(len(candidates))

    tiles: list[TileResult] = []
    # Tiles with 60-95 % valid pixels, used only if too few complete tiles exist.
    relaxed_pool: list[tuple[np.ndarray, np.ndarray, float, float, TileResult]] = []
    pooled: list[np.ndarray] = []
    n_texture = 0

    for idx in order:
        if len(tiles) >= MAX_READ_TILES or n_texture >= MAX_TEXTURE_TILES:
            break
        r0, c0, abs_lat = candidates[idx]
        raw = np.asarray(raster.data[r0:r0 + th, c0:c0 + tw])
        fp = geom.footprint(r0, c0, th, tw)
        nd = nodata_mask(raw, raster, sentinels)
        phys = raw.astype(np.float64) * raster.scale + raster.offset
        valid = fp & ~nd

        oor = valid & out_of_range(phys, quantity)
        hole_px, hole_n = small_holes(nd, fp)
        tr = TileResult(weight=geom.area_weight(abs_lat), fp_px=int(fp.sum()),
                        hole_px=hole_px, hole_n=hole_n, oor_px=int(oor.sum()))
        tiles.append(tr)

        good = valid & ~oor
        if log_domain:
            pos = good & (phys > 0)
            phys = np.where(pos, np.log10(np.where(pos, phys, 1.0)), np.nan)
            good = pos
        if good.any():
            vals = phys[good]
            if vals.size > POOL_PER_TILE:
                vals = rng.choice(vals, POOL_PER_TILE, replace=False)
            pooled.append(vals)

        frac_valid = good.sum() / good.size
        angle = geom.track_angle(r0, c0, th, tw)
        if frac_valid >= TEXTURE_VALID_STRICT:
            _texture_metrics(phys, good, tr, geom, abs_lat, angle, axes, q_floor)
            n_texture += 1
        elif frac_valid >= TEXTURE_VALID_RELAXED and len(relaxed_pool) < MAX_TEXTURE_TILES:
            relaxed_pool.append((phys, good, abs_lat, angle, tr))

    if n_texture < MIN_TEXTURE_TILES and relaxed_pool:
        relaxed_pool.sort(key=lambda item: -item[1].mean())  # most complete first
        for phys, good, abs_lat, angle, tr in relaxed_pool[: MAX_TEXTURE_TILES - n_texture]:
            _texture_metrics(phys, good, tr, geom, abs_lat, angle, axes, q_floor)
            n_texture += 1

    out["tiles_read"] = len(tiles)
    out["tiles_texture"] = n_texture

    # ---- aggregation -----------------------------------------------------
    sub: dict[str, float] = {}
    tex = [t for t in tiles if t.texture]
    if tex:
        metrics, texture_sub = _texture_subscores(tex, q_floor)
        out.update(metrics)
        sub.update(texture_sub)

    fp_w = sum(t.weight * t.fp_px for t in tiles)
    if fp_w > 0:
        # Holes are coverage, not degradation: only out-of-range values count,
        # so dropout_fraction == out_of_range_fraction.
        oor_frac = sum(t.weight * t.oor_px for t in tiles) / fp_w
        holes_mpx = sum(t.hole_n for t in tiles) / (sum(t.fp_px for t in tiles) / 1e6)
        out.update(dropout_fraction=_fmt(oor_frac), out_of_range_fraction=_fmt(oor_frac),
                   small_holes_per_mpx=_fmt(holes_mpx))
        sub["S_dropout"] = clip01(1.0 - oor_frac / DROPOUT_BAD_FRACTION)

    values = np.concatenate(pooled) if pooled else np.empty(0)
    values = values[np.isfinite(values)]
    if values.size >= 1000:
        clip_excess, gap = _clip_quant(values, raster, fraction_kind and not log_domain)
        out.update(clip_excess_fraction=_fmt(clip_excess), hist_gap_fraction=_fmt(gap))
        sub["S_clipquant"] = clip01(1.0 - clip_excess / CLIP_BAD_FRACTION) * clip01(1.0 - gap)

    for key in WEIGHTS:
        out[key] = _fmt(sub.get(key, float("nan")))
    if sub:
        wsum = sum(WEIGHTS[k] for k in sub)
        score = sum(WEIGHTS[k] * v for k, v in sub.items()) / wsum
        out["quality_score"] = f"{score:.6f}"
        missing = [k for k in WEIGHTS if k not in sub]
        out["status"] = "ok" if not missing else "partial (missing " + ",".join(missing) + ")"
    else:
        out["status"] = "no valid data"
    return out


def _texture_metrics(phys: np.ndarray, good: np.ndarray, tr: TileResult, geom: Geometry,
                     abs_lat: float, track_angle: float, axes: tuple[int, ...], q_floor: float) -> None:
    """Fill the resolution, noise, stripe and patch fields of *tr*."""
    filled = phys
    if not good.all():
        idx = ndimage.distance_transform_edt(~good, return_distances=False, return_indices=True)
        filled = phys[tuple(idx)]
    detr = detrend_plane(filled)

    # Effective resolution per axis; the worst axis defines the image.  In
    # cylindrical maps the longitude axis is over-sampled by 1/cos(lat): its
    # ratio is corrected and used only up to |lat| = 60 deg.  In polar
    # stereographic maps both axes are corrected by the scale factor k.
    ratios = []
    for ax in (0, 1):
        s_ax = noise_sigma(detr, good, (ax,))
        if math.isfinite(s_ax):
            s_ax = max(s_ax, q_floor)
        r_ax = psd_resolution_ratio(detr, ax, s_ax)
        if not math.isfinite(r_ax):
            continue
        if geom.polar:
            ratios.append(min(1.0, r_ax * geom.scale_factor(abs_lat)))
        elif ax == 0:
            ratios.append(r_ax)
        else:
            coslat = math.cos(math.radians(abs_lat))
            if coslat >= 0.5:
                ratios.append(min(1.0, r_ax / coslat))
    ratio = min(ratios) if ratios else float("nan")

    # Noise is measured at the effective sampling step: in an over-sampled
    # or interpolated image the pixel-to-pixel differences are artificially
    # smooth and would otherwise give a spuriously high SNR.
    step = int(round(1.0 / ratio)) if math.isfinite(ratio) and ratio > 0 else 1
    step = max(1, min(step, 16))
    sl = tuple(slice(None, None, step) if ax in axes else slice(None) for ax in (0, 1))
    sigma_n = noise_sigma(detr[sl], good[sl], axes)
    if math.isfinite(sigma_n):
        sigma_n = max(sigma_n, q_floor, 1e-12 * max(1.0, float(np.nanmax(np.abs(filled)))))
    sigma_s = float(np.subtract(*np.percentile(detr[good], [75, 25])) / 1.349)
    signal = math.sqrt(max(sigma_s ** 2 - sigma_n ** 2, 0.0)) if math.isfinite(sigma_n) else float("nan")

    tr.texture = True
    tr.sigma_n = sigma_n
    if not (math.isfinite(signal) and sigma_n > 0):
        tr.snr_db = float("nan")
    elif signal == 0:
        tr.snr_db = -60.0  # pure noise
    else:
        tr.snr_db = 20.0 * math.log10(signal / sigma_n)
    tr.res_ratio = ratio
    tr.stripe = stripe_ratio(detr, track_angle)
    tr.patch_resid = patch_residuals(detr, good)


def _clip_quant(values: np.ndarray, raster: Raster, fraction_kind: bool) -> tuple[float, float]:
    n = values.size
    vmin, vmax = float(values.min()), float(values.max())
    excess = 0.0
    for v in (vmin, vmax):
        if fraction_kind and (abs(v) < 1e-9 or abs(v - 1.0) < 1e-9):
            continue
        excess += max(0.0, float(np.mean(values == v)) - 1.0 / n)
    if vmin == vmax:
        return excess, 1.0

    p5, p95 = np.percentile(values, [5, 95])
    central = values[(values >= p5) & (values <= p95)]
    if p95 <= p5 or central.size < 100:
        return excess, 0.0
    lo, hi, bins = p5, p95, 256
    if raster.is_integer and raster.scale != 0:
        # Few integer levels: one bin per level, centred on it.
        levels = int(round((p95 - p5) / abs(raster.scale))) + 1
        if levels <= bins:
            half = abs(raster.scale) / 2.0
            lo, hi, bins = p5 - half, p95 + half, levels
    bins = min(bins, max(1, central.size // 50))  # >= 50 samples per bin on average
    hist, _ = np.histogram(central, bins=np.linspace(lo, hi, bins + 1))
    gap = float(np.mean(hist == 0)) if bins > 1 else 0.0
    return excess, gap


def _fmt(x: float) -> str:
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        return ""
    return f"{x:.6g}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _worker(args: tuple[dict[str, str], str]) -> dict[str, object]:
    row, root = args
    try:
        return analyse_image(row, Path(root))
    except Exception as exc:  # noqa: BLE001 - report and continue
        out = _empty_result(row)
        out["status"] = f"error: {exc}"
        return out


def _write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict], encoding: str) -> None:
    """Write a CSV to a temporary file, then replace *path* in one step (never left half-written)."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _load_checkpoint(metrics_out: Path, wanted: set[str]) -> dict[str, dict[str, object]]:
    """Results already in the metrics file (previous interrupted run), errors excluded."""
    if not metrics_out.exists():
        return {}
    with metrics_out.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != METRIC_COLUMNS:
            raise SystemExit(f"{metrics_out}: columns differ from the current ones, cannot resume "
                             "(run without --resume to start again)")
        return {r["product_id"]: r for r in reader
                if r["product_id"] in wanted and not r["status"].startswith("error")}


def main(argv: list[str] | None = None) -> int:
    dataset_root = DEFAULT_ROOT
    parser = argparse.ArgumentParser(description="Compute per-image quality scores and update data_inventory.csv.")
    parser.add_argument("--root", default=str(dataset_root), help="dataset root")
    parser.add_argument("--inventory", default=None, help="inventory CSV (default: <root>/preprocessing/sources/data_inventory.csv)")
    parser.add_argument("--metrics-out", default=None,
                        help="CSV with all metrics and sub-scores (default: <root>/preprocessing/sources/data_quality_metrics.csv)")
    parser.add_argument("--only", nargs="+", metavar="PRODUCT_ID", help="analyse only these products")
    parser.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    parser.add_argument("--resume", action="store_true",
                        help="skip the images already in the metrics file (continue an interrupted run)")
    parser.add_argument("--dry-run", action="store_true", help="do not modify the inventory CSV")
    args = parser.parse_args(argv)
    if args.resume and args.only:
        parser.error("--resume cannot be combined with --only")

    root = Path(args.root).resolve()
    inventory = Path(args.inventory) if args.inventory else inventory_path(root)
    metrics_out = Path(args.metrics_out) if args.metrics_out else metrics_path(root)

    with inventory.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "quality_score" not in fieldnames:
        fieldnames.append("quality_score")

    only = set(args.only or [])
    targets = [r for r in rows if not only or r["product_id"] in only]

    # Checkpoint (full runs only): every finished image is appended at once to the metrics
    # file, so an interrupted run loses nothing and can go on with --resume.  With --only the
    # metrics file is not touched, not to replace the complete one with a partial one.
    checkpoint = not args.only
    results: dict[str, dict[str, object]] = {}
    if args.resume:
        results = _load_checkpoint(metrics_out, {r["product_id"] for r in targets})
    if checkpoint:
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        _write_csv_atomic(metrics_out, METRIC_COLUMNS, [results[r["product_id"]] for r in targets
                                                         if r["product_id"] in results], "utf-8")

    todo = [r for r in targets if r["product_id"] not in results]
    print(f"Images to analyse: {len(todo)}  (already done: {len(results)}, jobs={args.jobs})", flush=True)

    t0 = time.time()
    tasks = [(r, str(root)) for r in todo]
    # File name of each image without folder and extension, for the progress lines.
    image_names = {r["product_id"]: Path(r["data_file"]).stem for r in targets}

    def collect(finished) -> None:
        handle = metrics_out.open("a", encoding="utf-8", newline="") if checkpoint else None
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS, lineterminator="\n") if handle else None
        try:
            for i, res in enumerate(finished, 1):
                results[res["product_id"]] = res
                if writer:
                    writer.writerow(res)
                    handle.flush()
                score = res["quality_score"] or "-"
                print(f"[{i}/{len(tasks)}] {image_names[res['product_id']]}  quality_score={score}  "
                      f"({res['status']}, {time.time() - t0:.0f} s)", flush=True)
        finally:
            if handle:
                handle.close()

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(_worker, t) for t in tasks]
            collect(fut.result() for fut in as_completed(futures))
    else:
        collect(map(_worker, tasks))
    print(f"Elapsed: {time.time() - t0:.0f} s", flush=True)

    # Final metrics file in inventory order (during the run it is in completion order).
    ordered = [results[r["product_id"]] for r in targets if r["product_id"] in results]
    if checkpoint:
        _write_csv_atomic(metrics_out, METRIC_COLUMNS, ordered, "utf-8")
        print(f"Metrics      : {metrics_out}")

    if args.dry_run:
        for res in ordered:
            print({k: res[k] for k in METRIC_COLUMNS if res[k] != ""})
        return 0

    for r in rows:
        if r["product_id"] in results:
            r["quality_score"] = results[r["product_id"]]["quality_score"]
    _write_csv_atomic(inventory, fieldnames, rows, "utf-8-sig")
    print(f"Inventory    : {inventory} (quality_score updated)")

    errors = [r for r in ordered if str(r["status"]).startswith("error")]
    for r in errors:
        print(f"  ERROR {r['product_id']}: {r['status']}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
