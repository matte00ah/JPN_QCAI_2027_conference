#!/usr/bin/env python3
"""Build a CSV inventory of the lunar data downloaded for this dataset.

The script recursively scans a dataset directory for PDS3 ``.lbl`` files and
keeps only those that describe an ``.img`` file which is actually present on
disk (a label may point to its image through ``^IMAGE``/``FILE_NAME`` or the
image may simply share the label stem).  This intentionally excludes companion
labels for browse PNGs or other non-image products.  Image files that carry
their PDS3 label inline (as LROC products do) are also recognised, so every
actual image gets one inventory row.

For every image product it extracts the information requested for the
inventory: mission, sensor, subset, product id, processing level, modality,
reference system/projection, native resolution, geographic footprint, image
dimensions, data type, no-data value, scaling, bands, acquisition time,
illumination angles, version, source product and global/regional/polar group.
Radar (Mini-RF) products additionally receive look direction and radar
incidence angle columns.

Controlled vocabularies (used by the analysis plan and data_analysis.py):

* ``modality``: optical (LROC), DEM (LOLA elevation, slope, roughness),
  radar (Mini-RF), far-UV (LAMP), illumination (LOLA PSR / visibility);
* ``quantity``: the physical quantity (elevation, slope, roughness,
  reflectance, stokes_s1 ... circular_polarization_ratio, permanent_shadow,
  solar_visibility, earth_visibility, uv_albedo, uv_snr, ...);
* ``value_kind``: continuous, binary_mask, fraction, ratio, uncertainty
  (decides which quality checks make sense);
* ``unit``: unit of the physical values (after scale and offset);
* ``native_resolution_m``: pixel size in metres whatever the label unit;
* ``group``: global (one file covers the Moon), global_tile (tile of a
  global mosaic), polar (footprint contains a pole, one hemisphere only),
  regional.

``derived_from`` lists source products named in the label (keywords or the
free-text description).  ``scale_factor``/``offset`` default to the PDS3
values 1/0 when absent, and ``nodata_value`` is ``none declared`` when the
label declares no missing constant (non-finite values may still occur).  Other missing or non-applicable values are
written as empty CSV cells.

Every run rebuilds all rows from the labels.  If the output CSV already
exists, the columns it has beyond ``COLUMNS`` (e.g. ``quality_score`` written
by data_quality.py) are carried over by product id.

Usage
-----
    python Dataset/scripts/A_sources/data_inventory.py
    python Dataset/scripts/A_sources/data_inventory.py --root /path/to/dataset --out inventory.csv
    python Dataset/scripts/A_sources/data_inventory.py --include-missing   # keep labels without .img

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Default dataset root: one folder per celestial body (Dataset/Moon, Dataset/Mars), each
# holding one folder per instrument plus the pre-processing outputs of that body.
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "Moon"
DEFAULT_OUTPUT_NAME = "data_inventory.csv"
METRICS_NAME = "data_quality_metrics.csv"


# Every chapter of the pre-processing plan writes its results in its own folder
# below <root>/preprocessing/: chapter A (sources) in "sources", chapter B
# (diagnostics) in "diagnostics/<macro-step>", ...  These helpers are the only
# place where the layout is defined.
def preprocessing_dir(root: Path, *parts: str) -> Path:
    """Output folder of one chapter of the pre-processing plan."""
    return Path(root) / "preprocessing" / Path(*parts)


def sources_dir(root: Path) -> Path:
    """Output folder of chapter A (inventory, quality metrics, coverage)."""
    return preprocessing_dir(root, "sources")


def inventory_path(root: Path) -> Path:
    return sources_dir(root) / DEFAULT_OUTPUT_NAME


def metrics_path(root: Path) -> Path:
    return sources_dir(root) / METRICS_NAME


def clean_dataset_dir(root: Path, stage: str = "native") -> Path:
    """One checkpoint ("stage") of the cleaned dataset, below <root>/preprocessing/clean_dataset/.

    A stage is written only after a heavy step or where the geometry changes, and is never
    modified by a later step.  "native" (chapter C): cleaned images in their original geometry,
    same folder tree and file names as <root>, so an inventory row opens its cleaned image with
    this folder as root.
    """
    return preprocessing_dir(root, "clean_dataset", stage)


def cleaning_dir(root: Path, step: str) -> Path:
    """Logs of the chapter-C cleaning steps of one macro-step (e.g. "C1")."""
    return preprocessing_dir(root, "cleaning", step)

# A value that is absent or explicitly marked as undefined in a PDS label.
NULL_TOKENS = {
    "",
    "N/A",
    "N.A.",
    "NA",
    "NULL",
    "NONE",
    "UNK",
    "UNKNOWN",
    "TBD",
    "T.B.D.",
    "NOT APPLICABLE",
}

# One column for every requested field.  Fields with several natural
# sub-values (bounding box, angles, scale/offset, bands) are split into
# separate columns so that the CSV can be parsed without additional work.
COLUMNS = [
    "mission",
    "sensor",
    "subset",
    "product_id",
    "processing_level",
    "modality",
    "quantity",
    "value_kind",
    "unit",
    "reference_system",
    "projection",
    "native_resolution",
    "native_resolution_m",
    "bbox_min_lat_deg",
    "bbox_min_lon_deg",
    "bbox_max_lat_deg",
    "bbox_max_lon_deg",
    "footprint_polygon_wkt",
    "dimensions",
    "lines",
    "samples",
    "data_type",
    "nodata_value",
    "scale_factor",
    "offset",
    "bands",
    "band_width",
    "start_time_utc",
    "stop_time_utc",
    "incidence_angle_deg",
    "emission_angle_deg",
    "phase_angle_deg",
    "solar_azimuth_deg",
    "data_version",
    "derived_from",
    "group",
    # Radar-specific (Mini-RF / MRFLRO)
    "look_direction",
    "look_azimuth_angle_deg",
    "radar_incidence_angle_deg",
    # Provenance within the local download
    "data_file",
    "label_file",
]

BASE_KEY_RE = re.compile(r"[A-Z0-9_:^]+")
HEX_VALUE_RE = re.compile(r"(\d+)#([0-9A-Fa-f]+)#$")
UNIT_RE = re.compile(r"<\s*([^<>]*)\s*>\s*$")
EXTENSION_RE = re.compile(r"\.(?:img|lbl|png|jpe?g|xml|tiff?|gif)$", re.IGNORECASE)
# Product-like names sometimes mentioned in DESCRIPTION fields, e.g.
# "... based on the LOLA elevation map LDEM_75N_120M ...".
DESCRIPTION_PRODUCT_RE = re.compile(
    r"\b(?:LDEM|LDSM|LDRM|GLOBAL|LSB|LXB|LSZ|LXZ)_[A-Z0-9_]+"
    r"|\bSLDEM[A-Z0-9_]*_[A-Z0-9_]+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PdsValue:
    """A scalar value from a PDS3 label, with its optional unit."""

    text: str
    unit: str | None = None


# ---------------------------------------------------------------------------
# PDS3 label parsing
# ---------------------------------------------------------------------------


def strip_comments(text: str) -> str:
    """Remove ``/* ... */`` comments without touching quoted strings.

    Newlines found inside comments are preserved so that statements are not
    accidentally joined together.
    """

    out: list[str] = []
    i = 0
    n = len(text)
    quote: str | None = None

    while i < n:
        ch = text[i]

        if quote is not None:
            out.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue

        if ch in ('"', "'"):
            quote = ch
            out.append(ch)
            i += 1
            continue

        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end == -1:
                break
            out.append("\n" * text.count("\n", i, end + 2))
            i = end + 2
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def iter_statements(text: str):
    """Yield logical ``KEY = VALUE`` statements.

    PDS3 allows quoted strings and ``{...}`` lists to span several physical
    lines, therefore a simple line-by-line parser is not sufficient.
    """

    buffer: list[str] = []
    quote: str | None = None
    brace_depth = 0

    for ch in text:
        if quote is not None:
            buffer.append(ch)
            if ch == quote:
                quote = None
            continue

        if ch in ('"', "'"):
            quote = ch
            buffer.append(ch)
            continue

        if ch == "{":
            brace_depth += 1
            buffer.append(ch)
            continue

        if ch == "}":
            brace_depth = max(0, brace_depth - 1)
            buffer.append(ch)
            continue

        if ch == "\n" and brace_depth == 0:
            statement = "".join(buffer).strip()
            if statement:
                yield statement
            buffer = []
            continue

        buffer.append(ch)

    statement = "".join(buffer).strip()
    if statement:
        yield statement


def unquote(text: str) -> str:
    """Remove surrounding single/double quotes and normalise whitespace."""

    text = text.strip()
    if len(text) >= 2 and text[0] in ('"', "'") and text[-1] == text[0]:
        text = text[1:-1]
    return re.sub(r"\s+", " ", text).strip()


def split_unit(value: str) -> tuple[str, str | None]:
    """Split a trailing PDS unit, e.g. ``100 <m/pix>`` -> ``100``, ``m/pix``."""

    match = UNIT_RE.search(value)
    if not match:
        return value.strip(), None
    return value[: match.start()].strip(), match.group(1).strip()


class PdsLabel:
    """Parsed PDS3 label.

    ``root`` holds keys outside any ``OBJECT`` block.  ``objects`` maps an
    object path (for example ``UNCOMPRESSED_FILE/IMAGE``) to its keys.
    """

    def __init__(self, path: Path, text: str | None = None) -> None:
        self.path = path
        self.root: dict[str, PdsValue] = {}
        self.objects: dict[str, dict[str, PdsValue]] = {}
        if text is None:
            text = path.read_text(encoding="utf-8", errors="replace")
        self._parse(text)

    def _parse(self, text: str) -> None:
        stack: list[str] = []

        for statement in iter_statements(strip_comments(text)):
            key, separator, raw_value = statement.partition("=")
            if not separator:
                continue

            key = key.strip().upper()
            if not BASE_KEY_RE.fullmatch(key):
                continue

            raw_value = raw_value.strip()

            if key == "OBJECT":
                object_name = unquote(raw_value).upper()
                # Labels for multispectral/cube products may contain several
                # objects with the same name (one BAND_BIN per band).  Give
                # repeated objects a unique path so their keys are preserved.
                if "/".join(stack + [object_name]) in self.objects:
                    index = 2
                    while "/".join(stack + [f"{object_name}#{index}"]) in self.objects:
                        index += 1
                    object_name = f"{object_name}#{index}"
                stack.append(object_name)
                continue

            if key == "END_OBJECT":
                if stack:
                    stack.pop()
                continue

            value, unit = split_unit(raw_value)
            value = unquote(value) if value[:1] in ('"', "'") else re.sub(
                r"\s+", " ", value
            ).strip()

            scope = "/".join(stack)
            target = self.root if not stack else self.objects.setdefault(scope, {})
            target[key] = PdsValue(value, unit)

    def _object(self, suffix: str) -> dict[str, PdsValue]:
        """Return the first object whose path is/ends with *suffix*."""

        for path, values in self.objects.items():
            if path == suffix or path.endswith("/" + suffix):
                return values
        return {}

    @property
    def image(self) -> dict[str, PdsValue]:
        return self._object("IMAGE")

    @property
    def projection(self) -> dict[str, PdsValue]:
        return self._object("IMAGE_MAP_PROJECTION")


def lookup(mapping: dict[str, PdsValue], key: str) -> PdsValue | None:
    """Look up *key*, ignoring optional XML-style namespaces (``LRO:KEY``)."""

    if key in mapping:
        return mapping[key]

    wanted = key.split(":")[-1]
    for candidate, value in mapping.items():
        if candidate.split(":")[-1] == wanted:
            return value
    return None


def find(label: PdsLabel, key: str) -> PdsValue | None:
    """Look up *key* in the root, image, projection, then any other object."""

    for mapping in (label.root, label.image, label.projection):
        value = lookup(mapping, key)
        if value is not None:
            return value

    for mapping in label.objects.values():
        value = lookup(mapping, key)
        if value is not None:
            return value
    return None


# ---------------------------------------------------------------------------
# Value conversion helpers
# ---------------------------------------------------------------------------


def text_value(value: PdsValue | None) -> str | None:
    """Return a cleaned string, or ``None`` for PDS null markers."""

    if value is None:
        return None

    text = value.text.strip()
    return None if text.upper() in NULL_TOKENS else text


def number(value: PdsValue | None) -> float | int | None:
    """Convert a PDS scalar to a number, supporting ``16#FF#`` notation."""

    text = text_value(value)
    if text is None:
        return None

    hex_match = HEX_VALUE_RE.fullmatch(text)
    if hex_match:
        try:
            return int(hex_match.group(2), int(hex_match.group(1)))
        except ValueError:
            return None

    # PDS/Fortran labels occasionally use D instead of E for exponents.
    normalised = text.replace("D", "E").replace("d", "e")
    try:
        return float(normalised)
    except ValueError:
        return None


def format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        if value.is_integer() and abs(value) < 1e16:
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def format_quantity(value: PdsValue | None) -> str:
    """Format a value with its label unit, e.g. ``0.5`` or ``100 m/pix``."""

    if value is None:
        return ""

    numeric = number(value)
    if numeric is None:
        return text_value(value) or ""

    result = format_number(numeric)
    if value.unit:
        result += " " + value.unit
    return result


def clean_product_id(value: PdsValue | None) -> str | None:
    """Return a product id without a trailing file extension."""

    text = text_value(value)
    if text is None:
        return None
    return EXTENSION_RE.sub("", text).strip() or None


def decode_nodata(value: PdsValue, sample_type: str | None, bits: int | None) -> str:
    """Decode a PDS no-data constant (including IEEE hexadecimal NaNs)."""

    text = text_value(value)
    if text is None:
        return ""

    hex_match = HEX_VALUE_RE.fullmatch(text)
    if hex_match and sample_type and "REAL" in sample_type.upper():
        width = int(bits or 32)
        if width in (32, 64):
            raw = int(hex_match.group(2), int(hex_match.group(1)))
            try:
                decoded = struct.unpack(
                    ">f" if width == 32 else ">d",
                    raw.to_bytes(width // 8, byteorder="big", signed=False),
                )[0]
            except (OverflowError, struct.error, ValueError):
                return text

            if math.isnan(decoded):
                return f"NaN ({text})"
            if math.isinf(decoded):
                sign = "+" if decoded > 0 else "-"
                return f"{sign}Infinity ({text})"
            return f"{format_number(decoded)} ({text})"

    return format_number(number(value)) or text


# ---------------------------------------------------------------------------
# Metadata inference
# ---------------------------------------------------------------------------


def infer_mission(label: PdsLabel, mission_dir: str) -> str | None:
    for key in ("MISSION_ID", "INSTRUMENT_HOST_ID"):
        value = text_value(find(label, key))
        if value:
            return value

    host_name = (
        text_value(find(label, "MISSION_NAME"))
        or text_value(find(label, "INSTRUMENT_HOST_NAME"))
        or ""
    ).upper()
    if "LUNAR RECONNAISSANCE ORBITER" in host_name:
        return "LRO"
    return mission_dir.upper() or None


def infer_sensor(label: PdsLabel, mission_dir: str) -> str | None:
    value = text_value(find(label, "INSTRUMENT_ID"))
    if value:
        return value
    return mission_dir.upper() or None


def infer_processing_level(label: PdsLabel) -> str | None:
    """Map PDS product/data-set descriptors to raw/calibrated/derived."""

    product_type = (text_value(find(label, "PRODUCT_TYPE")) or "").upper()
    data_set_id = (text_value(find(label, "DATA_SET_ID")) or "").upper()
    data_set_name = (text_value(find(label, "DATA_SET_NAME")) or "").upper()
    description = (text_value(find(label, "DESCRIPTION")) or "").upper()
    combined = " ".join((product_type, data_set_id, data_set_name, description))

    # The processing-level token in a PDS data-set id, e.g.
    # LRO-L-MRFLRO-5-CDR-MAP-V1.0 -> CDR.
    match = re.search(r"-\d+-([A-Z0-9]+)-", data_set_id)
    level_token = match.group(1) if match else ""

    for token in (product_type, level_token):
        if token in {"EDR", "RAW"}:
            return "raw"

    for token in (product_type, level_token):
        if token in {"RDR", "CDR"}:
            return "calibrated"

    if "CALIBRATED" in combined or "MAP-PROJECTED" in combined:
        return "calibrated"

    for token in (product_type, level_token):
        if token in {"DDR", "GDR"}:
            return "derived"

    if any(
        word in combined
        for word in ("DERIVED", "MOSAIC", "GRIDDED", "BROWSE", "MAP OF")
    ):
        return "derived"

    if "RAW" in combined:
        return "raw"

    return None


def infer_modality(label: PdsLabel, sensor: str | None) -> str | None:
    """Infer the modality, using the vocabulary of the analysis plan.

    ``optical`` (LROC), ``DEM`` (LOLA elevation and its derivatives: slope,
    roughness), ``radar`` (Mini-RF), ``far-UV`` (LAMP) and ``illumination``
    (LOLA permanent-shadow and visibility maps).  ``RGB`` and
    ``multispectral`` are kept for multi-band products only.
    """

    sensor = (sensor or "").upper()
    product = (text_value(find(label, "PRODUCT_ID")) or "").upper()

    # Instrument-based rules first: they are unambiguous for this dataset and
    # avoid keyword accidents (e.g. "elevation" quoted in a slope label).
    if sensor == "LOLA":
        if any(token in product for token in ("AVGVISIB", "LPSR", "VISIB", "ILLUM", "PSR")):
            return "illumination"
        return "DEM"
    if sensor == "LAMP":
        return "far-UV"
    if sensor == "MRFLRO":
        return "radar"
    if sensor == "LROC":
        # Each WAC_HAPKE file is one band of the 7-band multispectral mosaic.
        if product.startswith("WAC_HAPKE"):
            return "multispectral"
        bands = band_count(label) or 1
        return "optical" if bands <= 1 else "multispectral"
    text = " ".join(
        filter(
            None,
            (
                product,
                text_value(find(label, "DATA_SET_NAME")),
                text_value(find(label, "DESCRIPTION")),
                text_value(find(label, "INSTRUMENT_NAME")),
            ),
        )
    ).upper()

    if sensor == "MRFLRO" or any(
        word in text for word in ("RADAR", "MINI-RF", "SAR", "CPR", "LPR")
    ):
        return "radar"

    # A red/green/blue band set is an RGB product even when the label also
    # carries a generic "multispectral" description.
    band_names: set[str] = set()
    for path, values in label.objects.items():
        if "BAND" not in path.upper():
            continue
        name = text_value(lookup(values, "BAND_NAME"))
        if name:
            band_names.add(name.upper())
    if {"RED", "GREEN", "BLUE"} <= band_names:
        return "RGB"

    if any(word in text for word in ("RGB", "COLOR", "COLOUR", "RED-GREEN-BLUE")):
        return "RGB"

    # LOLA illumination / visibility / PSR maps are raster products derived
    # from a DEM, but they are not elevation models themselves.
    if sensor == "LOLA" and any(
        token in product for token in ("AVGVISIB", "LPSR", "VISIB", "ILLUM")
    ):
        return None

    if any(
        word in text
        for word in ("ELEVATION", "TOPOGRAPHY", "DIGITAL ELEVATION", "SHAPE MAP")
    ):
        return "DEM"

    if any(
        word in text
        for word in ("MULTISPECTRAL", "SPECTROMETER", "SPECTROGRAPH", "SPECTRAL")
    ):
        return "multispectral"

    if any(
        word in text
        for word in ("MONOCHROME", "GRAYSCALE", "GREYSCALE", "PANCHROMATIC")
    ):
        return "optical"

    return None


# Mini-RF Level 3 product token -> physical quantity.
RADAR_QUANTITIES = {
    "S1": "stokes_s1",
    "S2": "stokes_s2",
    "S3": "stokes_s3",
    "S4": "stokes_s4",
    "CPR": "circular_polarization_ratio",
    "LPR": "linear_polarization_ratio",
    "M": "degree_of_polarization",
    "MC": "degree_of_circular_polarization",
    "ML": "degree_of_linear_polarization",
    "OC": "opposite_sense_circular_power",
    "SC": "same_sense_circular_power",
}

VALUE_KINDS = {
    "permanent_shadow": "binary_mask",
    "solar_visibility": "fraction",
    "earth_visibility": "fraction",
    "degree_of_polarization": "fraction",
    "uv_snr": "uncertainty",
    "circular_polarization_ratio": "ratio",
    "linear_polarization_ratio": "ratio",
    "degree_of_circular_polarization": "ratio",
    "degree_of_linear_polarization": "ratio",
    "uv_offon_ratio_albedo": "ratio",
    "uv_offon_ratio_brightness": "ratio",
}

UNIT_ALIASES = {
    "METER": "m",
    "METERS": "m",
    "KILOMETER": "km",
    "KILOMETERS": "km",
    "DEGREE": "deg",
    "DEGREES": "deg",
}


def label_description(label: PdsLabel) -> str:
    """Root and IMAGE descriptions joined, for free-text metadata."""

    parts = [
        text_value(label.root.get("DESCRIPTION")),
        text_value(lookup(label.image, "DESCRIPTION")),
    ]
    return " ".join(part for part in parts if part)


def radar_token(product_id: str) -> str | None:
    match = re.match(r"GLOBAL_([A-Z0-9]+)_\d+PPD", product_id.upper())
    return match.group(1) if match else None


def infer_quantity(
    label: PdsLabel, sensor: str | None, product_id: str, subset: str
) -> str | None:
    """Physical quantity stored in the image (elevation, slope, CPR, ...)."""

    sensor = (sensor or "").upper()
    pid = product_id.upper()
    sub = subset.upper()

    if sensor == "MRFLRO":
        return RADAR_QUANTITIES.get(radar_token(pid) or "")
    if sensor == "LOLA":
        if "VISIB" in pid:
            return "earth_visibility" if "EARTH" in pid else "solar_visibility"
        if pid.startswith("LPSR"):
            return "permanent_shadow"
        if pid.startswith("LDSM"):
            return "slope"
        if pid.startswith("LDRM"):
            return "roughness"
        if pid.startswith(("LDEM", "SLDEM")):
            return "elevation"
    if sensor == "LROC":
        return "reflectance"
    if sensor == "LAMP":
        if "ERROR" in sub:
            return "uv_snr"
        if "RATIO" in sub:
            return "uv_offon_ratio_" + ("albedo" if "ALBED" in pid else "brightness")
        for token, quantity in (
            ("ALBEDO", "uv_albedo"),
            ("BRIGHTNESS", "uv_brightness"),
            ("ILLUMINATION", "uv_illumination"),
        ):
            if token in sub:
                return quantity

    name = text_value(lookup(label.image, "NAME"))
    return name.lower() if name else None


def infer_unit(label: PdsLabel, value_kind: str) -> str | None:
    """Physical unit of the stored values (after scale and offset)."""

    unit = text_value(lookup(label.image, "UNIT"))
    if unit:
        return UNIT_ALIASES.get(unit.upper(), unit)

    description = label_description(label)
    match = re.search(r"\bin\s+(photons/cm2/sec(?:/sr)?)", description, re.IGNORECASE)
    if match:
        return match.group(1)
    if re.search(r"\bin\s+meters\b", description, re.IGNORECASE):
        return "m"
    if value_kind in {"binary_mask", "fraction", "ratio", "uncertainty"} or re.search(
        r"dimensionless", description, re.IGNORECASE
    ):
        return "dimensionless"
    return None


def reference_system(label: PdsLabel) -> str | None:
    projection = label.projection
    coord_type = text_value(lookup(projection, "COORDINATE_SYSTEM_TYPE"))
    coord_name = text_value(lookup(projection, "COORDINATE_SYSTEM_NAME"))
    latitude_type = text_value(lookup(projection, "KEYWORD_LATITUDE_TYPE"))

    pieces = [piece for piece in (coord_type, coord_name, latitude_type) if piece]
    # Avoid repeating the same token (MRFLRO uses PLANETOCENTRIC in both fields).
    unique: list[str] = []
    for piece in pieces:
        if piece.upper() not in {item.upper() for item in unique}:
            unique.append(piece)

    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    return f"{unique[0]} ({'; '.join(unique[1:])})"


def native_resolution(label: PdsLabel) -> str | None:
    projection = label.projection

    scale = lookup(projection, "MAP_SCALE")
    if scale is not None:
        return format_quantity(scale) or None

    resolution = lookup(projection, "MAP_RESOLUTION")
    if resolution is not None:
        return format_quantity(resolution) or None

    return None


MOON_RADIUS_M = 1_737_400.0


def native_resolution_m(label: PdsLabel) -> float | None:
    """Pixel size in metres, whatever unit the label uses.

    MAP_SCALE may be in km/pixel, m/pix or METERS/PIXEL; MAP_RESOLUTION is in
    pixels/degree and is converted on the 1737.4 km IAU sphere.
    """

    projection = label.projection
    scale = lookup(projection, "MAP_SCALE")
    value = number(scale)
    if value is not None:
        unit = (scale.unit or "").upper()
        if unit.startswith("KM") or unit.startswith("KILOMETER"):
            return value * 1000.0
        return value

    ppd = number(lookup(projection, "MAP_RESOLUTION"))
    if ppd:
        return 2.0 * math.pi * MOON_RADIUS_M / 360.0 / ppd
    return None


def bounding_box(label: PdsLabel) -> tuple[float | None, float | None, float | None, float | None]:
    projection = label.projection
    min_lat = number(lookup(projection, "MINIMUM_LATITUDE"))
    max_lat = number(lookup(projection, "MAXIMUM_LATITUDE"))
    west = number(lookup(projection, "WESTERNMOST_LONGITUDE"))
    east = number(lookup(projection, "EASTERNMOST_LONGITUDE"))
    projection_name = (text_value(lookup(projection, "MAP_PROJECTION_TYPE")) or "").upper()

    if min_lat is not None and max_lat is not None:
        corrected = False

        # Polar labels normally include the pole as 90 (or -90).  A few
        # archived labels contain a sign error (e.g. MINIMUM_LATITUDE=90,
        # MAXIMUM_LATITUDE=-82.9); the projection centre tells us which pole
        # the cap belongs to and lets us repair the boundary.
        if "POLAR" in projection_name:
            center_lat = number(lookup(projection, "CENTER_LATITUDE"))
            if center_lat is not None and abs(center_lat) >= 80:
                raw_values = (min_lat, max_lat)
                if any(abs(abs(value) - 90) < 0.1 for value in raw_values):
                    boundaries = [value for value in raw_values if abs(value) < 89.9]
                    if boundaries:
                        boundary = min(boundaries, key=abs)
                        if center_lat > 0:
                            min_lat, max_lat = abs(boundary), 90.0
                        else:
                            min_lat, max_lat = -90.0, -abs(boundary)
                        corrected = True

        if not corrected and min_lat > max_lat:
            min_lat, max_lat = max_lat, min_lat

    # Polar labels use "N/A" for longitude because the product spans every
    # meridian.  Represent that as the full 0-360 range.
    if west is None or east is None:
        if "POLAR" in projection_name:
            west, east = 0.0, 360.0
    else:
        # Global products are sometimes written as 360 -> 0 or 0 -> 360.
        if abs(east - west) >= 359.9:
            west, east = 0.0, 360.0
        else:
            west = west % 360
            east = east % 360
            # Preserve the 330 -> 360 style rather than turning it into
            # 330 -> 0, which would look like a reversed interval.
            if east == 0 and west > 0:
                east = 360.0

    return min_lat, max_lat, west, east


def is_global_product(
    min_lat: float | None, max_lat: float | None, west: float | None, east: float | None
) -> bool:
    if min_lat is None or max_lat is None:
        return False

    lat_span = abs(max_lat - min_lat)
    if west is None or east is None:
        lon_span = 360.0
    else:
        lon_span = (east - west) % 360
        if lon_span == 0:
            lon_span = 360.0

    return lat_span >= 170 and lon_span >= 350


def build_polygon(
    min_lat: float | None,
    max_lat: float | None,
    west: float | None,
    east: float | None,
) -> str | None:
    """Build a WKT footprint polygon from the label's map extents.

    Longitudes follow the PDS convention (degrees east, 0-360).  Polar
    products span every meridian, so their footprint becomes a rectangle from
    0 to 360 degrees once the missing longitudes have been filled in.
    """

    if min_lat is None or max_lat is None:
        return None

    if west is None or east is None:
        west, east = 0.0, 360.0

    unwrapped_east = east + 360.0 if east <= west else east
    points = [
        (west, min_lat),
        (unwrapped_east, min_lat),
        (unwrapped_east, max_lat),
        (west, max_lat),
        (west, min_lat),
    ]

    coordinates = ", ".join(
        f"{format_number(lon)} {format_number(lat)}" for lon, lat in points
    )
    return f"POLYGON (({coordinates}))"


def infer_group(
    min_lat: float | None,
    max_lat: float | None,
    west: float | None,
    east: float | None,
    projection: str | None,
) -> str | None:
    """Geographic group of a single file.

    ``global``: the file covers the whole Moon; ``polar``: the footprint
    contains a pole and does not cross the equator; ``regional``: anything
    else.  Tiles of a global mosaic are relabelled ``global_tile`` later by
    :func:`assign_global_tiles`, which needs all records.
    """

    if is_global_product(min_lat, max_lat, west, east):
        return "global"
    if min_lat is None or max_lat is None:
        return "polar" if "POLAR" in (projection or "").upper() else None

    touches_pole = max_lat >= 89.9 or min_lat <= -89.9
    one_hemisphere = min_lat >= 0 or max_lat <= 0
    if touches_pole and one_hemisphere:
        return "polar"
    return "regional"


def _footprint_area_fraction(record: dict[str, str]) -> float:
    """Fraction of the sphere covered by the bounding box of a record."""

    try:
        min_lat = float(record["bbox_min_lat_deg"])
        max_lat = float(record["bbox_max_lat_deg"])
        west = float(record["bbox_min_lon_deg"])
        east = float(record["bbox_max_lon_deg"])
    except (KeyError, ValueError):
        return 0.0
    lon_span = (east - west) % 360 or 360.0
    band = math.sin(math.radians(max_lat)) - math.sin(math.radians(min_lat))
    return max(0.0, band) * lon_span / 720.0


def assign_global_tiles(records: list[dict[str, str]]) -> None:
    """Mark the tiles of a global mosaic as ``global_tile``.

    Files are grouped by sensor, subset, quantity and pixel size (so that,
    e.g., the four LDEM_256 quarters or the ten WAC_GLOBAL tiles form one
    family).  A family whose footprints add up to the whole sphere and span
    all latitudes is a tiled global mosaic.
    """

    families: dict[tuple, list[dict[str, str]]] = {}
    for record in records:
        try:
            pixel = round(float(record["native_resolution_m"]))
        except ValueError:
            pixel = record["native_resolution_m"]
        key = (record["sensor"], record["subset"], record["quantity"], pixel)
        families.setdefault(key, []).append(record)

    for members in families.values():
        if len(members) < 2:
            continue
        intervals = []
        for record in members:
            try:
                intervals.append(
                    (float(record["bbox_min_lat_deg"]), float(record["bbox_max_lat_deg"]))
                )
            except ValueError:
                break
        else:
            intervals.sort()
            reach = -90.0
            for low, high in intervals:
                if low > reach + 0.1:
                    break
                reach = max(reach, high)
            area = sum(_footprint_area_fraction(record) for record in members)
            if reach >= 89.9 and area >= 0.99:
                for record in members:
                    if record["group"] != "global":
                        record["group"] = "global_tile"


def illumination_angles(label: PdsLabel) -> dict[str, str]:
    """Extract incidence, emission, phase and solar azimuth angles.

    Mosaics have no single keyword value; when the description quotes an
    incidence range (e.g. WAC_GLOBAL: "incidence angles (53-70 degrees") the
    range is written as ``low-high``.
    """

    def angle(*keys: str) -> str:
        for key in keys:
            value = find(label, key)
            if value is not None:
                return format_number(number(value))
        return ""

    incidence = angle("INCIDENCE_ANGLE", "ANGLE_OF_INCIDENCE")
    if not incidence:
        match = re.search(
            r"incidence\s+angles?\s*\(?\s*(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*deg",
            label_description(label),
            re.IGNORECASE,
        )
        if match:
            incidence = f"{match.group(1)}-{match.group(2)}"

    return {
        "incidence_angle_deg": incidence,
        "emission_angle_deg": angle("EMISSION_ANGLE"),
        "phase_angle_deg": angle("PHASE_ANGLE"),
        "solar_azimuth_deg": angle(
            "SOLAR_AZIMUTH_ANGLE", "SOLAR_AZIMUTH", "SUB_SOLAR_AZIMUTH"
        ),
    }


def radar_metadata(label: PdsLabel) -> dict[str, str]:
    look_direction = text_value(
        find(label, "LOOK_DIRECTION")
    ) or text_value(find(label, "OBSERVATION_DIRECTION"))

    look_azimuth = find(label, "LOOK_AZIMUTH_ANGLE")
    radar_incidence = find(label, "INCIDENCE_ANGLE") or find(label, "ANGLE_OF_INCIDENCE")

    return {
        "look_direction": look_direction or "",
        "look_azimuth_angle_deg": format_number(number(look_azimuth)),
        "radar_incidence_angle_deg": format_number(number(radar_incidence)),
    }


def derived_from(label: PdsLabel) -> str | None:
    """Collect explicit source-product links from the label.

    A few derived products (notably LOLA illumination/visibility maps) do not
    use ``SOURCE_PRODUCT_ID`` but name their source product in the free-text
    DESCRIPTION, for example "... based on the LOLA elevation map
    LDEM_75N_120M ...".  Those references are captured as a fallback.
    """

    values: list[str] = []
    for key in ("SOURCE_PRODUCT_ID", "ORIGINAL_PRODUCT_ID", "DERIVED_FROM"):
        product = clean_product_id(find(label, key))
        if product and product not in values:
            values.append(product)

    current_product = clean_product_id(find(label, "PRODUCT_ID"))
    description = text_value(find(label, "DESCRIPTION"))
    if description:
        for match in DESCRIPTION_PRODUCT_RE.findall(description):
            product = clean_product_id(PdsValue(match))
            if (
                product
                and product != current_product
                and product not in values
            ):
                values.append(product)

    return "; ".join(values) or None


def documented_sources(label: PdsLabel, sensor: str | None, product_id: str) -> list[str]:
    """Source products stated in the free-text description of the label.

    Only relations written in the label are used:

    * Mini-RF: the defining formula ("defined as CPR = SC / OS", "M =
      [(S2^2 + S3^2 + S4^2)^0.5]/S1", ...) gives the sibling products, and
      every Level 3 mosaic is built from Level 2 CDRs;
    * LOLA: the altimetry mission phase and, for SLDEM2015, the Kaguya/SELENE
      Terrain Camera co-registration;
    * LROC: the WAC images and orbit range of the monthly mosaics;
    * LAMP: the off-band and on-band maps used by the off/on ratios.
    """

    sensor = (sensor or "").upper()
    description = label_description(label)
    pid = product_id.upper()
    sources: list[str] = []

    if sensor == "MRFLRO":
        own = radar_token(pid)
        match = re.search(r"defined\s+as\s+(.*?)(?:\.\s|\.$|$)", description, re.IGNORECASE)
        if match and own:
            for token in re.findall(r"\b(S[1-4]|SC|OC|OS|M)\b", match.group(1).upper()):
                token = "OC" if token == "OS" else token
                if token != own:
                    sibling = pid.replace(f"GLOBAL_{own}_", f"GLOBAL_{token}_", 1)
                    if sibling not in sources:
                        sources.append(sibling)
        if re.search(r"Level 2 CDRs?", description, re.IGNORECASE):
            sources.append("Mini-RF Level 2 CDRs (multiple acquisitions)")

    elif sensor == "LOLA":
        phase = re.search(r"mission\s+phase\s+(LRO_[A-Z]{2}_\d+)", description, re.IGNORECASE)
        if re.search(r"altimetr|LOLA (?:data|topograph)|LOLA instrument", description, re.IGNORECASE):
            text = "LOLA altimetry"
            if phase:
                text += f" through {phase.group(1).upper()}"
            sources.append(text)
        # Only a co-registration makes SELENE a source (LDEM_45N/S labels
        # mention SELENE merely as a comparison).
        if re.search(r"co-?registered\s+with\s+[^.]{0,40}(SELENE|Kaguya)", description, re.IGNORECASE):
            sources.append("Kaguya/SELENE Terrain Camera DEM")

    elif sensor == "LROC":
        orbits = re.search(r"from\s+orbit\s+(\d+)\s+to\s+orbit\s+(\d+)", description, re.IGNORECASE)
        if orbits:
            sources.append(f"LROC WAC images, orbits {orbits.group(1)}-{orbits.group(2)}")
        elif re.search(r"\bWAC\b", description):
            sources.append("LROC WAC images")

    elif sensor == "LAMP":
        match = re.search(r"off-band\s+to\s+on-band\s+(albedo|brightness)", description, re.IGNORECASE)
        if match:
            kind = "albed" if match.group(1).lower() == "albedo" else "brite"
            lower = "_ratio_" in product_id
            for band in ("off", "on"):
                replacement = f"_{band}_{kind}_" if lower else f"_{band}_{kind}_".upper()
                sibling = re.sub(r"_ratio_[a-z]+_", replacement, product_id, flags=re.IGNORECASE)
                if sibling != product_id:
                    sources.append(sibling)

    return sources


def band_width(label: PdsLabel) -> str | None:
    """Return the spectral width/range of the product when one is described."""

    image = label.image

    explicit = (
        lookup(image, "BANDWIDTH")
        or lookup(image, "BAND_WIDTH")
        or lookup(label.root, "BANDWIDTH")
        or lookup(label.root, "BAND_WIDTH")
    )
    if explicit is not None and text_value(explicit) is not None:
        return format_quantity(explicit)

    # Multispectral labels may describe each band in its own BAND_BIN object.
    bandwidths: list[str] = []
    for path, values in label.objects.items():
        if "BAND" not in path.upper():
            continue
        value = lookup(values, "BANDWIDTH") or lookup(values, "BAND_WIDTH")
        if value is not None:
            text = format_quantity(value)
            if text and text not in bandwidths:
                bandwidths.append(text)
    if bandwidths:
        return "; ".join(bandwidths)

    minimum = lookup(label.root, "MINIMUM_WAVELENGTH") or lookup(
        image, "MINIMUM_WAVELENGTH"
    )
    maximum = lookup(label.root, "MAXIMUM_WAVELENGTH") or lookup(
        image, "MAXIMUM_WAVELENGTH"
    )

    if minimum is not None and maximum is not None:
        low = format_number(number(minimum))
        high = format_number(number(maximum))
        unit = minimum.unit or maximum.unit
        result = f"{low}-{high}"
        if unit:
            result += f" {unit}"
        return result

    band_name = lookup(image, "BAND_NAME") or lookup(label.root, "BAND_NAME")
    if band_name is not None:
        return text_value(band_name)

    # LROC WAC labels give the filter centre wavelength in nm (FILTER_NAME = 321).
    filter_name = number(lookup(image, "FILTER_NAME"))
    if filter_name is not None:
        return f"{format_number(filter_name)} nm"

    return None


def band_count(label: PdsLabel) -> int | None:
    image = label.image
    value = number(lookup(image, "BANDS") or lookup(label.root, "BANDS"))
    if value is not None:
        return int(value)

    # Some multispectral labels use one BAND_BIN object per band.
    band_objects = [
        path
        for path in label.objects
        if "BAND_BIN" in path.upper() or path.upper().endswith("BAND")
    ]
    if band_objects:
        return len(band_objects)

    # Most PDS3 single-image products omit BANDS; it is one by definition.
    return 1 if image else None


# ---------------------------------------------------------------------------
# Image discovery and record construction
# ---------------------------------------------------------------------------


def resolve_case_insensitive(path: Path) -> Path | None:
    if path.exists():
        return path

    parent = path.parent
    try:
        for child in parent.iterdir():
            if child.name.lower() == path.name.lower():
                return child
    except OSError:
        pass
    return None


def find_image(label_path: Path, label: PdsLabel) -> Path | None:
    """Locate the ``.img`` file described by a label."""

    candidates: list[Path] = []

    for key in ("^IMAGE", "FILE_NAME"):
        value = text_value(find(label, key))
        if value:
            name = Path(value.replace("\\", "/")).name
            candidates.append(label_path.with_name(name))

    candidates.append(label_path.with_suffix(".img"))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        resolved = resolve_case_insensitive(candidate)
        if resolved is not None and resolved.is_file() and resolved.suffix.lower() == ".img":
            return resolved

    return None


EMBEDDED_LABEL_LIMIT = 1 << 20  # 1 MiB is far larger than any PDS3 header.


def read_embedded_label(path: Path) -> str | None:
    """Return the PDS3 label stored inline at the start of *path*, if any.

    Some archives (notably LROC) distribute ``.IMG`` files whose label lives
    in the first records of the image file instead of a separate ``.lbl``.
    Such a label is plain text terminated by a line containing only ``END``.
    """

    try:
        with path.open("rb") as handle:
            head = handle.read(EMBEDDED_LABEL_LIMIT)
    except OSError:
        return None

    text = head.decode("latin-1", errors="replace")
    if not text.lstrip().startswith("PDS_VERSION_ID"):
        return None

    offset = 0
    for line in text.splitlines(keepends=True):
        offset += len(line)
        if line.strip() == "END":
            return text[:offset]
    return None


def build_record(
    root: Path,
    label_path: Path,
    image_path: Path | None,
    label: PdsLabel | None = None,
) -> dict[str, str]:
    label = label or PdsLabel(label_path)
    relative = label_path.relative_to(root)

    mission_dir = relative.parts[0] if len(relative.parts) > 1 else ""
    mission = infer_mission(label, mission_dir)
    sensor = infer_sensor(label, mission_dir)

    # Subset = path below the mission folder, without generic data/image
    # directory names, so that e.g. LAMP/DATA/ALBEDO becomes "ALBEDO".
    subset_parts = list(relative.parts[1:-1]) if len(relative.parts) > 2 else []
    if subset_parts and subset_parts[0].lower() == "data":
        subset_parts = subset_parts[1:]
    if subset_parts and subset_parts[-1].lower() in {"img", "float_img"}:
        subset_parts = subset_parts[:-1]
    subset = "/".join(subset_parts)

    product_id = clean_product_id(find(label, "PRODUCT_ID"))
    if product_id is None:
        # A few older LOLA labels omit PRODUCT_ID; use the image file name so
        # the id keeps the archive's upper-case convention.
        product_id = clean_product_id(find(label, "FILE_NAME")) or label_path.stem

    # The local LROC folders (WAC_global, WAC_Hapke/<band>, ...) are a
    # download layout only: the archive subset (WAC_GLOBAL, WAC_ORBITS,
    # WAC_HAPKE) is taken from the product id, the band is in band_width.
    if sensor == "LROC":
        subset = "_".join(product_id.split("_")[:2])

    processing_level = infer_processing_level(label)
    modality = infer_modality(label, sensor)
    quantity = infer_quantity(label, sensor, product_id, subset)
    value_kind = VALUE_KINDS.get(quantity or "", "continuous")
    unit = infer_unit(label, value_kind)

    projection = text_value(lookup(label.projection, "MAP_PROJECTION_TYPE"))
    ref_system = reference_system(label)
    resolution = native_resolution(label)
    resolution_m = native_resolution_m(label)

    min_lat, max_lat, west, east = bounding_box(label)
    polygon = build_polygon(min_lat, max_lat, west, east)
    group = infer_group(min_lat, max_lat, west, east, projection)

    image = label.image
    lines = number(lookup(image, "LINES") or lookup(label.root, "LINES"))
    samples = number(
        lookup(image, "LINE_SAMPLES") or lookup(label.root, "LINE_SAMPLES")
    )
    bands = band_count(label)

    dimensions_parts = [format_number(lines), format_number(samples)]
    if bands is not None and bands > 1:
        dimensions_parts.append(format_number(bands))
    dimensions = " x ".join(part for part in dimensions_parts if part) or None

    sample_type = text_value(
        lookup(image, "SAMPLE_TYPE") or lookup(label.root, "SAMPLE_TYPE")
    )
    sample_bits = number(
        lookup(image, "SAMPLE_BITS") or lookup(label.root, "SAMPLE_BITS")
    )
    if sample_type and sample_bits is not None:
        data_type = f"{sample_type} ({format_number(sample_bits)}-bit)"
    else:
        data_type = sample_type

    nodata_parts: list[str] = []
    for key in (
        "MISSING_CONSTANT",
        "CORE_NULL",
        "NULL_VALUE",
        "MISSING_VALUE",
        "NODATA_VALUE",
        "INVALID_CONSTANT",
    ):
        value = lookup(image, key) or lookup(label.root, key)
        if value is not None and text_value(value) is not None:
            decoded = decode_nodata(value, sample_type, int(sample_bits) if sample_bits else None)
            if decoded and decoded not in nodata_parts:
                nodata_parts.append(decoded)
    # "none declared" = the label has no missing constant (LOLA LDEM and
    # SLDEM2015 are fully interpolated grids; LAMP may still contain NaN).
    nodata = "; ".join(nodata_parts) or "none declared"

    # PDS3 defaults when the keywords are absent: physical = DN * 1 + 0.
    scale_factor = (
        format_quantity(
            lookup(image, "SCALING_FACTOR") or lookup(label.root, "SCALING_FACTOR")
        )
        or "1"
    )
    offset = (
        format_quantity(lookup(image, "OFFSET") or lookup(label.root, "OFFSET"))
        or "0"
    )

    sources = [
        item for item in (derived_from(label) or "").split("; ") if item
    ]
    for item in documented_sources(label, sensor, product_id):
        if item not in sources:
            sources.append(item)

    record = {
        "mission": mission or "",
        "sensor": sensor or "",
        "subset": subset,
        "product_id": product_id or "",
        "processing_level": processing_level or "",
        "modality": modality or "",
        "quantity": quantity or "",
        "value_kind": value_kind,
        "unit": unit or "",
        "reference_system": ref_system or "",
        "projection": projection or "",
        "native_resolution": resolution or "",
        "native_resolution_m": format_number(
            round(resolution_m, 4) if resolution_m is not None else None
        ),
        "bbox_min_lat_deg": format_number(min_lat),
        "bbox_min_lon_deg": format_number(west),
        "bbox_max_lat_deg": format_number(max_lat),
        "bbox_max_lon_deg": format_number(east),
        "footprint_polygon_wkt": polygon or "",
        "dimensions": dimensions or "",
        "lines": format_number(lines),
        "samples": format_number(samples),
        "data_type": data_type or "",
        "nodata_value": nodata or "",
        "scale_factor": scale_factor or "",
        "offset": offset or "",
        "bands": format_number(bands),
        "band_width": band_width(label) or "",
        "start_time_utc": text_value(find(label, "START_TIME")) or "",
        "stop_time_utc": text_value(find(label, "STOP_TIME")) or "",
        "data_version": text_value(find(label, "PRODUCT_VERSION_ID")) or "",
        "derived_from": "; ".join(sources),
        "group": group or "",
        "data_file": image_path.relative_to(root).as_posix() if image_path else "",
        "label_file": relative.as_posix(),
    }

    record.update(illumination_angles(label))

    if modality == "radar":
        record.update(radar_metadata(label))
    else:
        record.update(
            {
                "look_direction": "",
                "look_azimuth_angle_deg": "",
                "radar_incidence_angle_deg": "",
            }
        )

    return record


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def iter_labels(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".lbl":
            continue
        relative_parts = path.relative_to(root).parts
        if any(part == "__pycache__" or part.startswith(".") for part in relative_parts):
            continue
        yield path


def iter_label_sources(root: Path):
    """Yield ``(path, embedded_text)`` for every image product label.

    Ordinary ``.lbl`` files are yielded with ``embedded_text=None``.  Image
    files carrying an inline PDS3 label are yielded with their path and the
    extracted label text.  Images that already have an external label are not
    yielded twice.
    """

    for label_path in iter_labels(root):
        yield label_path, None

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".img":
            continue
        relative_parts = path.relative_to(root).parts
        if any(part == "__pycache__" or part.startswith(".") for part in relative_parts):
            continue
        if resolve_case_insensitive(path.with_suffix(".lbl")) is not None:
            continue
        text = read_embedded_label(path)
        if text is not None:
            yield path, text


def carry_over_columns(path: Path, records: list[dict[str, str]]) -> list[str]:
    """Copy columns added by other scripts (e.g. ``quality_score``) from an
    existing inventory into the new records, matched by product id.

    Returns the names of the carried-over columns; new products get empty
    cells.  Without this, re-running the inventory would drop those columns.
    """

    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        extra = [name for name in reader.fieldnames or [] if name not in COLUMNS]
        previous = {row["product_id"]: row for row in reader}
    for record in records:
        old = previous.get(record["product_id"], {})
        for name in extra:
            record[name] = old.get(name, "")
    return extra


def write_csv(path: Path, records: list[dict[str, str]], extra: list[str]) -> None:
    # Same encoding and line endings as data_quality.py, which rewrites the file.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=COLUMNS + extra, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)


def main(argv: list[str] | None = None) -> int:
    dataset_root = DEFAULT_ROOT

    parser = argparse.ArgumentParser(
        description="Create a CSV inventory of all PDS3 image products downloaded for the dataset."
    )
    parser.add_argument(
        "--root",
        metavar="DIR",
        default=str(dataset_root),
        help=f"dataset root to scan (default: {dataset_root})",
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        default=None,
        help="output CSV file (default: <root>/preprocessing/sources/data_inventory.csv)",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="also inventory labels whose .img file is not present",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    output = Path(args.out).expanduser() if args.out else inventory_path(root)
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()

    if not root.is_dir():
        print(f"error: dataset root not found: {root}", file=sys.stderr)
        return 2

    records: list[dict[str, str]] = []
    skipped: list[Path] = []
    errors: list[str] = []

    for source_path, embedded_text in iter_label_sources(root):
        try:
            label = PdsLabel(source_path, text=embedded_text)
            if embedded_text is None:
                image_path = find_image(source_path, label)
            else:
                image_path = source_path
        except Exception as exc:  # noqa: BLE001 - report and continue
            errors.append(f"  {source_path}: {exc}")
            continue

        if image_path is None and not args.include_missing:
            skipped.append(source_path)
            continue

        try:
            records.append(build_record(root, source_path, image_path, label))
        except Exception as exc:  # noqa: BLE001 - report and continue
            errors.append(f"  {source_path}: {exc}")

    assign_global_tiles(records)
    records.sort(key=lambda row: (row["mission"], row["subset"], row["product_id"]))
    extra = carry_over_columns(output, records)
    write_csv(output, records, extra)

    print(f"Dataset root : {root}")
    print(f"Output CSV   : {output}")
    print(f"Image records: {len(records)}")

    if records:
        missions = Counter(record["mission"] or "(unknown)" for record in records)
        print("By mission   : " + ", ".join(f"{name}={count}" for name, count in sorted(missions.items())))
        sensors = Counter(record["sensor"] or "(unknown)" for record in records)
        print("By sensor    : " + ", ".join(f"{name}={count}" for name, count in sorted(sensors.items())))

    if skipped:
        print(f"Skipped      : {len(skipped)} label(s) without a matching .img")
        for path in skipped:
            print(f"  - {path.relative_to(root).as_posix()}")

    if errors:
        print(f"Errors       : {len(errors)}", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
