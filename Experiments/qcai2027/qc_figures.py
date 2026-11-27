#!/usr/bin/env python3
"""Figures and tables of the QCAI 2027 experiment (plan phases QD and QL, and the visual checks).

Outputs: check figures, one per phase, in ``results/figures/``; figures and tables of the
paper in ``results/paper/`` (PNG at 300 dpi; ``--pdf`` also writes the vector PDF); the
numbers in the ``QB``/``QC``/``QD`` reports of ``results/`` (``qc_data.save_report``).

Commands
--------
qb-checks        QB2.1 + QB3.2 checks -> ``figures/qb_checks.png``: (a) Theophilus
                 (D = 100 km) and (b) a mare area with 1-3 km craters across the WAC
                 seam at the equator, WAC crops with the catalogue craters (ellipses
                 1/cos(lat) wider E-W); (c) the cells used by each set after the
                 tile buffer, on the WAC mosaic of R.
tile-check       QC8.1 -> ``figures/qc_checks.png`` + report section QC8.1: 10 random
                 tiles (kept and valid, seed of the config) as the backbone sees
                 them (``TileReader``, layers in [0, 1]): WAC, relief, slope,
                 hillshade, with the catalogue craters (solid: 1-8 km targets;
                 dashed: D >= 8 km) and the owned 11 x 11 cells (QC7.2); numeric
                 check on the craters with D >= 2 km fully inside the tiles: floor
                 lower than the rim (relief), walls steeper than the centre
                 (slope), inner east wall brighter than the west wall (WAC, Sun
                 from the west, D14).
region-map       QD-A1, Fig. D1 -> ``paper/fig_d1_region_map.png`` + section QD-A1:
                 WAC mosaic of R, 5 deg blocks coloured by set (S1-val hatched),
                 dot area = density of 1-8 km craters per block, main maria; side
                 panel: density of the blocks of each set.
sfd              QD-A2, Fig. D2 -> ``paper/fig_d2_sfd.png`` + section QD-A2:
                 cumulative size-frequency distribution of the catalogue craters of
                 R (log-log), label bands, 1-2 km shaded as completeness limit,
                 power law fitted on 2-8 km; bottom: observed / power law.
example-tiles    QD-A3, Fig. D3 -> ``paper/fig_d3_example_tiles.png`` + section
                 QD-A3: a dense, a medium and a sparse tile (90th / 50th / 10th
                 percentile of the positive sub-tiles among the 121 owned; ties
                 broken with the seed); columns WAC, relief, slope, hillshade and
                 the labels on the 16 x 16 sub-tile grid.
dataset-table    QD-A4, Table D1 -> ``paper/table_d1_dataset.tex`` (LaTeX, booktabs;
                 printed as Markdown): per set (report section QC8.2,
                 ``split_blocks.csv``) and per modality (dataset inventory and
                 quality table: native pixel, effective resolution = native /
                 ``resolution_ratio`` (B1.3), quality score).
complementarity  QD-A5, Fig. D4 -> ``paper/fig_d4_complementarity.png`` + section
                 QD-A5: Pearson r between the WAC and the hillshade of the DEM in
                 every sub-tile of the dataset, with the Sun of the WAC (from the
                 west, D14; computed here, not an input layer) and from the north
                 (the input layer, QC4.3): (a) map of the median r per tile, (b)
                 histogram; (c) rim visibility of the 1-8 km catalogue craters in
                 each modality (z-score of the rim contrast against a background
                 ring: WAC east - west inner wall, DEM rim - floor relief), as
                 both / WAC only / DEM only / neither per diameter band.

quantum-vs-classical  QJ1, Fig. J1 (Fig. 2 of the paper) -> ``paper/fig_j1_quantum_vs_classical.png`` +
                 section QJ1-figure of ``qj_classical.json``: (a) test AP vs qubits and (b) vs training size for
                 the three QSVC maps, the RBF and the linear SVM (mean +- std over the seeds; stage 1 alone dashed);
                 (c) paired AP difference classical - quantum on the same samples; (d) F1 per representation.
                 Colour = method family (quantum blue, RBF orange, linear aqua), marker / line = method.

Usage
-----
    python Experiments/qcai2027/qc_figures.py <command> [--pdf]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_data import (CONFIG_PATH, MOON_RADIUS_M, SETS, block_area_km2, load_config, load_report,  # noqa: E402
                     read_region, region_layout, save_report, written)

# (name, crater_id at the centre of the crop or None, centre row, centre col, size px, title)
OVERLAY_CROPS = [
    ("theophilus", "07-1-000385", None, None, 1800, "Theophilus (D = 100 km) and surroundings"),
    ("equator_seam", None, 9097, 15300, 600, "Mare area across the WAC seam at the equator (dashed)"),
]


# QB3.3: one hue per set family (first three slots of the categorical palette, which stay
# distinct for colour-blind readers); S1-val shares the S1 hue and is told apart by hatching
SET_STYLE = {"s1_train": ("#2a78d6", "", "S1-train"), "s1_val": ("#2a78d6", "////", "S1-val"),
             "qk_pool": ("#eb6834", "", "QK-pool"), "test": ("#1baf7a", "", "Test")}
INK, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
CRATER_COLOR = "#eb6834"      # one hue for the catalogue outlines, visible on the grey images


# QD-A1: main maria inside R, approximate centres (IAU Gazetteer of Planetary Nomenclature), labels only
# (name, lat, lon, horizontal alignment of the label: "right" keeps it inside R at the east edge)
MARIA = [("Tranquillitatis", 8.5, 31.4, "center"), ("Serenitatis", 26.5, 18.0, "center"), ("Crisium", 17.0, 59.1, "center"),
         ("Fecunditatis", -7.8, 51.3, "center"), ("Nectaris", -15.2, 35.5, "center"), ("Smythii", -1.3, 89.5, "right")]


def density_size(d):
    """Marker area (pt^2) of a block with d craters of 1-8 km per 1,000 km2 (QB3.3, QD-A1)."""
    return 2.0 * np.asarray(d)


def draw_blocks(ax, blocks, size_scale: float = 1.0, alpha: float = 0.45) -> None:
    """5 deg blocks coloured by set (S1-val hatched) with the crater density as dot area (QB3.3, QD-A1)."""
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle

    for b in blocks.itertuples():
        colour, hatch, _ = SET_STYLE[b.set]
        w, h = b.lon_max - b.lon_min, b.lat_max - b.lat_min
        ax.add_patch(Rectangle((b.lon_min, b.lat_min), w, h, facecolor=to_rgba(colour, alpha), edgecolor=SURFACE, lw=0.8))
        if hatch:
            ax.add_patch(Rectangle((b.lon_min, b.lat_min), w, h, facecolor="none", edgecolor=to_rgba(SURFACE, 0.9),
                                   hatch=hatch, lw=0))
    ax.scatter((blocks["lon_min"] + blocks["lon_max"]) / 2, (blocks["lat_min"] + blocks["lat_max"]) / 2,
               s=size_scale * density_size(blocks["density_per_1000km2"]), facecolor=SURFACE, edgecolor=INK, lw=0.6, zorder=3)


def block_legend_handles(densities=(10, 30, 60), size_scale: float = 1.0) -> tuple[list, list]:
    """Legend handles of the sets and of the density dots."""
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    sets = [Patch(facecolor=to_rgba(c, 0.6), edgecolor=SURFACE if not h else INK_MUTED, hatch=h, label=lab)
            for c, h, lab in SET_STYLE.values()]
    dens = [Line2D([], [], ls="", marker="o", markersize=float(np.sqrt(size_scale * density_size(d))), markerfacecolor=SURFACE,
                   markeredgecolor=INK, markeredgewidth=0.6, label=f"{d} / 1,000 km²") for d in densities]
    return sets, dens


def figures_dir(cfg: dict) -> Path:
    """Check figures (one per phase)."""
    d = cfg["paths"]["results"] / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_figure(fig, out: Path, pdf: bool = False, dpi: int = 300) -> None:
    """PNG (and, for the paper, the vector PDF if ``pdf``)."""
    import matplotlib.pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf") if pdf else ("png",):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=dpi, facecolor=SURFACE)
        written(out.with_suffix(f".{ext}"))
    plt.close(fig)


def paper_path(cfg: dict, name: str) -> Path:
    """Figures and tables of the paper."""
    return cfg["paths"]["results"] / "paper" / name


def pixel_km(grid) -> float:
    """N-S pixel size (km) of an equirectangular grid."""
    return 2 * math.pi * MOON_RADIUS_M / 360.0 / grid.ppd / 1000.0


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_MUTED, length=2, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(INK_MUTED)
        sp.set_linewidth(0.5)


def draw_craters(ax, craters, r0: int, c0: int, px_km: float, color: str = "red", lw: float = 0.6, ls: str = "-") -> None:
    """Catalogue craters as ellipses on an axis showing the pixels from (r0, c0) of the WAC grid of R
    (N-S diameter D / pixel size, 1/cos(lat) wider E-W on the equirectangular grid)."""
    from matplotlib.patches import Ellipse

    for k in craters.itertuples():
        h = k.diam_km / px_km
        ax.add_patch(Ellipse((k.col - c0, k.row - r0), h / math.cos(math.radians(k.lat)), h,
                             fill=False, color=color, lw=lw, ls=ls))


# ---------------------------------------------------------------------------
# QB checks (QB2.1 crater positions, QB3.2 cells used per set)
# ---------------------------------------------------------------------------

# (crater_id at the centre of the crop or None, centre row, centre col, size px, title)
OVERLAY_CROPS = [("07-1-000385", None, None, 1800, "(a) Theophilus (D = 100 km), catalogue craters"),
                 (None, 9097, 15300, 600, "(b) mare across the WAC seam at the equator (dashed)")]


def draw_crop(ax, src, craters, r0: int, c0: int, size: int, title: str) -> int:
    """WAC crop of R with the craters whose centre falls in it (and the WAC seam); returns their number."""
    img = read_region(src, slice(r0, r0 + size), slice(c0, c0 + size))
    sel = craters[craters["row"].between(r0 - 0.5, r0 + size - 0.5) & craters["col"].between(c0 - 0.5, c0 + size - 0.5)]
    ax.imshow(img, cmap="gray", vmin=np.nanpercentile(img, 1), vmax=np.nanpercentile(img, 99))
    draw_craters(ax, sel, r0, c0, pixel_km(src.grid), color=CRATER_COLOR, lw=0.6)
    for seam in {p.dst_rows.start for p in src.pieces} - {0}:
        if r0 < seam < r0 + size:
            ax.axhline(seam - 0.5 - r0, color="#3987e5", ls="--", lw=0.9)
    km = size * pixel_km(src.grid)
    ax.set_title(f"{title}\n{len(sel)} craters, crop {km:.0f} x {km:.0f} km", fontsize=8, color=INK, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    style_axes(ax)
    return len(sel)


def run_qb_checks(cfg: dict, step: int = 16) -> str:
    """QB2.1 + QB3.2 check figure -> ``figures/qb_checks.png``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle

    reg = cfg["region"]
    src = region_layout(cfg, "wac")
    g = src.grid
    craters = pd.read_csv(cfg["paths"]["output"] / "craters_R.csv")
    fig = plt.figure(figsize=(11, 10.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85])
    for k, (cid, rc, cc, size, title) in enumerate(OVERLAY_CROPS):
        if cid is not None:
            row = craters.set_index("crater_id").loc[cid]
            rc, cc = int(round(row["row"])), int(round(row["col"]))
        draw_crop(fig.add_subplot(gs[0, k]), src, craters, rc - size // 2, cc - size // 2, size, title)

    # (c) cells used per set after the buffer of QB3.2 (WAC only = not used)
    ax = fig.add_subplot(gs[1, :])
    wac = read_region(src, slice(0, None, step), slice(0, None, step))
    ax.imshow(wac, cmap="gray", vmin=np.nanpercentile(wac, 1), vmax=np.nanpercentile(wac, 99),
              extent=[g.west, g.east, g.south, g.north], interpolation="antialiased")
    cell_set = np.load(cfg["paths"]["output"] / "split_cells.npz")["cell_set"]
    cell_px = int(cfg["tiling"]["cell_px"])
    rgba = np.zeros(cell_set.shape + (4,))
    for k, name in enumerate(SETS):
        rgba[cell_set == k] = to_rgba(SET_STYLE[name][0], 0.5)
    ax.imshow(rgba, extent=[g.west, g.west + cell_set.shape[1] * cell_px / g.ppd,
                            g.north - cell_set.shape[0] * cell_px / g.ppd, g.north], interpolation="nearest")
    blocks = pd.read_csv(cfg["paths"]["output"] / "split_blocks.csv")
    for b in blocks[blocks["set"] == "s1_val"].itertuples():
        ax.add_patch(Rectangle((b.lon_min, b.lat_min), b.lon_max - b.lon_min, b.lat_max - b.lat_min, facecolor="none",
                               edgecolor=to_rgba(SURFACE, 0.9), hatch=SET_STYLE["s1_val"][1], lw=0))
    ax.set_xlim(reg["lon_min"], reg["lon_max"])
    ax.set_ylim(reg["lat_min"], reg["lat_max"])
    ax.set_xticks(np.arange(reg["lon_min"], reg["lon_max"] + 1, 15))
    ax.set_yticks(np.arange(reg["lat_min"], reg["lat_max"] + 1, 10))
    ax.set_xlabel("longitude (°E)", color=INK_MUTED, fontsize=8)
    ax.set_ylabel("latitude (°N)", color=INK_MUTED, fontsize=8)
    ax.set_title("(c) sub-tiles used by each set after the tile buffer (grey = not used: buffer bands, border of R)",
                 fontsize=8, color=INK, loc="left")
    style_axes(ax)
    ax.legend(handles=block_legend_handles()[0], loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8,
              labelcolor=INK)
    fig.suptitle("QB checks: catalogue craters on the WAC (QB2.1) and geographic split after the buffer (QB3.2)",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = figures_dir(cfg) / "qb_checks.png"
    save_figure(fig, out, dpi=150)
    return str(out)


# ---------------------------------------------------------------------------
# QC8.1 - Visual check of random tiles
# ---------------------------------------------------------------------------

TILE_COLUMNS = (("wac", 0, "WAC 643 nm"), ("dem", 0, "relief"), ("dem", 1, "slope"), ("dem", 2, "hillshade (Sun N)"))


def crater_polar(shape: tuple[int, int], rr: float, cc: float, rad: float, cos_lat: float) -> tuple[np.ndarray, np.ndarray]:
    """Pixels of a window around a crater centred at (rr, cc) with radius ``rad`` (N-S px): normalised
    distance rho = distance / radius and E-W offset dx (px, positive = east), E-W scaled by cos(lat)."""
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    dx = (cols - cc) * cos_lat
    return np.hypot(dx, rows - rr) / rad, dx


def crater_profiles(layers: dict, craters, r0: int, c0: int, size: int, px_km: float, d_min_km: float) -> list[dict]:
    """For the craters with D >= d_min_km whose circle is inside the window: mean relief of the floor
    (rho < 0.5) and of the rim (0.9-1.1), mean slope of the centre (rho < 0.3) and of the walls
    (0.6-1.0), mean WAC of the inner east and west walls (0.5-1.0); rho = distance / radius, E-W
    distances scaled by cos(lat)."""
    out = []
    for k in craters.itertuples():
        rad = k.diam_km / px_km / 2.0                                          # radius in N-S px
        cl = math.cos(math.radians(k.lat))
        rr, cc = k.row - r0, k.col - c0
        if k.diam_km < d_min_km or rr - 1.1 * rad < 0 or rr + 1.1 * rad > size - 1 \
                or cc - 1.1 * rad / cl < 0 or cc + 1.1 * rad / cl > size - 1:
            continue
        rho, dx = crater_polar((size, size), rr, cc, rad, cl)
        ring = lambda a, b: (rho >= a) & (rho < b)
        out.append({"crater_id": k.crater_id, "diam_km": float(k.diam_km),
                    "relief_floor": float(layers["relief"][rho < 0.5].mean()), "relief_rim": float(layers["relief"][ring(0.9, 1.1)].mean()),
                    "slope_centre": float(layers["slope"][rho < 0.3].mean()), "slope_wall": float(layers["slope"][ring(0.6, 1.0)].mean()),
                    "wac_east_wall": float(layers["wac"][ring(0.5, 1.0) & (dx > 0)].mean()),
                    "wac_west_wall": float(layers["wac"][ring(0.5, 1.0) & (dx < 0)].mean())})
    return out


def run_tile_check(cfg: dict, n_tiles: int = 10, per_row: int = 2, d_min_km: float = 2.0) -> dict:
    """QC8.1: one check figure of random tiles with the catalogue craters, and the numeric crater check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    from qc_tiles import TileReader

    reader = TileReader(cfg, inputs=("wac", "dem"))
    tiles = pd.read_csv(cfg["paths"]["output"] / "tiles.csv")
    craters = pd.read_csv(cfg["paths"]["output"] / "craters_R.csv")
    d_lo, d_hi = (float(x) for x in cfg["labels"]["diam_range_km"])
    tile_px, cell_px = reader.tile_px, reader.cell_px
    px_km = pixel_km(region_layout(cfg, "wac").grid)
    cand = tiles[tiles["kept"].to_numpy(bool) & reader.tile_used.ravel()]
    pick = cand.iloc[np.sort(np.random.default_rng(int(cfg["seed"])).choice(len(cand), n_tiles, replace=False))]

    n_rows, n_cols = math.ceil(n_tiles / per_row), per_row * len(TILE_COLUMNS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols, 2.45 * n_rows + 0.6), facecolor=SURFACE, squeeze=False)
    profiles, listed = [], []
    for k, t in enumerate(pick.itertuples()):
        row_axes = axes[k // per_row, (k % per_row) * len(TILE_COLUMNS):(k % per_row + 1) * len(TILE_COLUMNS)]
        r0, c0 = int(t.row0), int(t.col0)
        img = reader.read(int(t.ti), int(t.tj), unit=True)["inputs"]
        sel = craters[craters["row"].between(r0 - 0.5, r0 + tile_px - 0.5) & craters["col"].between(c0 - 0.5, c0 + tile_px - 0.5)]
        target, large = sel[sel["diam_km"].between(d_lo, d_hi, inclusive="left")], sel[sel["diam_km"] >= d_hi]
        physical = {n: reader.raw(int(t.ti), int(t.tj), n) for n in ("wac", "relief", "slope")}
        profiles += crater_profiles(physical, sel, r0, c0, tile_px, px_km, d_min_km)
        listed.append(f"{t.tile} {t.set} {t.lat_centre:.1f}N {t.lon_centre:.1f}E")
        off = 2 * cell_px                                                    # owned block (QC7.2), in px
        for ax, (inp, ch, name) in zip(row_axes, TILE_COLUMNS):
            ax.imshow(img[inp][ch], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            draw_craters(ax, target, r0, c0, px_km, color=CRATER_COLOR, lw=0.7)
            draw_craters(ax, large, r0, c0, px_km, color=CRATER_COLOR, lw=0.7, ls="--")
            ax.add_patch(Rectangle((off - 0.5, off - 0.5), 11 * cell_px, 11 * cell_px, fill=False, ec=SURFACE, lw=0.6,
                                   ls=(0, (2, 2))))
            ax.set_xlim(-0.5, tile_px - 0.5)
            ax.set_ylim(tile_px - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            style_axes(ax)
            if k < per_row:
                ax.set_title(name, fontsize=8, color=INK)
        row_axes[0].set_ylabel(f"tile {t.tile} ({t.set})\n{t.lat_centre:.1f}°N {t.lon_centre:.1f}°E", fontsize=7, color=INK)
    handles = [Line2D([], [], color=CRATER_COLOR, lw=1.0, label=f"catalogue crater {d_lo:g}-{d_hi:g} km (target)"),
               Line2D([], [], color=CRATER_COLOR, lw=1.0, ls="--", label=f"catalogue crater ≥ {d_hi:g} km"),
               Line2D([], [], color=INK_MUTED, lw=0.8, ls=(0, (2, 2)), label="owned 11 x 11 sub-tiles (QC7.2)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("QC8.1: 10 random tiles as read by the backbone (layers in [0, 1]); tile = 25.6 km N-S",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save_figure(fig, figures_dir(cfg) / "qc_checks.png", dpi=110)

    pr = pd.DataFrame(profiles)
    frac = lambda m: round(float(np.mean(m)), 3) if len(m) else None
    report = {
        "tiles": listed, "craters_checked_d_ge_km": d_min_km, "craters_checked": int(len(pr)),
        "fraction_floor_below_rim": frac(pr["relief_floor"] < pr["relief_rim"]),
        "fraction_walls_steeper_than_centre": frac(pr["slope_wall"] > pr["slope_centre"]),
        "fraction_east_wall_brighter_than_west_wall": frac(pr["wac_east_wall"] > pr["wac_west_wall"]),
        "median_rim_minus_floor_m": round(float((pr["relief_rim"] - pr["relief_floor"]).median()), 1),
        "median_east_over_west_wall_wac": round(float((pr["wac_east_wall"] / pr["wac_west_wall"]).median()), 3),
        "by_diameter_km": {f"{a:g}-{b:g}": {"craters": int(m.sum()),
                                            "floor_below_rim": frac(pr["relief_floor"][m] < pr["relief_rim"][m]),
                                            "walls_steeper_than_centre": frac(pr["slope_wall"][m] > pr["slope_centre"][m]),
                                            "east_wall_brighter": frac(pr["wac_east_wall"][m] > pr["wac_west_wall"][m])}
                           for a, b in ((d_min_km, 4.0), (4.0, d_hi), (d_hi, np.inf))
                           for m in [pr["diam_km"].between(a, b, inclusive="left").to_numpy()] if m.any()},
    }
    save_report(cfg, "QC8.1", report)
    return report


# ---------------------------------------------------------------------------
# QD-A1 - Region map (Fig. D1)
# ---------------------------------------------------------------------------


def run_region_map(cfg: dict, step: int = 8, width_in: float = 7.16, pdf: bool = False) -> dict:
    """QD-A1: map of R with the split and the crater density, and the density of the blocks per set."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    reg = cfg["region"]
    src = region_layout(cfg, "wac")
    g = src.grid
    wac = read_region(src, slice(0, None, step), slice(0, None, step))
    blocks = pd.read_csv(cfg["paths"]["output"] / "split_blocks.csv")
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "pdf.fonttype": 42})

    fig = plt.figure(figsize=(width_in, 3.55), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.55, 1.0], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax.imshow(wac, cmap="gray", vmin=np.nanpercentile(wac, 1), vmax=np.nanpercentile(wac, 99),
              extent=[g.west, g.east, g.south, g.north], interpolation="antialiased")
    scale = 0.3                                                          # dot areas for the small figure
    draw_blocks(ax, blocks, size_scale=scale, alpha=0.3)                 # lighter: the terrain must stay visible
    for name, lat, lon, ha in MARIA:
        ax.text(lon, lat, f"Mare\n{name}", ha=ha, va="center", fontsize=5.5, color=INK, linespacing=0.9, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.75))
    ax.set_xlim(reg["lon_min"], reg["lon_max"])
    ax.set_ylim(reg["lat_min"], reg["lat_max"])
    ax.set_xticks(np.arange(reg["lon_min"], reg["lon_max"] + 1, 15))
    ax.set_yticks(np.arange(reg["lat_min"], reg["lat_max"] + 1, 10))
    ax.set_xlabel("longitude (°E)", color=INK_MUTED)
    ax.set_ylabel("latitude (°N)", color=INK_MUTED)

    # side panel: density of the blocks of each set (dots, median as a bar)
    bx = fig.add_subplot(gs[1])
    rng = np.random.default_rng(int(cfg["seed"]))
    summary = {}
    for k, name in enumerate(SETS):
        d = blocks.loc[blocks["set"] == name, "density_per_1000km2"].to_numpy()
        colour, hatch, label = SET_STYLE[name]
        y = k + rng.uniform(-0.18, 0.18, d.size)
        bx.scatter(d, y, s=9, facecolor=colour, edgecolor=SURFACE, lw=0.4, zorder=3)
        bx.plot([np.median(d)] * 2, [k - 0.32, k + 0.32], color=INK, lw=1.4, zorder=4)
        summary[name] = {"blocks": int(d.size), **{k: round(float(v), 2) for k, v in
                                                     zip(("min", "q25", "median", "q75", "max"), np.percentile(d, [0, 25, 50, 75, 100]))}}
    bx.set_yticks(range(len(SETS)))
    bx.set_yticklabels([SET_STYLE[n][2] for n in SETS], color=INK)
    bx.set_ylim(len(SETS) - 0.5, -0.5)
    bx.set_xlabel("1-8 km craters / 1,000 km²", color=INK_MUTED)
    bx.grid(axis="x", color=INK_MUTED, alpha=0.2, lw=0.5)
    bx.set_title("block density (bar = median)", fontsize=7, color=INK, loc="left")
    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        a.tick_params(colors=INK_MUTED, length=2)
        for sp in a.spines.values():
            sp.set_color(INK_MUTED)
            sp.set_linewidth(0.5)
    bx.spines[["top", "right"]].set_visible(False)

    set_h, dens_h = block_legend_handles(size_scale=scale)
    fig.legend(handles=set_h + dens_h, loc="lower center", ncol=len(set_h) + len(dens_h), frameon=False, fontsize=6.5,
               labelcolor=INK, bbox_to_anchor=(0.5, -0.01), handlelength=1.4, columnspacing=1.2)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.93, bottom=0.2)
    save_figure(fig, paper_path(cfg, "fig_d1_region_map.png"), pdf=pdf)
    report = {"density_per_1000km2_per_set": summary}
    save_report(cfg, "QD-A1", report)
    return report


# ---------------------------------------------------------------------------
# QD-A2 - Crater size-frequency distribution (Fig. D2)
# ---------------------------------------------------------------------------


def run_sfd(cfg: dict, fit_km: tuple[float, float] = (2.0, 8.0), d_max_plot_km: float = 100.0, width_in: float = 3.5,
            pdf: bool = False) -> dict:
    """QD-A2: cumulative SFD of the craters of R with the completeness limit of the catalogue."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    reg = cfg["region"]
    bands = [float(x) for x in cfg["labels"]["bands_km"]]
    area = float(block_area_km2(reg["lat_min"], reg["lat_max"], reg["lon_max"] - reg["lon_min"]))
    cr = pd.read_csv(cfg["paths"]["output"] / "craters_R.csv")
    d = np.sort(cr.loc[cr["in_region"].astype(bool), "diam_km"].to_numpy())[::-1]       # decreasing
    n_cum = np.arange(1, d.size + 1) / area                                              # N(>= d) per km2

    def cum(x):
        return np.searchsorted(-d, -np.asarray(x), side="right") / area

    # power law N(>= D) = a D^b fitted (least squares in log-log) on sqrt(2) steps inside the complete range
    grid = fit_km[0] * np.sqrt(2.0) ** np.arange(int(round(2 * np.log2(fit_km[1] / fit_km[0]))) + 1)
    b, log_a = np.polyfit(np.log10(grid), np.log10(cum(grid)), 1)
    law = lambda x: 10 ** log_a * np.asarray(x) ** b
    check_d = [1.0, float(np.sqrt(2.0)), 2.0]
    completeness = {f"{x:.2f}": float(cum(x) / law(x)) for x in check_d}
    # local cumulative slope per sqrt(2) bin from 1 km
    edges = np.sqrt(2.0) ** np.arange(0, 2 * int(np.log2(d_max_plot_km)) + 1)
    edges = edges[edges <= 64.0]
    slopes = {f"{a:.2f}-{c:.2f}": float(np.log10(cum(c) / cum(a)) / np.log10(c / a)) for a, c in zip(edges[:-1], edges[1:])}

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "pdf.fonttype": 42})
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(width_in, 3.4), sharex=True, facecolor=SURFACE,
                                 gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.08})
    sel = d <= d_max_plot_km
    for a_ in (ax, rx):
        a_.set_facecolor(SURFACE)
        a_.axvspan(bands[0], bands[1], color=INK_MUTED, alpha=0.12, lw=0)
        a_.axvspan(bands[-1], d_max_plot_km, color=INK_MUTED, alpha=0.05, lw=0)
        for e in bands:
            a_.axvline(e, color=INK_MUTED, lw=0.5, ls=":")
        a_.tick_params(colors=INK_MUTED, length=2, which="both")
        a_.grid(color=INK_MUTED, alpha=0.15, lw=0.4, which="major")
        for sp in a_.spines.values():
            sp.set_color(INK_MUTED)
            sp.set_linewidth(0.5)
        a_.spines[["top", "right"]].set_visible(False)

    # top: cumulative SFD and the power law
    ax.step(d[sel], n_cum[sel], where="post", color=SET_STYLE["s1_train"][0], lw=1.4, label="catalogue craters in R")
    xx = np.array([1.0, d_max_plot_km])
    ax.plot(xx, law(xx), color=INK, lw=0.8, ls="--", label=f"power law fitted on {fit_km[0]:g}-{fit_km[1]:g} km (slope {b:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("cumulative N(≥ D) per km²", color=INK_MUTED)
    ytop = float(law(1.0)) * 1.6
    ax.set_ylim(float(n_cum[sel][0]) * 0.5, ytop)
    yl = ytop / 1.25
    for a, c in zip(bands[:-1], bands[1:]):
        ax.text(np.sqrt(a * c), yl, f"{a:g}-{c:g} km", ha="center", va="top", fontsize=6, color=INK)
    ax.text(np.sqrt(bands[-1] * d_max_plot_km), yl, f"≥ {bands[-1]:g} km (ignore)", ha="center", va="top", fontsize=6,
            color=INK_MUTED)
    ax.text(np.sqrt(bands[0] * bands[1]), float(n_cum[sel][0]) * 0.65, "catalogue\ncompleteness\nlimit", ha="center",
            va="bottom", fontsize=5.5, color=INK_MUTED)
    ax.legend(loc="lower left", bbox_to_anchor=(0.2, 0.0), frameon=False, fontsize=6, labelcolor=INK)

    # bottom: observed / power law
    xs = np.exp(np.linspace(np.log(1.0), np.log(d_max_plot_km), 300))
    ratio = cum(xs) / law(xs)
    rx.plot(xs, ratio, color=SET_STYLE["s1_train"][0], lw=1.2)
    rx.axhline(1.0, color=INK, lw=0.8, ls="--")
    rx.set_ylim(0.6, max(1.2, float(ratio.max()) * 1.08))
    rx.set_ylabel("observed /\npower law", color=INK_MUTED)
    rx.annotate(f"{completeness['1.00']:.2f} at 1 km", xy=(1.0, completeness["1.00"]), xytext=(1.12, 0.66),
                fontsize=5.5, color=INK, va="bottom")
    rx.set_xlim(1.0, d_max_plot_km)
    rx.set_xticks([1, 2, 4, 8, 16, 32, 64])
    rx.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    rx.set_xlabel("crater diameter D (km)", color=INK_MUTED)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.12)
    save_figure(fig, paper_path(cfg, "fig_d2_sfd.png"), pdf=pdf)
    report = {
        "area_R_km2": round(area), "craters_in_R": int(d.size),
        "fit_2_8_km": {"slope": round(float(b), 3), "log10_a": round(float(log_a), 4)},
        "observed_over_power_law": {**{k: round(v, 3) for k, v in completeness.items()},
                                    "range_in_fit_km": [round(float(ratio[(xs >= fit_km[0]) & (xs <= fit_km[1])].min()), 3),
                                                        round(float(ratio[(xs >= fit_km[0]) & (xs <= fit_km[1])].max()), 3)],
                                    "max_above_8_km": round(float(ratio[xs >= bands[-1]].max()), 3)},
        "local_cumulative_slope_per_sqrt2_bin_km": {k: round(v, 2) for k, v in slopes.items()},
    }
    save_report(cfg, "QD-A2", report)
    return report


# ---------------------------------------------------------------------------
# QD-A3 - Example tiles (Fig. D3)
# ---------------------------------------------------------------------------

BAND_COLORS = ("#86b6ef", "#2a78d6", "#104281")     # 1-2 / 2-4 / 4-8 km: one sequential hue, light -> dark


def run_example_tiles(cfg: dict, percentiles=(90, 50, 10), width_in: float = 7.16, pdf: bool = False) -> dict:
    """QD-A3: dense / medium / sparse tile, the 4 input layers and the labels on the sub-tile grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    from qc_tiles import TileReader, owned_cells

    reader = TileReader(cfg, inputs=("wac", "dem"))
    tile_px, stride_px, cell_px = reader.tile_px, reader.stride_px, reader.cell_px
    tiles = pd.read_csv(cfg["paths"]["output"] / "tiles.csv")
    craters = pd.read_csv(cfg["paths"]["output"] / "craters_R.csv")
    lab = np.load(cfg["paths"]["output"] / "cell_labels.npz")
    cls, band = lab["cls"], lab["band"]
    bands = [float(x) for x in lab["bands_km"]]
    px_km = 2 * math.pi * MOON_RADIUS_M / 360.0 / region_layout(cfg, "wac").grid.ppd / 1000.0
    k, s_c = tile_px // cell_px, stride_px // cell_px

    # positive owned sub-tiles of every used tile
    used = tiles["kept"].to_numpy(bool) & reader.tile_used.ravel()
    n_pos = np.array([int((cls[owned_cells(a, b, tile_px, stride_px, cell_px)[1]] == 1).sum())
                      for a, b in zip(tiles["ti"], tiles["tj"])])
    rng = np.random.default_rng(int(cfg["seed"]))
    picks = []
    for q, name in zip(percentiles, ("dense", "medium", "sparse")):
        target = int(round(np.percentile(n_pos[used], q)))
        cand = np.flatnonzero(used & (n_pos == target))
        picks.append((name, q, int(rng.choice(cand))))

    cols = list(TILE_COLUMNS) + [(None, None, "labels (16 x 16 sub-tiles)")]
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(width_in, 1.5 * len(picks) + 0.55), facecolor=SURFACE,
                             squeeze=False)
    listed, any_ignore, any_large = {}, False, False
    for row_axes, (name, q, t_idx) in zip(axes, picks):
        t = tiles.iloc[t_idx]
        ti, tj, r0, c0 = int(t.ti), int(t.tj), int(t.row0), int(t.col0)
        img = reader.read(ti, tj, unit=True)["inputs"]
        sel = craters[craters["row"].between(r0 - 0.5, r0 + tile_px - 0.5) & craters["col"].between(c0 - 0.5, c0 + tile_px - 0.5)]
        win = (slice(ti * s_c, ti * s_c + k), slice(tj * s_c, tj * s_c + k))      # the 16 x 16 cells of the tile
        tc, tb = cls[win], band[win]
        any_ignore |= bool((tc == -1).any())
        any_large |= bool((sel["diam_km"] >= bands[-1]).any())
        (lr, lc), (gr, gc) = owned_cells(ti, tj, tile_px, stride_px, cell_px)
        owned_band = band[gr, gc][cls[gr, gc] == 1]
        listed[name] = {"percentile": q, "tile": int(t.tile), "set": t.set,
                        "lat_lon": [round(float(t.lat_centre), 2), round(float(t.lon_centre), 2)],
                        "positive_owned_per_band_1_2_4_8_km": [int((owned_band == b).sum()) for b in range(len(bands) - 1)]}
        for ax, (inp, ch, title) in zip(row_axes, cols):
            if inp is not None:
                ax.imshow(img[inp][ch], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            else:
                # labels: dimmed WAC, sub-tile classes, owned block, catalogue craters
                ax.imshow(img["wac"][0], cmap="gray", vmin=-0.3, vmax=1.3, interpolation="nearest")
                ext = (-0.5, tile_px - 0.5, tile_px - 0.5, -0.5)
                rgba = np.zeros((k, k, 4))
                for b, colour in enumerate(BAND_COLORS):
                    rgba[(tc == 1) & (tb == b)] = to_rgba(colour, 1.0)
                rgba[tc == -1] = to_rgba(INK_MUTED, 0.8)
                ax.imshow(rgba, extent=ext, interpolation="nearest")
                dim = np.zeros((k, k, 4))
                dim[...] = to_rgba(SURFACE, 0.7)
                dim[lr, lc] = 0.0
                ax.imshow(dim, extent=ext, interpolation="nearest")
                for e in range(0, tile_px + 1, cell_px):
                    ax.axhline(e - 0.5, color=SURFACE, lw=0.2, alpha=0.6)
                    ax.axvline(e - 0.5, color=SURFACE, lw=0.2, alpha=0.6)
                ax.add_patch(Rectangle((lc.start * cell_px - 0.5, lr.start * cell_px - 0.5), (lc.stop - lc.start) * cell_px,
                                       (lr.stop - lr.start) * cell_px, fill=False, ec=INK, lw=0.8, ls=(0, (3, 2))))
                draw_craters(ax, sel[sel["diam_km"] < bands[-1]], r0, c0, px_km, color=CRATER_COLOR, lw=0.6)
                draw_craters(ax, sel[sel["diam_km"] >= bands[-1]], r0, c0, px_km, color=CRATER_COLOR, lw=0.6, ls="--")
            ax.set_xlim(-0.5, tile_px - 0.5)
            ax.set_ylim(tile_px - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(INK_MUTED)
                sp.set_linewidth(0.5)
        row_axes[0].set_ylabel(f"{name}: {n_pos[t_idx]} of 121\npositive sub-tiles\n{t.lat_centre:.1f}°N {t.lon_centre:.1f}°E",
                               fontsize=6.5, color=INK)
    for ax, (_, _, title) in zip(axes[0], cols):
        ax.set_title(title, fontsize=7, color=INK)
    # scale bar: 10 km on the first panel
    bar = 10.0 / px_km
    axes[0][0].plot([12, 12 + bar], [tile_px - 16] * 2, color=SURFACE, lw=2.0, solid_capstyle="butt")
    axes[0][0].text(12 + bar / 2, tile_px - 24, "10 km", color=SURFACE, ha="center", va="bottom", fontsize=6)
    handles = [Patch(facecolor=c, label=f"positive, crater {a:g}-{b:g} km") for c, a, b in zip(BAND_COLORS, bands[:-1], bands[1:])]
    if any_ignore:                                                      # only the entries that appear
        handles.append(Patch(facecolor=to_rgba(INK_MUTED, 0.8), label=f"ignore (only ≥ {bands[-1]:g} km)"))
    handles += [Line2D([], [], color=INK, lw=0.8, ls=(0, (3, 2)), label="sub-tiles evaluated in this tile (11 x 11)"),
                Line2D([], [], color=CRATER_COLOR, lw=0.8, label="catalogue crater")]
    if any_large:
        handles.append(Line2D([], [], color=CRATER_COLOR, lw=0.8, ls="--", label=f"catalogue crater ≥ {bands[-1]:g} km"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.5, labelcolor=INK)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.95, bottom=0.14, wspace=0.04, hspace=0.06)
    save_figure(fig, paper_path(cfg, "fig_d3_example_tiles.png"), pdf=pdf)
    report = {"tiles": listed}
    save_report(cfg, "QD-A3", report)
    return report


# ---------------------------------------------------------------------------
# QD-A4 - Dataset table (Table D1)
# ---------------------------------------------------------------------------

# (modality, products key of the config, description, layers given to the backbone)
MODALITIES = [("Optical", "wac", "LROC WAC global mosaic, 643 nm", "reflectance (x3)"),
              ("Elevation", "dem", "SLDEM2015 (LOLA + Kaguya TC), 512 ppd", "relief, slope, hillshade")]


def run_dataset_table(cfg: dict) -> dict:
    """QD-A4: Table D1 (per set and per modality) as LaTeX, printed as Markdown."""
    import pandas as pd

    from qc_data import read_inventory

    counts = pd.DataFrame(load_report(cfg, "QC8.2")["per_set"]).set_index("set")
    blocks = pd.read_csv(cfg["paths"]["output"] / "split_blocks.csv")
    bands = [float(x) for x in cfg["labels"]["bands_km"]]
    band_cols = [f"positive_{a:g}_{b:g}_km" for a, b in zip(bands[:-1], bands[1:])]
    area = blocks.groupby("set")["area_km2"].sum()
    rows = []
    for name in list(SETS) + ["total"]:
        c = counts.loc[name]
        nb = int((blocks["set"] == name).sum()) if name != "total" else len(blocks)
        ar = float(area[name] if name != "total" else area.sum())
        rows.append({"Set": SET_STYLE[name][2] if name != "total" else "Total", "Blocks": nb,
                     "Area (%)": 100 * ar / area.sum(), "Tiles": int(c["tiles_used"]), "Sub-tiles": int(c["subtiles"]),
                     **{f"Pos. {a:g}-{b:g} km": int(c[k]) for (a, b), k in zip(zip(bands[:-1], bands[1:]), band_cols)},
                     "Negative": int(c["negative"]), "Ignore": int(c["ignore"]),
                     "Pos. (%)": 100 * float(c["positive_fraction_of_labelled"])})
    sets_df = pd.DataFrame(rows)

    inv = read_inventory(cfg)
    q = pd.read_csv(cfg["paths"]["quality_metrics"]).set_index("product_id")
    mods = []
    for label, key, desc, layers in MODALITIES:
        pids = cfg["products"][key]
        native = sorted({float(inv[p]["native_resolution_m"]) for p in pids})
        eff = sorted({float(inv[p]["native_resolution_m"]) / float(q.loc[p, "resolution_ratio"]) for p in pids})
        qs = q.loc[pids, "quality_score"].astype(float)
        mods.append({"Modality": label, "Product": desc, "Product files": len(pids),
                     "Native pixel (m)": "/".join(f"{x:.1f}" for x in native),
                     "Effective res. (m)": "/".join(f"{x:.0f}" for x in eff),
                     "Quality Q": f"{qs.min():.2f}" if qs.min().round(2) == qs.max().round(2) else f"{qs.min():.2f}-{qs.max():.2f}",
                     "Input layers (100 m grid)": layers})
    mod_df = pd.DataFrame(mods)

    def fmt(v):
        return f"{v:,.1f}" if isinstance(v, float) else (f"{v:,}" if isinstance(v, int) else str(v))

    def latex(df: pd.DataFrame, caption: str, label: str) -> str:
        cols = "l" + "r" * (len(df.columns) - 1) if df is sets_df else "llrrrrl"
        head = " & ".join(df.columns).replace("%", r"\%")
        body = []
        for r in df.itertuples(index=False):
            if df is sets_df and r[0] == "Total":
                body.append(r"\midrule")
            cells = (fmt(v).replace(",", "{,}") if not isinstance(v, str) else v.replace("(x3)", r"($\times$3)") for v in r)
            body.append(" & ".join(cells) + r" \\")
        return "\n".join([r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                          r"\footnotesize", rf"\begin{{tabular}}{{{cols}}}", r"\toprule", head + r" \\", r"\midrule",
                          *body, r"\bottomrule", r"\end{tabular}", r"\end{table*}"])

    tex = "\n\n".join([
        "% QD-A4 (Table D1), written by qc_figures.py dataset-table",
        latex(sets_df, "Dataset of the region R (30\\textdegree S-30\\textdegree N, 0-90\\textdegree E) per set: 5\\textdegree{} "
                       "blocks, tiles of 256 px used after the buffer, owned $16\\times16$ px sub-tiles (1.6 km) and their labels.",
              "tab:dataset_sets"),
        latex(mod_df, "Modalities: products, native pixel, effective resolution (native pixel / resolution ratio of the "
                      "spectral estimate) and quality score of the dataset diagnostics.", "tab:dataset_modalities")])
    out = paper_path(cfg, "table_d1_dataset.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex + "\n")
    written(out)

    def md(df):
        return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
                         + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])
    return {"output": str(out), "markdown": md(sets_df) + "\n\n" + md(mod_df)}


# ---------------------------------------------------------------------------
# QD-A5 - Complementarity of the modalities (Fig. D4)
# ---------------------------------------------------------------------------

# visibility categories of a crater rim (QD-A5): (key, label, colour = categorical slots 3 / 1 / 2, muted grey)
VISIBILITY = [("both", "both", "#1baf7a"), ("wac_only", "WAC only", "#2a78d6"), ("dem_only", "DEM only", "#eb6834"),
              ("neither", "neither", "#c3c2b7")]
# diverging map of the correlation: blue (negative) - neutral grey (0) - red (positive), poles of the reference ramps
CORR_CMAP = ("#104281", "#f0efec", "#a32b2b")


def cell_corr(a: np.ndarray, b: np.ndarray, cell_px: int) -> np.ndarray:
    """Pearson correlation of two images in every cell of cell_px x cell_px pixels (rows and
    columns beyond the last whole cell ignored); NaN if a cell has a non-finite pixel or no variance."""
    nr, nc = a.shape[0] // cell_px, a.shape[1] // cell_px

    def cells(x):
        x = np.asarray(x[:nr * cell_px, :nc * cell_px], dtype=np.float64)
        x = x.reshape(nr, cell_px, nc, cell_px).transpose(0, 2, 1, 3).reshape(nr, nc, -1)
        return x - x.mean(axis=-1, keepdims=True)

    x, y = cells(a), cells(b)
    den = np.sqrt((x * x).sum(-1) * (y * y).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (x * y).sum(-1) / den, np.nan)


def region_cell_corr(cfg: dict, azimuths: dict, strip_rows: int = 512) -> dict:
    """Per-cell correlation (QB2.2 cell grid) between the WAC and the hillshade of the DEM lit from
    each of ``azimuths`` {name: azimuth deg} (altitude of QC4.3), computed by strips over R."""
    from qc_preprocess import dem_gradient, hillshade, raster_path

    cell_px = int(cfg["tiling"]["cell_px"])
    alt = float(cfg["derivatives"]["hillshade_altitude_deg"])
    grid = region_layout(cfg, "wac").grid
    wac = np.load(raster_path(cfg, "wac"), mmap_mode="r")
    dem = np.load(raster_path(cfg, "dem"), mmap_mode="r")
    nr, nc = wac.shape[0] // cell_px, wac.shape[1] // cell_px
    out = {k: np.full((nr, nc), np.nan, dtype=np.float32) for k in azimuths}
    step = strip_rows - strip_rows % cell_px
    for r0 in range(0, nr * cell_px, step):
        rows = slice(r0, min(r0 + step, nr * cell_px))
        gx, gy = dem_gradient(dem, rows, grid)
        w = wac[rows]
        for k, az in azimuths.items():
            out[k][r0 // cell_px:rows.stop // cell_px] = cell_corr(w, hillshade(gx, gy, az, alt), cell_px)
    return out


def crater_visibility(cfg: dict, craters, rings: dict) -> "pd.DataFrame":
    """Rim contrast of every crater in the WAC and in the DEM, as z-scores against the local background:
    z_wac = (inner east wall - inner west wall) / std of the WAC in the background ring (Sun from the
    west, D14: the east wall faces the Sun); z_dem = (rim - floor relief) / std of the relief in the
    background ring.  ``rings`` gives the rho ranges (rho = distance / radius, E-W scaled by cos lat).
    Craters whose background ring leaves R or that contain WAC no-data get NaN."""
    import pandas as pd

    from qc_preprocess import raster_path

    wac = np.load(raster_path(cfg, "wac"), mmap_mode="r")
    relief = np.load(raster_path(cfg, "relief"), mmap_mode="r")
    px_km = pixel_km(region_layout(cfg, "wac").grid)
    lines, samples = wac.shape
    out_rho = rings["background"][1]
    z = np.full((len(craters), 2), np.nan)
    for i, k in enumerate(craters.itertuples()):
        rad = k.diam_km / px_km / 2.0
        cl = math.cos(math.radians(k.lat))
        hr, hc = math.ceil(out_rho * rad) + 1, math.ceil(out_rho * rad / cl) + 1
        r0, c0 = int(round(k.row)) - hr, int(round(k.col)) - hc
        if r0 < 0 or c0 < 0 or r0 + 2 * hr + 1 > lines or c0 + 2 * hc + 1 > samples:
            continue
        win = (slice(r0, r0 + 2 * hr + 1), slice(c0, c0 + 2 * hc + 1))
        w, h = np.asarray(wac[win], dtype=np.float64), np.asarray(relief[win], dtype=np.float64)
        if not np.isfinite(w).all():
            continue
        rho, dx = crater_polar(w.shape, k.row - r0, k.col - c0, rad, cl)
        ring = {n: (rho >= a) & (rho < b) for n, (a, b) in rings.items()}
        z[i, 0] = (w[ring["walls"] & (dx > 0)].mean() - w[ring["walls"] & (dx < 0)].mean()) / w[ring["background"]].std()
        z[i, 1] = (h[ring["rim"]].mean() - h[ring["floor"]].mean()) / h[ring["background"]].std()
    return pd.DataFrame({"z_wac": z[:, 0], "z_dem": z[:, 1]}, index=craters.index)


def visibility_classes(zw: np.ndarray, zd: np.ndarray, z_min: float) -> np.ndarray:
    """Category of each crater (keys of VISIBILITY): rim visible in a modality if its z >= z_min."""
    vw, vd = zw >= z_min, zd >= z_min
    return np.select([vw & vd, vw & ~vd, ~vw & vd], ["both", "wac_only", "dem_only"], "neither")


def corr_summary(r: np.ndarray) -> dict:
    r = r[np.isfinite(r)]
    q = np.percentile(r, [5, 25, 50, 75, 95])
    return {"subtiles": int(r.size), **{f"p{p}": round(float(v), 3) for p, v in zip((5, 25, 50, 75, 95), q)},
            "fraction_r_ge_0.5": round(float((r >= 0.5).mean()), 3), "fraction_abs_r_lt_0.2": round(float((np.abs(r) < 0.2).mean()), 3)}


def run_complementarity(cfg: dict, z_min: float = 2.0, z_sensitivity=(1.0, 3.0), width_in: float = 7.16,
                        pdf: bool = False) -> dict:
    """QD-A5: per sub-tile correlation WAC vs hillshade (lit like the WAC and lit from the north, the
    input layer), and rim visibility of the catalogue craters in the WAC and in the DEM."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from qc_data import cell_blocks
    from qc_tiles import dataset_cells

    out_dir = cfg["paths"]["output"]
    reg, cell_px = cfg["region"], int(cfg["tiling"]["cell_px"])
    grid = region_layout(cfg, "wac").grid
    tiles = pd.read_csv(out_dir / "tiles.csv")
    blocks = pd.read_csv(out_dir / "split_blocks.csv")
    lab = np.load(out_dir / "cell_labels.npz")
    cls, bands = lab["cls"], [float(x) for x in lab["bands_km"]]
    owner = np.load(out_dir / "split_cells.npz")["owner_tile"]
    _, in_data, _ = dataset_cells(tiles, np.load(out_dir / "tile_validity.npz")["tile_used"], owner)

    # (1) per sub-tile correlation over R, kept on the sub-tiles of the dataset
    az = {"wac_sun": float(cfg["derivatives"]["wac_sun_azimuth_deg"]), "north": float(cfg["derivatives"]["hillshade_azimuth_deg"])}
    r = region_cell_corr(cfg, az)
    r = {k: np.where(in_data, v, np.nan) for k, v in r.items()}
    n_c = round((reg["lon_max"] - reg["lon_min"]) / float(cfg["split"]["block_deg"]))
    stratum = blocks.set_index(blocks["bi"] * n_c + blocks["bj"])["stratum"]
    cell_stratum = stratum.reindex(cell_blocks(grid, cls.shape, cell_px, reg, float(cfg["split"]["block_deg"])).ravel()).to_numpy().reshape(cls.shape)
    strata = sorted(blocks["stratum"].unique())
    corr = {k: {"all": corr_summary(v[in_data]), "positive": corr_summary(v[in_data & (cls == 1)]),
                "negative": corr_summary(v[in_data & (cls == 0)]),
                **{f"stratum_{s}": corr_summary(v[in_data & (cell_stratum == s)]) for s in strata}}
            for k, v in r.items()}
    corr["undefined_subtiles"] = {k: int((in_data & ~np.isfinite(v)).sum()) for k, v in r.items()}
    corr["stratum_density_median_per_1000km2"] = {f"stratum_{s}": round(float(blocks.loc[blocks["stratum"] == s, "density_per_1000km2"].median()), 1)
                                                  for s in strata}
    # median per used tile, drawn on its owned sub-tiles
    m = in_data & np.isfinite(r["wac_sun"])
    tile_med = pd.Series(r["wac_sun"][m]).groupby(owner[m]).median()
    tile_map = np.full(cls.shape, np.nan)
    tile_map[m] = tile_med.reindex(owner[m]).to_numpy()

    # (2) rim visibility of the 1-8 km craters whose centre is in a sub-tile of the dataset
    d_lo, d_hi = (float(x) for x in cfg["labels"]["diam_range_km"])
    craters = pd.read_csv(out_dir / "craters_R.csv")
    craters = craters[craters["diam_km"].between(d_lo, d_hi, inclusive="left")]
    ci = np.floor((craters["row"].to_numpy() + 0.5) / cell_px).astype(int)      # cell of the centre, as QB2.2
    cj = np.floor((craters["col"].to_numpy() + 0.5) / cell_px).astype(int)
    inside = (ci >= 0) & (ci < cls.shape[0]) & (cj >= 0) & (cj < cls.shape[1])
    keep = np.zeros(len(craters), bool)
    keep[inside] = in_data[ci[inside], cj[inside]]
    craters = craters[keep].sort_values("row")                                   # row order: memory-map locality
    rings = {"floor": (0.0, 0.5), "walls": (0.5, 1.0), "rim": (0.9, 1.1), "background": (1.5, 2.5)}
    vis = crater_visibility(cfg, craters, rings)
    ok = vis.notna().all(axis=1).to_numpy()
    craters, vis = craters[ok], vis[ok]
    band = np.digitize(craters["diam_km"].to_numpy(), bands[1:-1])
    band_names = [f"{a:g}-{b:g}" for a, b in zip(bands[:-1], bands[1:])]

    def fractions(z0: float) -> dict:
        c = visibility_classes(vis["z_wac"].to_numpy(), vis["z_dem"].to_numpy(), z0)
        return {name: {k: round(float((c[band == b] == k).mean()), 3) for k, _, _ in VISIBILITY}
                for b, name in enumerate(band_names)} | {"all": {k: round(float((c == k).mean()), 3) for k, _, _ in VISIBILITY}}

    main = fractions(z_min)
    visibility = {
        "craters": {n: int((band == b).sum()) for b, n in enumerate(band_names)} | {"all": int(len(craters))},
        "craters_dropped_ring_outside_R_or_no_data": int((~ok).sum()), "rings_rho": rings, "z_min": z_min,
        "fractions_per_band": main,
        "median_z": {n: {"wac": round(float(vis["z_wac"][band == b].median()), 2), "dem": round(float(vis["z_dem"][band == b].median()), 2)}
                     for b, n in enumerate(band_names)},
        "spearman_z_wac_z_dem": round(float(vis["z_wac"].corr(vis["z_dem"], method="spearman")), 3),
        "sensitivity": {f"z_min_{z0:g}": {n: {k: v for k, v in f.items() if k in ("wac_only", "dem_only")} for n, f in fractions(z0).items()}
                        for z0 in z_sensitivity},
    }

    # figure: (a) map of the median r per tile, (b) histogram of r per sub-tile, (c) visibility per band
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(width_in, 2.9), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.0, 0.9], wspace=0.5)
    ax, hx, vx = (fig.add_subplot(gs[k]) for k in range(3))
    cmap = LinearSegmentedColormap.from_list("corr", CORR_CMAP)
    cmap.set_bad(SURFACE)
    lat0, lon0 = grid.rc_to_latlon(-0.5, -0.5)
    lat1, lon1 = grid.rc_to_latlon(cls.shape[0] * cell_px - 0.5, cls.shape[1] * cell_px - 0.5)
    im = ax.imshow(tile_map, cmap=cmap, vmin=-1, vmax=1, extent=[lon0, lon1, lat1, lat0], interpolation="nearest")
    for name, lat, lon, ha in MARIA:
        ax.text(lon, lat, f"Mare\n{name}", ha=ha, va="center", fontsize=5.5, color=INK, linespacing=0.9, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.75))
    ax.set_xlim(reg["lon_min"], reg["lon_max"])
    ax.set_ylim(reg["lat_min"], reg["lat_max"])
    ax.set_xticks(np.arange(reg["lon_min"], reg["lon_max"] + 1, 15))
    ax.set_yticks(np.arange(reg["lat_min"], reg["lat_max"] + 1, 10))
    ax.set_xlabel("longitude (°E)", color=INK_MUTED)
    ax.set_ylabel("latitude (°N)", color=INK_MUTED)
    ax.set_title("(a) median r per tile (hillshade lit like the WAC)", fontsize=7, color=INK, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=6, colors=INK_MUTED, length=2)
    cb.outline.set_linewidth(0.4)

    edges = np.linspace(-1, 1, 81)
    series = [(r["wac_sun"][in_data], "Sun from W (as WAC), all", SET_STYLE["s1_train"][0], "-", True),
              (r["wac_sun"][in_data & (cls == 1)], "Sun from W, positive", "#104281", "--", False),
              (r["north"][in_data], "Sun from N (input layer)", "#898781", "-", False)]
    for v, label, colour, ls, fill in series:
        v = v[np.isfinite(v)]
        h, _ = np.histogram(v, bins=edges, density=True)
        if fill:
            hx.stairs(h, edges, fill=True, color=colour, alpha=0.25, lw=0)
        hx.stairs(h, edges, color=colour, lw=1.0, ls=ls, label=label)
    hx.axvline(0, color=INK_MUTED, lw=0.5)
    hx.set_xlim(-1, 1)
    hx.set_ylim(0, 1.45 * hx.get_ylim()[1])                                  # room for the legend
    hx.set_xlabel("r(WAC, hillshade) per sub-tile", color=INK_MUTED)
    hx.set_ylabel("density", color=INK_MUTED)
    hx.set_title("(b) per sub-tile correlation", fontsize=7, color=INK, loc="left")
    hx.legend(loc="upper left", frameon=False, fontsize=5.5, labelcolor=INK, handlelength=1.6, borderaxespad=0.2)

    y = np.arange(len(band_names))
    left = np.zeros(len(band_names))
    for k, label, colour in VISIBILITY:
        f = np.array([main[n][k] for n in band_names])
        vx.barh(y, f, left=left, height=0.62, color=colour, edgecolor=SURFACE, lw=1.0, label=label)
        for yy, ll, ff in zip(y, left, f):
            if ff >= 0.08:
                vx.text(ll + ff / 2, yy, f"{100 * ff:.0f}", ha="center", va="center", fontsize=5.5,
                        color="#ffffff" if k != "neither" else INK)
        left += f
    vx.set_yticks(y)
    vx.set_yticklabels([f"{n} km\n{visibility['craters'][n]:,}" for n in band_names], color=INK)
    vx.set_ylim(len(band_names) - 0.5, -0.5)
    vx.set_xlim(0, 1)
    vx.set_xticks([0, 0.5, 1])
    vx.set_xticklabels(["0", "50", "100"])
    vx.set_xlabel("% of the craters", color=INK_MUTED)
    vx.set_title(f"(c) rim visible (z ≥ {z_min:g}) in", fontsize=7, color=INK, loc="left")
    vx.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, frameon=False, fontsize=5.5, labelcolor=INK,
              handlelength=1.0, columnspacing=0.8)
    for a in (ax, hx, vx):
        style_axes(a)
    for a in (hx, vx):
        a.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.91, bottom=0.27)
    save_figure(fig, paper_path(cfg, "fig_d4_complementarity.png"), pdf=pdf)
    report = {"hillshade_azimuth_deg": az, "hillshade_altitude_deg": float(cfg["derivatives"]["hillshade_altitude_deg"]),
              "correlation_per_subtile": corr, "rim_visibility": visibility}
    save_report(cfg, "QD-A5", report)
    return report


# ---------------------------------------------------------------------------
# QE1.2 - Check figure of the token grid
# ---------------------------------------------------------------------------


def draw_token_check(cfg: dict, tiles: list) -> None:
    """QE1.2 -> ``figures/qe_checks.png``: per tile (name, tile id, row of ``tiles.csv``, {"wac": WAC
    layer in [0, 1], "lab": 16 x 16 classes, "tokens": {input: 16 x 16 x D}}), the WAC with the
    catalogue craters and the PCA-RGB of the WAC and DEM tokens with the positive cells outlined."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    from qc_features import token_pca_rgb

    craters = pd.read_csv(cfg["paths"]["output"] / "craters_R.csv")
    tile_px, cell_px = int(cfg["tiling"]["tile_px"]), int(cfg["tiling"]["cell_px"])
    d_lo, d_hi = (float(x) for x in cfg["labels"]["diam_range_km"])
    px_km = pixel_km(region_layout(cfg, "wac").grid)
    ext = (-0.5, tile_px - 0.5, tile_px - 0.5, -0.5)
    cols = ["WAC 643 nm + catalogue craters", "PCA-RGB of the WAC tokens", "PCA-RGB of the DEM tokens"]
    fig, axes = plt.subplots(len(tiles), 3, figsize=(7.2, 2.45 * len(tiles) + 0.5), facecolor=SURFACE, squeeze=False)
    for row_axes, (name, tile, t, d) in zip(axes, tiles):
        r0, c0 = int(t.row0), int(t.col0)
        sel = craters[craters["row"].between(r0 - 0.5, r0 + tile_px - 0.5) & craters["col"].between(c0 - 0.5, c0 + tile_px - 0.5)
                      & craters["diam_km"].between(d_lo, d_hi, inclusive="left")]
        row_axes[0].imshow(d["wac"], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        draw_craters(row_axes[0], sel, r0, c0, px_km, color=CRATER_COLOR, lw=0.7)
        for ax, inp in zip(row_axes[1:], ("wac", "dem")):
            ax.imshow(token_pca_rgb(d["tokens"][inp]), extent=ext, interpolation="nearest")
            for i, j in np.argwhere(d["lab"] == 1):
                ax.add_patch(Rectangle((j * cell_px - 0.5, i * cell_px - 0.5), cell_px, cell_px, fill=False, ec="#ffffff", lw=0.9))
                ax.add_patch(Rectangle((j * cell_px - 0.5, i * cell_px - 0.5), cell_px, cell_px, fill=False, ec=INK, lw=0.4))
        for ax in row_axes:
            ax.set_xlim(-0.5, tile_px - 0.5)
            ax.set_ylim(tile_px - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            style_axes(ax)
        row_axes[0].set_ylabel(f"{name}: tile {tile}\n{t.lat_centre:.1f}°N {t.lon_centre:.1f}°E", fontsize=7, color=INK)
    for ax, title in zip(axes[0], cols):
        ax.set_title(title, fontsize=8, color=INK)
    handles = [Line2D([], [], color=CRATER_COLOR, lw=1.0, label=f"catalogue crater {d_lo:g}-{d_hi:g} km"),
               Rectangle((0, 0), 1, 1, fill=False, ec=INK, lw=0.8, label="positive sub-tile (label of the token)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=7, labelcolor=INK)
    fig.suptitle("QE1.2: token grid (16 x 16, one token = one 1.6 km sub-tile) against the labels, tiles of Fig. D3",
                 fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_figure(fig, figures_dir(cfg) / "qe_checks.png", dpi=130)


ATTENTION_COLOR = "#1f5fbf"   # one hue, opacity = attention (sequential)


def draw_attention_check(cfg: dict, tiles: list) -> None:
    """QF4.2 -> ``figures/qf_attention.png``: per tile (name, tile id, row of ``tiles.csv``, {"wac", "hill":
    layers in [0, 1], "lab": 16 x 16 classes, "query": (i, j), "attn": {"wac": WAC query over the DEM
    keys, "dem": DEM query over the WAC keys}, 16 x 16}), the WAC with the craters and the query, and
    the two last-block cross-attention maps of the query (head average, scaled to max 1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    craters = pd.read_csv(cfg["paths"]["output"] / "craters_R.csv")
    tile_px, cell_px = int(cfg["tiling"]["tile_px"]), int(cfg["tiling"]["cell_px"])
    d_lo, d_hi = (float(x) for x in cfg["labels"]["diam_range_km"])
    px_km = pixel_km(region_layout(cfg, "wac").grid)
    ext = (-0.5, tile_px - 0.5, tile_px - 0.5, -0.5)
    cmap = LinearSegmentedColormap.from_list("attn", [SURFACE, ATTENTION_COLOR])
    cols = ["WAC 643 nm, craters, query sub-tile", "WAC query → DEM keys (on the hillshade)", "DEM query → WAC keys (on the WAC)"]
    fig, axes = plt.subplots(len(tiles), 3, figsize=(7.2, 2.45 * len(tiles) + 0.7), facecolor=SURFACE, squeeze=False)
    for row_axes, (name, tile, t, d) in zip(axes, tiles):
        r0, c0 = int(t.row0), int(t.col0)
        sel = craters[craters["row"].between(r0 - 0.5, r0 + tile_px - 0.5) & craters["col"].between(c0 - 0.5, c0 + tile_px - 0.5)
                      & craters["diam_km"].between(d_lo, d_hi, inclusive="left")]
        qi, qj = d["query"]
        row_axes[0].imshow(d["wac"], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        draw_craters(row_axes[0], sel, r0, c0, px_km, color=CRATER_COLOR, lw=0.7)
        for ax, base, key in ((row_axes[1], d["hill"], "wac"), (row_axes[2], d["wac"], "dem")):
            ax.imshow(base, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            a = d["attn"][key] / d["attn"][key].max()
            rgba = np.zeros(a.shape + (4,))
            rgba[..., :3] = to_rgb(ATTENTION_COLOR)
            rgba[..., 3] = 0.85 * a
            ax.imshow(rgba, extent=ext, interpolation="nearest")
            for i, j in np.argwhere(d["lab"] == 1):
                ax.add_patch(Rectangle((j * cell_px - 0.5, i * cell_px - 0.5), cell_px, cell_px, fill=False, ec=INK, lw=0.4))
        for ax in row_axes:
            ax.add_patch(Rectangle((qj * cell_px - 0.5, qi * cell_px - 0.5), cell_px, cell_px, fill=False, ec="#ffffff", lw=1.6))
            ax.add_patch(Rectangle((qj * cell_px - 0.5, qi * cell_px - 0.5), cell_px, cell_px, fill=False, ec=INK, lw=0.7))
            ax.set_xlim(-0.5, tile_px - 0.5)
            ax.set_ylim(tile_px - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            style_axes(ax)
        row_axes[0].set_ylabel(f"{name}: tile {tile}\n{t.lat_centre:.1f}°N {t.lon_centre:.1f}°E", fontsize=7, color=INK)
    for ax, title in zip(axes[0], cols):
        ax.set_title(title, fontsize=8, color=INK)
    handles = [Line2D([], [], color=CRATER_COLOR, lw=1.0, label=f"catalogue crater {d_lo:g}-{d_hi:g} km"),
               Rectangle((0, 0), 1, 1, fill=False, ec=INK, lw=1.2, label="query sub-tile"),
               Rectangle((0, 0), 1, 1, fill=False, ec=INK, lw=0.4, label="positive sub-tile")]
    fig.legend(handles=handles, loc="lower left", ncol=3, frameon=False, fontsize=7, labelcolor=INK, bbox_to_anchor=(0.02, 0.0))
    fig.suptitle("QF4.2: last-block cross-attention of the trained F-X (head average), tiles of Fig. D3", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    cax = fig.add_axes((0.70, 0.022, 0.25, 0.012))
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), cax=cax, orientation="horizontal")
    cb.set_label("attention of the query (max = 1)", fontsize=7, color=INK)
    cb.ax.tick_params(labelsize=6, colors=INK_MUTED)
    cb.outline.set_visible(False)
    save_figure(fig, figures_dir(cfg) / "qf_attention.png", dpi=130)



# ---------------------------------------------------------------------------
# QJ1 / QL - quantum vs classical stage 2 (Fig. 2 of the paper, poster)
# ---------------------------------------------------------------------------

# colour = method family (first three slots of the categorical palette, validated all-pairs), marker / line style =
# the method inside the family: the three quantum maps share the blue
FAMILY_STYLE = {"QK-A": ("#2a78d6", "o", "-", "QSVC, QK-A"), "QK-Z": ("#2a78d6", "s", "--", "QSVC, QK-Z"),
                "QK-P": ("#2a78d6", "^", ":", "QSVC, QK-P"), "RBF": ("#eb6834", "D", "-", "RBF SVM"),
                "LIN": ("#1baf7a", "v", "-", "linear SVM")}
QJ_FIG_METHODS = ("QK-A", "QK-Z", "QK-P", "RBF", "LIN")


def quantum_classical_pairs(cfg: dict):
    """Every classical run of QJ1 matched with the quantum runs of QI4.3 / QI6.1 on the same sample (variant, d, n,
    seed): one row per pair with the test AP of both (chosen configurations only)."""
    import pandas as pd

    res = cfg["paths"]["results"]
    q = pd.concat([pd.read_csv(res / "qi4_runs.csv"), pd.read_csv(res / "qi6_runs.csv")], ignore_index=True)
    q = q[q["role"] == "chosen"][["variant", "map", "d", "n", "seed", "test_ap"]]
    c = pd.read_csv(res / "qj1_runs.csv")[["variant", "map", "d", "n", "seed", "test_ap"]]
    return q.merge(c, on=["variant", "d", "n", "seed"], suffixes=("_q", "_c"))


def run_quantum_vs_classical(cfg: dict, width_in: float = 7.16, pdf: bool = False) -> dict:
    """QJ1 figure -> ``paper/fig_j1_quantum_vs_classical.png`` (+ PDF) and report section QJ1-figure: (a) test AP vs
    qubits d (F-X, n = 500) and (b) vs training size n (F-X, best d) for the three QSVC maps, the RBF and the linear
    SVM, mean ± std over the seeds, stage 1 alone on the same candidates dashed, the linear probe of QG1 on the full
    vectors dash-dotted and, in (b), the linear SVM on the full vectors of the same samples dotted (QJ2); (c) paired difference of test AP
    classical - quantum on the same sample (every matched run of the grid, curve and representations); (d) F1 at
    probability 0.5 per representation at the best d (Table I).  Numbers of (a), (b), (d) from the runs CSV / reports
    of QI4-QJ1, the summary of the pairs of (c) saved in the report section."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.lines import Line2D

    res = cfg["paths"]["results"]
    main_n = int(cfg["stage2"]["main"]["n"])
    q = pd.read_csv(res / "qi4_runs.csv")
    q = q[q["role"] == "chosen"]
    c = pd.read_csv(res / "qj1_runs.csv")
    c = c[c["variant"] == cfg["stage2"]["main"]["variant"]]
    runs = pd.concat([q[["map", "d", "n", "seed", "test_ap"]], c[["map", "d", "n", "seed", "test_ap"]]], ignore_index=True)
    runs = runs[runs["map"].isin(QJ_FIG_METHODS)]
    qi6 = load_report(cfg, "QI6.1")
    qj1 = load_report(cfg, "QJ1")
    best_d = int(qj1["best_d"][0])
    s1_ap = qi6["variants"]["F-X"]["stage1_alone_qh2"]["subset_ap"]["all"]
    qj2 = load_report(cfg, "QJ2")                                       # reference lines without compression (QJ2)
    probe_ap = qj2["representations"]["F-X"]["qg1_probe_full_all_s1train"]["ap"]
    full_curve = {int(k[1:]): v["lin_full"][:2] for k, v in qj2["fx_learning_curve"].items()}

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "pdf.fonttype": 42, "axes.titlesize": 7.5})
    fig, axes = plt.subplots(2, 2, figsize=(width_in, 4.9), gridspec_kw={"hspace": 0.6, "wspace": 0.42})
    fig.patch.set_facecolor(SURFACE)
    (ax_d, ax_n), (ax_p, ax_r) = axes

    def curves(ax, sub, xkey, offsets, logx=False):
        for k, m in enumerate(QJ_FIG_METHODS):
            col, mk, ls, _ = FAMILY_STYLE[m]
            g = sub[sub["map"] == m].groupby(xkey)["test_ap"].agg(["mean", "std"]).reset_index()
            x = g[xkey].to_numpy(float)
            x = x * offsets[k] if logx else x + offsets[k]                      # dodge the error bars
            ax.errorbar(x, g["mean"], yerr=g["std"], color=col, marker=mk, ls=ls, lw=1.2, ms=4, capsize=1.5,
                        elinewidth=0.6, mfc=col if m in ("RBF", "LIN", "QK-A") else SURFACE, mec=col, mew=0.9, zorder=3)
        ax.axhline(s1_ap, color=INK_MUTED, ls=(0, (4, 2)), lw=0.9, zorder=1)
        ax.text(0.02, s1_ap - 0.006, f"stage 1 alone (AP {s1_ap:.2f})", transform=ax.get_yaxis_transform(),
                va="top", ha="left", color=INK_MUTED, fontsize=6.5)
        ax.axhline(probe_ap, color=INK_MUTED, ls=(0, (6, 2, 1, 2)), lw=0.9, zorder=1)
        ax.text(0.02, probe_ap - 0.006, f"linear probe, full vectors, all S1-train (AP {probe_ap:.2f})",
                transform=ax.get_yaxis_transform(), va="top", ha="left", color=INK_MUTED, fontsize=6.5)
        ax.set_ylim(0.28, 0.63)
        ax.set_ylabel("test AP (2,000 candidates)")
        ax.grid(axis="y", color="#e4e3df", lw=0.5, zorder=0)

    curves(ax_d, runs[runs["n"] == main_n], "d", [-0.6, -0.3, 0.0, 0.3, 0.6])
    ax_d.set_xticks([int(v) for v in cfg["quantum"]["dims"]])
    ax_d.set_xlabel("qubits = PCA components d")
    ax_d.set_title(f"(a) AP vs number of qubits (F-X, n = {main_n})", loc="left")
    curves(ax_n, runs[runs["d"] == best_d], "n", [0.9, 0.95, 1.0, 1.05, 1.1], logx=True)
    xs = sorted(full_curve)
    ax_n.errorbar(np.array(xs, float) * 1.15, [full_curve[k][0] for k in xs], yerr=[full_curve[k][1] for k in xs], color=INK_MUTED,
                  ls=(0, (1, 1.5)), marker="o", ms=3, mfc=SURFACE, mec=INK_MUTED, lw=1.0, elinewidth=0.6, capsize=1.2, zorder=2)
    ax_n.set_xscale("log")
    ns = [int(v) for v in cfg["qk_samples"]["sizes"]]
    ax_n.set_xticks(ns)
    ax_n.set_xticklabels([f"{v:,}" for v in ns])
    ax_n.minorticks_off()
    ax_n.set_xlabel("training samples n (log scale)")
    ax_n.set_title(f"(b) AP vs training size (F-X, d = {best_d})", loc="left")

    # (c) paired differences
    pairs = quantum_classical_pairs(cfg)
    rng = np.random.default_rng(int(cfg["seed"]))
    summary = {}
    for i, qm in enumerate(("QK-A", "QK-Z", "QK-P")):
        for j, cm in enumerate(("RBF", "LIN")):
            dv = pairs[(pairs["map_q"] == qm) & (pairs["map_c"] == cm)]
            diff = (dv["test_ap_c"] - dv["test_ap_q"]).to_numpy()
            y = i + (j - 0.5) * 0.34
            col, mk = FAMILY_STYLE[cm][0], FAMILY_STYLE[cm][1]
            ax_p.scatter(diff, y + rng.uniform(-0.08, 0.08, diff.size), s=5, color=col, alpha=0.35, lw=0, zorder=2)
            q1, med, q3 = np.percentile(diff, [25, 50, 75])
            ax_p.plot([q1, q3], [y, y], color=INK, lw=1.6, solid_capstyle="butt", zorder=3)
            ax_p.plot(med, y, marker=mk, color=col, ms=5, mec=INK, mew=0.6, zorder=4)
            summary[f"{cm}-{qm}"] = {"pairs": int(diff.size), "median": round(float(med), 4), "iqr": [round(float(q1), 4), round(float(q3), 4)],
                                     "mean": round(float(diff.mean()), 4), "share_classical_better": round(float(np.mean(diff > 0)), 3),
                                     "min_max": [round(float(diff.min()), 4), round(float(diff.max()), 4)]}
    ax_p.axvline(0, color=INK_MUTED, lw=0.8, zorder=1)
    ax_p.set_yticks(range(3))
    ax_p.set_yticklabels(["vs QK-A", "vs QK-Z", "vs QK-P"])
    ax_p.invert_yaxis()
    ax_p.set_xlabel("AP classical − AP quantum on the same training sample\n← quantum better        classical better →")
    n_pairs = summary["RBF-QK-A"]["pairs"]
    ax_p.set_title(f"(c) Paired AP difference ({n_pairs} samples each)", loc="left")
    ax_p.text(0.99, 0.98, "bar = IQR, marker = median", transform=ax_p.transAxes, ha="right", va="top", color=INK_MUTED,
              fontsize=6.3)
    ax_p.grid(axis="x", color="#e4e3df", lw=0.5, zorder=0)

    # (d) F1 per representation
    reps = ["M-W", "M-D", "F-E", "F-CM", "F-X", "F-C"]
    names = {"M-W": "WAC", "M-D": "DEM", "F-E": "early fus.", "F-CM": "concat+MLP", "F-X": "cross-attn",
             "F-C": "concat"}
    key = f"_d{best_d}"
    f1s = {}
    for v in reps:
        f1s[v] = {m: qi6["variants"][v][m + key]["f1"] for m in ("QK-A", "QK-Z", "QK-P")}
        f1s[v].update({m: qj1["representations"][v][m + key]["f1"] for m in ("RBF", "LIN")})
    for i, v in enumerate(reps):
        for k, m in enumerate(QJ_FIG_METHODS):
            col, mk, _, _ = FAMILY_STYLE[m]
            mean, std = f1s[v][m]
            y = i + (k - 2) * 0.13
            ax_r.errorbar(mean, y, xerr=std, color=col, marker=mk, ms=4, ls="none", elinewidth=0.6, capsize=1.2,
                          mfc=col if m in ("RBF", "LIN", "QK-A") else SURFACE, mec=col, mew=0.9, zorder=3)
    ax_r.set_yticks(range(len(reps)))
    ax_r.set_yticklabels([f"{v} {names[v]}" for v in reps])
    ax_r.set_xlabel("F1 at probability 0.5 (2,000 candidates)")
    ax_r.set_title(f"(d) F1 per representation (d = {best_d}, n = {main_n})", loc="left")
    ax_r.grid(axis="x", color="#e4e3df", lw=0.5, zorder=0)

    for ax in axes.ravel():
        style_axes(ax)
        ax.title.set_color(INK)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)
    handles = [Line2D([], [], color=FAMILY_STYLE[m][0], marker=FAMILY_STYLE[m][1], ls=FAMILY_STYLE[m][2], lw=1.2, ms=4,
                      mfc=FAMILY_STYLE[m][0] if m in ("RBF", "LIN", "QK-A") else SURFACE, mec=FAMILY_STYLE[m][0],
                      label=FAMILY_STYLE[m][3]) for m in QJ_FIG_METHODS]
    handles.append(Line2D([], [], color=INK_MUTED, ls=(0, (4, 2)), lw=0.9, label="stage 1 alone"))
    handles.append(Line2D([], [], color=INK_MUTED, ls=(0, (6, 2, 1, 2)), lw=0.9, label="linear probe, full vectors, all S1-train"))
    handles.append(Line2D([], [], color=INK_MUTED, ls=(0, (1, 1.5)), marker="o", ms=3, mfc=SURFACE, lw=1.0,
                          label="linear SVM, full vectors, same samples (b)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.998), fontsize=6.8,
               labelcolor=INK, handlelength=2.6)
    fig.subplots_adjust(top=0.875, bottom=0.1, left=0.07, right=0.985)
    out = paper_path(cfg, "fig_j1_quantum_vs_classical.png")
    save_figure(fig, out, pdf=pdf)
    rep = {"paired_difference": summary, "best_d": best_d, "grid_n": main_n, "stage1_subset_ap_fx": s1_ap,
           "note": ("paired test-AP difference classical - quantum over every sample (variant, d, n, seed) of QI4.3, QI6.1 "
                    "and QJ1 with both runs (chosen configurations); share_classical_better = share of pairs with a positive "
                    "difference; panels (a), (b) from qi4_runs.csv / qj1_runs.csv, (d) from the reports QI6.1 / QJ1")}
    save_report(cfg, "QJ1-figure", rep)
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["qb-checks", "tile-check", "region-map", "sfd", "example-tiles", "dataset-table", "complementarity",
                                            "quantum-vs-classical"])
    ap.add_argument("--pdf", action="store_true", help="also write the vector PDF of a paper figure")
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "qb-checks":
        print(run_qb_checks(cfg))
    elif args.command == "tile-check":
        print(json.dumps(run_tile_check(cfg), indent=1))
    elif args.command == "region-map":
        print(json.dumps(run_region_map(cfg, pdf=args.pdf), indent=1))
    elif args.command == "sfd":
        print(json.dumps(run_sfd(cfg, pdf=args.pdf), indent=1))
    elif args.command == "example-tiles":
        print(json.dumps(run_example_tiles(cfg, pdf=args.pdf), indent=1))
    elif args.command == "dataset-table":
        r = run_dataset_table(cfg)
        print(r["output"] + "\n\n" + r["markdown"])
    elif args.command == "complementarity":
        print(json.dumps(run_complementarity(cfg, pdf=args.pdf), indent=1))
    elif args.command == "quantum-vs-classical":
        print(json.dumps(run_quantum_vs_classical(cfg, pdf=args.pdf), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
