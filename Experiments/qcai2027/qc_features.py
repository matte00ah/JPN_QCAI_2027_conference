#!/usr/bin/env python3
"""Backbone and token extraction of the QCAI 2027 experiment (plan phase QE).

QE1.1 - Backbone.  DINOv2 ViT-S/14 (``timm`` model ``qc_checks.DINO_MODEL``, the
weights of QA3.3), frozen (eval mode, no trainable parameter), one instance per
modality with the same weights.  A tile (``tiling.tile_px`` = 256 px, read by
``qc_tiles.TileReader`` as 3 layers in [0, 1], QC5.1) is resized in memory to
``backbone.img_size`` = 224 px (``backbone.resize``: bilinear with antialias; nothing
is written), then normalised with the mean / std of ``normalisation.json`` (QC5.1),
so the 14 px patches give a 16 x 16 grid of 384-d patch tokens = the sub-tile grid
(token (i, j) <-> cell (i, j) of the tile, 1.6 km).  The resize comes before the
normalisation: bilinear weights sum to 1, so the two commute (checked).

Commands
--------
backbone     QE1.1 check -> report section QE1.1 of ``results/qe_features.json``:
             the backbone is frozen; mean / std of QC5.1 equal to the pre-trained
             config; on ``n`` random used tiles (seed of the config) and every
             backbone input (WAC, DEM, early fusion): input range after the resize,
             resize and normalisation commute, token grid 16 x 16 x 384, finite,
             deterministic, same tokens in a batch and alone, time per tile.
alignment    QE1.2 check -> report section QE1.2 + ``results/figures/qe_checks.png``:
             (1) tokens with the positional embedding resampled once to 16 x 16 at
             load (``img_size`` 224) against ``timm``'s dynamic image size; (2)
             occlusion test on the 3 tiles of Fig. D3 + 5 random used tiles, every
             input: one 16 x 16 px source cell (the positive ones, then random
             others, 24 per tile) is set to the tile mean, and the token that
             changes most must be the token of that cell; (3) figure: WAC and the
             PCA-RGB of the WAC and DEM tokens of the 3 tiles of Fig. D3, with the
             positive cells and the catalogue craters.
extract      QE2.1 / QE2.2 -> ``<output>/tokens/``: patch tokens of the 12,790 used
             tiles (kept by the buffer and valid; the dropped tiles mix sets and are
             never used) for every backbone input (``--inputs``, default WAC, DEM and
             early fusion), one float16 memory-mapped ``<input>.npy`` (N, 16, 16, 384)
             each, ``index.npz`` (tile id, ti, tj, set of every row) and
             ``progress.npz`` (rows done: a stopped job resumes); checks in report
             sections QE2.1 (WAC, DEM) and QE2.2 (early): all rows done and finite,
             token norms, float16 rounding and device against CPU on 16 tiles.
             Runs on the GPU node (SLURM).
--selftest   synthetic checks (resize of a constant / of a ramp, token order: a
             change in one 14 px patch moves mostly the token at that position).

Usage
-----
    python Experiments/qcai2027/qc_features.py backbone [--n 8] [--device cpu|cuda]
    python Experiments/qcai2027/qc_features.py alignment [--device cpu|cuda]
    python Experiments/qcai2027/qc_features.py extract [--inputs wac dem early] [--batch 64] [--device cuda]
    python Experiments/qcai2027/qc_features.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_data import CONFIG_PATH, load_config, save_report  # noqa: E402


# ---------------------------------------------------------------------------
# QE1.1 - Backbone
# ---------------------------------------------------------------------------


def resize_unit(u, img_size: int, mode: str = "bilinear"):
    """(B, C, H, W) layers in [0, 1] (array or tensor) -> (B, C, img_size, img_size) float32 tensor,
    ``mode`` interpolation with antialias (QE1.1)."""
    import torch
    import torch.nn.functional as F

    x = u.float() if torch.is_tensor(u) else torch.from_numpy(np.array(u, dtype=np.float32))
    return F.interpolate(x, size=(img_size, img_size), mode=mode, antialias=True, align_corners=False)


class Backbone:
    """Frozen DINOv2 of QE1.1: tiles of 3 layers in [0, 1] -> 16 x 16 x 384 patch tokens."""

    def __init__(self, cfg: dict, device: str | None = None):
        import timm
        import torch

        from qc_checks import DINO_MODEL
        from qc_preprocess import load_normalisation

        bb = cfg["backbone"]
        self.img_size, self.mode = int(bb["img_size"]), str(bb["resize"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = timm.create_model(DINO_MODEL, pretrained=True, img_size=self.img_size).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.patch = int(self.model.patch_embed.patch_size[0])
        self.grid = self.img_size // self.patch
        self.dim = int(self.model.embed_dim)
        self.n_prefix = int(self.model.num_prefix_tokens)
        norm = load_normalisation(cfg)["backbone"]
        self.mean = torch.tensor(norm["mean"], dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(norm["std"], dtype=torch.float32).view(1, -1, 1, 1)

    def prepare(self, unit):
        """(B, 3, 256, 256) layers in [0, 1] -> (B, 3, 224, 224) normalised input of the backbone."""
        return (resize_unit(unit, self.img_size, self.mode) - self.mean) / self.std

    def tokens(self, unit):
        """(B, 3, 256, 256) layers in [0, 1] -> (B, 16, 16, 384) float32 patch tokens of the last layer
        (after the final norm, before the head; class token dropped); token (i, j) = patch row i, col j."""
        import torch

        with torch.inference_mode():
            f = self.model.forward_features(self.prepare(unit).to(self.device))[:, self.n_prefix:]
        return f.reshape(f.shape[0], self.grid, self.grid, self.dim).float().cpu()


def run_backbone_check(cfg: dict, n: int = 8, device: str | None = None) -> dict:
    """QE1.1: the frozen backbone on ``n`` random used tiles, for every backbone input."""
    import pandas as pd
    import torch

    from qc_preprocess import backbone_input
    from qc_tiles import TileReader

    bb = Backbone(cfg, device)
    reader = TileReader(cfg)
    tiles = pd.read_csv(cfg["paths"]["output"] / "tiles.csv")
    used = np.flatnonzero(tiles["kept"].to_numpy(bool) & reader.tile_used.ravel())
    pick = np.sort(np.random.default_rng(int(cfg["seed"])).choice(used, n, replace=False))
    units = {k: [] for k in reader.inputs}
    for t in pick:
        img = reader.read(int(tiles.at[t, "ti"]), int(tiles.at[t, "tj"]), unit=True)["inputs"]
        for k in units:
            units[k].append(img[k])
    pre = bb.model.pretrained_cfg
    report = {
        "model": pre.get("hf_hub_id"), "device": str(bb.device), "img_size": bb.img_size, "resize": f"{bb.mode}, antialias",
        "patch_px": bb.patch, "token_grid": [bb.grid, bb.grid], "token_dim": bb.dim, "prefix_tokens_dropped": bb.n_prefix,
        "check_frozen": bool(not bb.model.training and not any(p.requires_grad for p in bb.model.parameters())),
        "params_millions": round(sum(p.numel() for p in bb.model.parameters()) / 1e6, 3),
        "check_mean_std_equal_pretrained": bool(np.allclose(bb.mean.ravel(), pre["mean"]) and np.allclose(bb.std.ravel(), pre["std"])),
        "tiles": [int(tiles.at[t, "tile"]) for t in pick], "inputs": {},
    }
    for k, u in units.items():
        u = np.stack(u)                                                       # (n, 3, 256, 256) in [0, 1]
        r = resize_unit(u, bb.img_size, bb.mode).numpy()
        # resize and normalisation commute: resize(normalise(u)) == normalise(resize(u))
        x_norm_first = resize_unit(np.stack([backbone_input(list(a), bb.mean.ravel().tolist(), bb.std.ravel().tolist())
                                             for a in u]), bb.img_size, bb.mode)
        commute = float((x_norm_first - bb.prepare(u)).abs().max())
        t0 = time.perf_counter()
        tok = bb.tokens(u)
        dt = (time.perf_counter() - t0) / n
        again = bb.tokens(u)
        single = torch.cat([bb.tokens(u[i:i + 1]) for i in range(n)])
        norms = tok.norm(dim=-1)
        report["inputs"][k] = {
            "layers": reader.inputs[k], "tokens_shape": list(tok.shape),
            "resized_min_max": [round(float(r.min()), 4), round(float(r.max()), 4)],
            "max_abs_diff_resize_normalise_commute": float(f"{commute:.2e}"),
            "check_finite": bool(torch.isfinite(tok).all()),
            "max_abs_diff_repeat": float(f"{float((tok - again).abs().max()):.2e}"),
            "max_abs_diff_batch_vs_single": float(f"{float((tok - single).abs().max()):.2e}"),
            "token_norm_p5_p50_p95": [round(float(v), 2) for v in np.percentile(norms.numpy(), [5, 50, 95])],
            "seconds_per_tile": round(dt, 4),
        }
    save_report(cfg, "QE1.1", report)
    return report


# ---------------------------------------------------------------------------
# QE1.2 - Positional embedding and alignment of the token grid with the label grid
# ---------------------------------------------------------------------------


def token_pca_rgb(tok, q=(1, 99)) -> np.ndarray:
    """(16, 16, D) tokens of one tile -> (16, 16, 3) RGB in [0, 1]: first 3 principal components of
    the 256 tokens, each scaled by its percentiles ``q`` (visualisation only)."""
    x = np.asarray(tok, dtype=np.float64).reshape(-1, tok.shape[-1])
    x = x - x.mean(axis=0)
    pc = x @ np.linalg.svd(x, full_matrices=False)[2][:3].T
    lo, hi = np.percentile(pc, q[0], axis=0), np.percentile(pc, q[1], axis=0)
    return np.clip((pc - lo) / (hi - lo), 0, 1).reshape(*tok.shape[:2], 3)


def occlusion_offsets(bb: "Backbone", unit: np.ndarray, cells: list[tuple[int, int]], cell_px: int,
                      batch: int = 32) -> np.ndarray:
    """For each cell (i, j) of one tile (3, 256, 256 layers in [0, 1]): replace its cell_px x cell_px
    source pixels with the tile mean of each layer, and return the (row, col) offset of the token
    that changes most (L2 norm of the token difference) from (i, j), plus the ratio between the change
    of token (i, j) and the median change of the other tokens.  Rows: (d_row, d_col, ratio)."""
    import torch

    ref = bb.tokens(unit[None])[0]
    fill = unit.mean(axis=(1, 2))[:, None, None]
    out = []
    for s in range(0, len(cells), batch):
        part = cells[s:s + batch]
        x = np.repeat(unit[None], len(part), axis=0)
        for k, (i, j) in enumerate(part):
            x[k, :, i * cell_px:(i + 1) * cell_px, j * cell_px:(j + 1) * cell_px] = fill
        d = (bb.tokens(x) - ref).norm(dim=-1)                                # (b, 16, 16)
        for k, (i, j) in enumerate(part):
            a = np.unravel_index(int(d[k].argmax()), d[k].shape)
            others = torch.cat([d[k].ravel()[:i * bb.grid + j], d[k].ravel()[i * bb.grid + j + 1:]])
            out.append((a[0] - i, a[1] - j, float(d[k, i, j] / others.median())))
    return np.array(out)


def run_alignment_check(cfg: dict, n_random: int = 5, cells_per_tile: int = 24, device: str | None = None) -> dict:
    """QE1.2: (1) the positional embedding resampled at load (img_size 224) against timm's dynamic
    image size; (2) occlusion test of the alignment of the tokens with the label cells on the 3 tiles
    of Fig. D3 + ``n_random`` random used tiles, every backbone input; (3) the check figure
    ``figures/qe_checks.png`` (PCA-RGB of the tokens of the 3 tiles of Fig. D3 with the labels)."""
    import pandas as pd
    import timm
    import torch

    from qc_checks import DINO_MODEL
    from qc_data import load_report
    from qc_figures import draw_token_check
    from qc_tiles import TileReader

    bb = Backbone(cfg, device)
    reader = TileReader(cfg)
    tile_px, stride_px, cell_px = reader.tile_px, reader.stride_px, reader.cell_px
    k, s_c = tile_px // cell_px, stride_px // cell_px
    if k != bb.grid:
        raise ValueError(f"cell grid of a tile {k} x {k} != token grid {bb.grid} x {bb.grid}")
    tiles = pd.read_csv(cfg["paths"]["output"] / "tiles.csv")
    cls = np.load(cfg["paths"]["output"] / "cell_labels.npz")["cls"]
    examples = {name: int(v["tile"]) for name, v in load_report(cfg, "QD-A3")["tiles"].items()}
    rng = np.random.default_rng(int(cfg["seed"]))
    used = np.flatnonzero(tiles["kept"].to_numpy(bool) & reader.tile_used.ravel())
    used = np.setdiff1d(used, list(examples.values()))
    pick = list(examples.values()) + sorted(int(t) for t in rng.choice(used, n_random, replace=False))

    def read(t: int) -> tuple[dict, np.ndarray]:
        ti, tj = int(tiles.at[t, "ti"]), int(tiles.at[t, "tj"])
        return reader.read(ti, tj, unit=True)["inputs"], cls[ti * s_c:ti * s_c + k, tj * s_c:tj * s_c + k]

    # (1) fixed 224 px positional embedding vs dynamic image size (resampled at every forward)
    dyn = timm.create_model(DINO_MODEL, pretrained=True, dynamic_img_size=True).eval().to(bb.device)
    u = np.stack([read(t)[0]["wac"] for t in pick[:3]])
    with torch.inference_mode():
        f = dyn.forward_features(bb.prepare(u).to(bb.device))[:, dyn.num_prefix_tokens:].float().cpu()
    f = f.reshape(len(u), bb.grid, bb.grid, bb.dim)
    t_fix = bb.tokens(u)
    cos = torch.nn.functional.cosine_similarity(f, t_fix, dim=-1)
    pos = {"native_grid_of_the_weights": int(round((dyn.pos_embed.shape[1] - dyn.num_prefix_tokens) ** 0.5)),
           "token_grid": bb.grid, "max_abs_diff_tokens_fixed_vs_dynamic": float(f"{float((f - t_fix).abs().max()):.2e}"),
           "min_cosine_fixed_vs_dynamic": round(float(cos.min()), 5), "mean_cosine_fixed_vs_dynamic": round(float(cos.mean()), 5)}
    del dyn

    # (2) occlusion: positive cells first, then random others, up to cells_per_tile per tile
    occl, n_pos = {}, 0
    rows = {name: [] for name in reader.inputs}
    shown = {}
    for t in pick:
        img, lab = read(t)
        pos_cells = [tuple(c) for c in np.argwhere(lab == 1)][:cells_per_tile]
        rest = [tuple(c) for c in np.argwhere(lab != 1)]
        extra = [rest[i] for i in rng.choice(len(rest), cells_per_tile - len(pos_cells), replace=False)]
        cells = pos_cells + extra
        n_pos += len(pos_cells)
        for name in reader.inputs:
            o = occlusion_offsets(bb, img[name], cells, cell_px)
            rows[name].append(np.column_stack([o, np.r_[np.ones(len(pos_cells)), np.zeros(len(extra))]]))
        if t in examples.values():
            shown[t] = {"wac": img["wac"][0], "lab": lab,
                        "tokens": {name: bb.tokens(img[name][None])[0].numpy() for name in ("wac", "dem")}}
    for name, parts in rows.items():
        o = np.concatenate(parts)
        cheb = np.abs(o[:, :2]).max(axis=1)
        occl[name] = {"cells": int(len(o)), "fraction_argmax_at_the_cell": round(float((cheb == 0).mean()), 4),
                      "fraction_within_1_cell": round(float((cheb <= 1).mean()), 4),
                      "fraction_at_the_cell_positive_cells": round(float((cheb[o[:, 3] == 1] == 0).mean()), 4),
                      "change_ratio_cell_vs_median_other_p5_p50": [round(float(v), 1) for v in np.percentile(o[:, 2], [5, 50])],
                      "offsets_not_zero": sorted({(int(a), int(b)) for a, b in o[cheb > 0, :2]})[:10]}

    # (3) check figure
    tiles_fig = [(name, int(t), tiles.loc[t], shown[t]) for name, t in examples.items()]
    draw_token_check(cfg, tiles_fig)
    report = {"positional_embedding": pos,
              "occlusion": {"tiles": [int(tiles.at[t, "tile"]) for t in pick], "cells_per_tile": cells_per_tile,
                            "positive_cells": n_pos, "fill": "tile mean of each layer over one 16 x 16 px source cell",
                            "per_input": occl}}
    save_report(cfg, "QE1.2", report)
    return report


# ---------------------------------------------------------------------------
# QE2 - Token cache
# ---------------------------------------------------------------------------


def tokens_dir(cfg: dict) -> Path:
    """Token cache of QE2: ``<output>/tokens/``."""
    return cfg["paths"]["output"] / "tokens"


def load_tokens(cfg: dict, name: str) -> tuple[np.ndarray, dict]:
    """Cached tokens of one backbone input (``wac``, ``dem``, ``early``): memory-mapped float16
    (N, 16, 16, 384), and the index {tile, ti, tj, set} of the rows (row k = tile ``tile[k]``)."""
    d = tokens_dir(cfg)
    progress = np.load(d / "progress.npz")
    if name not in progress.files or not progress[name].all():
        raise ValueError(f"token cache of '{name}' incomplete: run qc_features.py extract")
    idx = np.load(d / "index.npz")
    return np.load(d / f"{name}.npy", mmap_mode="r"), {k: idx[k] for k in idx.files}


def run_extract(cfg: dict, inputs=("wac", "dem", "early"), batch: int = 64, device: str | None = None,
                n_check: int = 16, workers: int = 8) -> dict:
    """QE2.1 / QE2.2: patch tokens of every used tile (kept by the buffer and valid) and backbone input,
    written as float16 memory-mapped ``<output>/tokens/<input>.npy`` (N, 16, 16, 384) with
    ``index.npz``; resumable (``progress.npz``: rows done per input).  Checks on the full cache and
    on ``n_check`` random tiles (float16 rounding, device against CPU)."""
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd
    import torch

    from qc_data import written
    from qc_tiles import TileReader

    torch.backends.cuda.matmul.allow_tf32 = False                     # full float32 on the GPU
    torch.backends.cudnn.allow_tf32 = False
    bb = Backbone(cfg, device)
    reader = TileReader(cfg, inputs=tuple(inputs))
    tiles = pd.read_csv(cfg["paths"]["output"] / "tiles.csv")
    rows = np.flatnonzero(tiles["kept"].to_numpy(bool) & reader.tile_used.ravel())
    index = {"tile": tiles["tile"].to_numpy()[rows], "ti": tiles["ti"].to_numpy()[rows], "tj": tiles["tj"].to_numpy()[rows],
             "set": tiles["set"].to_numpy()[rows].astype(str)}
    n, shape = len(rows), (len(rows), bb.grid, bb.grid, bb.dim)
    d = tokens_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    if (d / "index.npz").exists():                                  # resume: same tiles in the same order
        old = np.load(d / "index.npz")
        if not all(np.array_equal(old[k], v) for k, v in index.items()):
            raise ValueError(f"{d / 'index.npz'} differs from the used tiles: remove the cache to rebuild it")
    else:
        np.savez(d / "index.npz", **index)
        written(d / "index.npz")
    progress = dict(np.load(d / "progress.npz")) if (d / "progress.npz").exists() else {}
    cache = {}
    for name in inputs:
        path = d / f"{name}.npy"
        if path.exists() and name in progress:
            cache[name] = np.load(path, mmap_mode="r+")
            if cache[name].shape != shape:
                raise ValueError(f"{path} has shape {cache[name].shape}, expected {shape}")
        else:
            cache[name] = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
            progress[name] = np.zeros(n, bool)

    todo = np.flatnonzero(~np.all([progress[k] for k in inputs], axis=0))
    print(f"{n} used tiles, {len(todo)} to extract, inputs {list(inputs)}, device {bb.device}", flush=True)
    read = lambda k: reader.read(int(index["ti"][k]), int(index["tj"][k]), unit=True)["inputs"]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        chunks = [todo[s:s + batch] for s in range(0, len(todo), batch)]
        nxt = pool.map(read, chunks[0]) if chunks else None
        for c, ks in enumerate(chunks):
            imgs = list(nxt)
            if c + 1 < len(chunks):
                nxt = pool.map(read, chunks[c + 1])                    # read the next batch while this one runs
            for name in inputs:
                cache[name][ks] = bb.tokens(np.stack([im[name] for im in imgs])).numpy().astype(np.float16)
                progress[name][ks] = True
            if c % 20 == 0 or c + 1 == len(chunks):
                for m in cache.values():
                    m.flush()
                np.savez(d / "progress.npz", **progress)
                done = int(np.all([progress[k] for k in inputs], axis=0).sum())
                print(f"  {done}/{n} tiles, {time.perf_counter() - t0:.0f} s", flush=True)
    seconds = time.perf_counter() - t0
    for name in inputs:
        written(d / f"{name}.npy")
    written(d / "progress.npz")

    # checks: (1) every row written and finite, token norms; (2) n_check random tiles recomputed in
    # float32: float16 rounding of the cache, and the same tiles on the CPU (if the device is a GPU)
    rng = np.random.default_rng(int(cfg["seed"]))
    chk = np.sort(rng.choice(n, n_check, replace=False))
    imgs = [read(k) for k in chk]
    cpu = Backbone(cfg, "cpu") if bb.device.type != "cpu" else None
    reports = {}
    for name in inputs:
        m = np.load(d / f"{name}.npy", mmap_mode="r")
        finite, norms = True, []
        for s in range(0, n, 512):
            x = np.asarray(m[s:s + 512], dtype=np.float32)
            finite &= bool(np.isfinite(x).all())
            norms.append(np.linalg.norm(x[:, ::4, ::4], axis=-1).ravel())
        u = np.stack([im[name] for im in imgs])
        f32 = bb.tokens(u)
        c16 = torch.from_numpy(np.asarray(m[chk], dtype=np.float32))
        cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1)
        r = {"layers": reader.inputs[name], "file": f"tokens/{name}.npy", "shape": list(m.shape), "dtype": str(m.dtype),
             "size_gb": round((d / f"{name}.npy").stat().st_size / 1e9, 2),
             "check_all_rows_done": bool(progress[name].all()), "check_finite": finite,
             "token_norm_p1_p50_p99": [round(float(v), 2) for v in np.percentile(np.concatenate(norms), [1, 50, 99])],
             "float16_max_abs_err": float(f"{float((c16 - f32).abs().max()):.2e}"),
             "float16_min_cosine": round(float(cos(c16, f32).min()), 6)}
        if cpu is not None:
            fc = cpu.tokens(u)
            r["device_vs_cpu_max_abs_diff"] = float(f"{float((fc - f32).abs().max()):.2e}")
            r["device_vs_cpu_min_cosine"] = round(float(cos(fc, f32).min()), 6)
        reports[name] = r
    common = {"tiles": n, "tiles_by_set": {s: int((index["set"] == s).sum()) for s in np.unique(index["set"])},
              "rows_extracted_in_this_run": int(len(todo)), "device": str(bb.device),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if bb.device.type == "cuda" else None,
              "batch": batch, "seconds": round(seconds, 1),
              "seconds_per_tile_all_inputs": round(seconds / max(len(todo), 1), 4), "check_tiles": n_check}
    out = {}
    for step, names in (("QE2.1", [k for k in inputs if k != "early"]), ("QE2.2", [k for k in inputs if k == "early"])):
        if names:
            out[step] = {**common, "inputs": {k: reports[k] for k in names}}
            save_report(cfg, step, out[step])
    return out


# ---------------------------------------------------------------------------
# Self-test (synthetic data)
# ---------------------------------------------------------------------------


def selftest() -> None:
    import torch

    # resize: a constant stays constant; a linear ramp stays within its range and keeps its mean
    c = resize_unit(np.full((1, 3, 256, 256), 0.3), 224)
    assert c.shape == (1, 3, 224, 224) and float((c - 0.3).abs().max()) < 1e-6
    ramp = np.broadcast_to(np.linspace(0, 1, 256, dtype=np.float32), (1, 3, 256, 256))
    r = resize_unit(ramp, 224)
    assert float(r.min()) >= 0.0 and float(r.max()) <= 1.0 and abs(float(r.mean()) - 0.5) < 1e-3

    # token order on the real model: a change in the 14 px patch (3, 10) of the 224 px input
    # moves the token at (3, 10) more than any other token
    cfg = load_config()
    bb = Backbone(cfg, "cpu")
    base = np.random.default_rng(0).uniform(0.3, 0.7, (1, 3, 256, 256)).astype(np.float32)
    x0 = bb.prepare(base)
    x1 = x0.clone()
    i, j, p = 3, 10, bb.patch
    x1[..., i * p:(i + 1) * p, j * p:(j + 1) * p] += 2.0
    with torch.inference_mode():
        f = [bb.model.forward_features(x)[:, bb.n_prefix:].reshape(bb.grid, bb.grid, bb.dim) for x in (x0, x1)]
    d = (f[1] - f[0]).norm(dim=-1)
    assert tuple(np.unravel_index(int(d.argmax()), d.shape)) == (i, j), d
    print("qc_features selftest: OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["backbone", "alignment", "extract"])
    ap.add_argument("--n", type=int, default=8, help="tiles of the QE1.1 check")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    ap.add_argument("--inputs", nargs="+", default=["wac", "dem", "early"], help="backbone inputs of extract")
    ap.add_argument("--batch", type=int, default=64, help="tiles per forward pass of extract")
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if args.command is None:
        ap.error("give a command or --selftest")
    cfg = load_config(args.config)
    if args.command == "backbone":
        print(json.dumps(run_backbone_check(cfg, args.n, args.device), indent=1))
    elif args.command == "alignment":
        print(json.dumps(run_alignment_check(cfg, device=args.device), indent=1))
    elif args.command == "extract":
        print(json.dumps(run_extract(cfg, tuple(args.inputs), args.batch, args.device), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
