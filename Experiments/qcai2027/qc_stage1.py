#!/usr/bin/env python3
"""Stage 1 and candidate sampling of the QCAI 2027 experiment (plan phase QH).

Stage 1 is a light classical head per representation (QF1 variant) that scores every sub-tile and keeps
the likely craters as candidates for the quantum stage.  It reads the 16 x 16 map of a tile (from the
caches, ``qc_fusion.variant_maps``), so unlike the probes of QG it sees the neighbouring sub-tiles.

QH1 - Stage-1 head.  ``Stage1``: Conv 3x3 (D -> 128) -> ReLU -> Conv 3x3 (128 -> 1) on the map
standardised with the per-channel mean / std of S1-train.  Loss: binary cross-entropy with class-balanced
weights (as the probes) on the labelled **owned** sub-tiles of each tile (the central 11 x 11, QC7.2; the
border is context only, each cell counted once).  AdamW, early stopping on the S1-val AP of the owned
sub-tiles.  Hyper-parameters in ``stage1`` of ``qc_config.yaml``.

QH2 - Threshold.  ``run_threshold``: per head, the threshold on S1-val that keeps 95 % of the owned
positive sub-tiles; candidate fraction, recall per band, precision; trigger of QB2.5 (share of the false
positives next to a target centre or inside the circle of a 3-8 km target crater, ``confusion_zones``).

QH3 - Candidates.  ``run_candidates``: every head on the owned sub-tiles of QK-pool and Test with its QH2
threshold; ``stage1_scores.npz`` keeps the logits of all the owned sub-tiles and the candidate flags
(``load_candidates``), for the sampling of QH4-QH5 and the end-to-end metrics of QI.

QH4 - QK training samples.  ``run_qk_samples``: from the labelled QK-pool candidates of each variant,
balanced samples (50 % positives) of n in {100, 250, 500, 1000, 2000}, 5 seeds; positives over the bands in
proportion to the candidates with at least 10 % per band, and over the density terciles in proportion;
negatives over the terciles in proportion; nested across sizes (``qk_samples`` in ``qc_config.yaml``).

QH5 - Test candidates.  ``run_test_subset``: per variant 2,000 labelled test candidates, natural class
ratio, proportional over the same strata; the same subset for every stage-2 method of that variant.

Commands
--------
train        QH1 -> ``Dataset/Moon/qcai2027/stage1/<variant>.pt`` (weights, standardisation,
             curve) and report section QH1 of ``results/qh_stage1.json``: per variant the
             parameters, epochs, best epoch, S1-val AP overall and per band (against the
             linear probe of QG1), check of the samples against QG1.  ``--smoke``: 40 tiles
             per set, 2 variants, 2 epochs, nothing written.

threshold    QH2 -> report section QH2: per variant the S1-val threshold for the target
             recall, recall overall and per band, candidate fraction, precision, negatives
             removed, trigger of QB2.5 (false positives in the confusion zones against
             all negatives).  ``--smoke`` as above.

candidates   QH3 -> ``Dataset/Moon/qcai2027/stage1/stage1_scores.npz`` and report section
             QH3: per set (QK-pool, Test) and variant the candidates, recall per band,
             precision; Test: AP of stage 1 alone with block-bootstrap CI; QK-pool: labelled
             candidates per density tercile, class and band.  ``--smoke`` as above.

qk-samples   QH4 -> ``Dataset/Moon/qcai2027/stage1/qk_samples.npz`` and report section QH4:
             per variant the composition of the samples and the checks (candidates only,
             labelled, unique, balanced, nested).  Seconds, no model.

test-subset  QH5 -> ``Dataset/Moon/qcai2027/stage1/test_subset.npz`` and report section QH5:
             per variant the 2,000 test candidates, their composition against all the
             candidates, checks.  Seconds, no model.

Usage
-----
    python Experiments/qcai2027/qc_stage1.py train --device cuda      (GPU job: run_qcai_gpu.sh)
    python Experiments/qcai2027/qc_stage1.py threshold --device cpu   (CPU job: run_qcai_cpu.sh)
    python Experiments/qcai2027/qc_stage1.py candidates --device cpu  (CPU job: run_qcai_cpu.sh)
    python Experiments/qcai2027/qc_stage1.py qk-samples
    python Experiments/qcai2027/qc_stage1.py test-subset
    python Experiments/qcai2027/qc_stage1.py train --smoke --device cpu
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_data import CONFIG_PATH, load_config, load_report, save_report, written  # noqa: E402
from qc_fusion import VARIANTS, TokenStore, owned_index, tile_classes, variant_maps  # noqa: E402
from qc_probes import average_precision, band_masks  # noqa: E402


def stage1_dir(cfg: dict) -> Path:
    """Trained stage-1 heads: ``<output>/stage1/``."""
    return cfg["paths"]["output"] / "stage1"


def owned_mask(cfg: dict) -> np.ndarray:
    """(16, 16) bool: the sub-tiles owned by every tile (same local window for all tiles, QC7.2)."""
    from qc_tiles import owned_cells

    t = cfg["tiling"]
    (lr, lc), _ = owned_cells(0, 0, int(t["tile_px"]), int(t["stride_px"]), int(t["cell_px"]))
    g = int(t["tile_px"]) // int(t["cell_px"])
    m = np.zeros((g, g), bool)
    m[lr, lc] = True
    return m


def set_maps(cfg: dict, store: TokenStore, vid: str, set_name: str, device, max_tiles: int | None = None,
             chunk: int = 512):
    """Maps (n, 16, 16, D) float16 tensor on ``device`` of the tiles of ``set_name``, their cache rows and
    the class (n, 16, 16) of every sub-tile (QB2.3)."""
    import torch

    rows = np.flatnonzero(store.index["set"] == set_name)[:max_tiles]
    maps = torch.cat([torch.from_numpy(variant_maps(store, vid, rows[s:s + chunk]).astype(np.float16)).to(device)
                      for s in range(0, rows.size, chunk)])
    g = maps.shape[1]
    return maps, rows, tile_classes(cfg, store, rows).reshape(-1, g, g)


def build_stage1(in_dim: int, hidden: int):
    """QH1 head: Conv 3x3 (in_dim -> hidden) -> ReLU -> Conv 3x3 (hidden -> 1), zero padding (16 x 16 kept)."""
    from torch import nn

    return nn.Sequential(nn.Conv2d(in_dim, hidden, 3, padding=1), nn.ReLU(), nn.Conv2d(hidden, 1, 3, padding=1))


def stage1_logits(model, maps, mu, sd, batch: int = 256):
    """Logits (n, 16, 16) of ``model`` on the maps (n, 16, 16, D), standardised with ``mu`` / ``sd``."""
    import torch

    model.eval()
    with torch.no_grad():
        return torch.cat([model(((maps[s:s + batch].float() - mu) / sd).permute(0, 3, 1, 2))[:, 0]
                          for s in range(0, maps.shape[0], batch)])


def train_stage1(cfg: dict, train: tuple, val: tuple, own: np.ndarray, device, epochs: int | None = None) -> tuple:
    """QH1: one stage-1 head on the S1-train maps, early stopping on the S1-val AP of the owned labelled
    sub-tiles.  ``train`` / ``val`` = (maps, classes) of ``set_maps``.  Returns (model, mu, sd, info)."""
    import torch
    import torch.nn.functional as F

    p = cfg["stage1"]
    seed = int(p["seed"])
    x_tr, c_tr = train
    x_va, c_va = val
    n, g, _, d = x_tr.shape
    # per-channel standardisation over all the positions of the S1-train maps
    s1 = torch.zeros(d, dtype=torch.float64, device=device)
    s2 = torch.zeros(d, dtype=torch.float64, device=device)
    for s in range(0, n, 256):
        x = x_tr[s:s + 256].double().reshape(-1, d)
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
    cnt = n * g * g
    mu = (s1 / cnt).float()
    sd = (s2 / cnt - (s1 / cnt) ** 2).clamp(min=1e-12).sqrt().float().clamp(min=1e-6)
    own_t = torch.from_numpy(own).to(device)
    cls_tr = torch.from_numpy(c_tr.astype(np.int64)).to(device)
    lab_tr = own_t[None] & (cls_tr >= 0)                                  # cells in the loss
    n_lab, n_pos = int(lab_tr.sum()), int((lab_tr & (cls_tr == 1)).sum())
    w_pos, w_neg = n_lab / (2 * n_pos), n_lab / (2 * (n_lab - n_pos))     # class-balanced, as the probes
    sel_va = own[None] & (c_va >= 0)
    y_va = (c_va[sel_va] == 1).astype(int)

    torch.manual_seed(seed)
    model = build_stage1(d, int(p["hidden"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"]))
    gen = torch.Generator(device=device).manual_seed(seed)
    batch = int(p["batch"])
    best, curve = (-1.0, 0, None), []
    for epoch in range(int(epochs or p["epochs"])):
        model.train()
        perm = torch.randperm(n, generator=gen, device=device)
        run = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            z = model(((x_tr[idx].float() - mu) / sd).permute(0, 3, 1, 2))[:, 0]
            yb, mb = (cls_tr[idx] == 1).float(), lab_tr[idx]
            w = torch.where(yb > 0, w_pos, w_neg) * mb
            loss = (w * F.binary_cross_entropy_with_logits(z, yb, reduction="none")).sum() / mb.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item() * idx.numel()
        ap = average_precision(stage1_logits(model, x_va, mu, sd).cpu().numpy()[sel_va], y_va)
        curve.append([round(run / n, 5), round(ap, 4)])
        print(f"    epoch {epoch + 1:2d}  train loss {run / n:.4f}  S1-val AP {ap:.4f}", flush=True)
        if ap > best[0]:
            best = (ap, epoch + 1, copy.deepcopy(model.state_dict()))
        if epoch + 1 - best[1] >= int(p["patience"]):
            break
    model.load_state_dict(best[2])
    info = {"parameters": int(sum(q.numel() for q in model.parameters())), "train_cells_in_loss": n_lab,
            "train_positives": n_pos, "best_epoch": best[1], "epochs_run": len(curve),
            "val_ap": round(best[0], 4), "curve_train_loss_val_ap": curve}
    return model.eval(), mu, sd, info


def run_train(cfg: dict, device: str = "cuda", smoke: bool = False) -> dict:
    """QH1: a stage-1 head per variant, saved; S1-val AP overall and per band against the linear probe."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    variants = list(VARIANTS)[:2] if smoke else list(VARIANTS)
    mx, ep = (40, 2) if smoke else (None, None)
    store = TokenStore(cfg)
    own = owned_mask(cfg)
    qg1 = None if smoke else load_report(cfg, "QG1")["variants"]
    out = {}
    for vid in variants:
        t0 = time.perf_counter()
        x_tr, _, c_tr = set_maps(cfg, store, vid, "s1_train", device, mx)
        x_va, r_va, c_va = set_maps(cfg, store, vid, "s1_val", device, mx)
        load_s = time.perf_counter() - t0
        print(f"QH1 {vid}: {x_tr.shape[0]} S1-train / {x_va.shape[0]} S1-val tiles, {load_s:.0f} s", flush=True)
        model, mu, sd, info = train_stage1(cfg, (x_tr, c_tr), (x_va, c_va), own, device, ep)
        # S1-val metrics on the owned labelled sub-tiles, in the order of owned_index (= the QG samples)
        z = stage1_logits(model, x_va, mu, sd).cpu().numpy()
        idx = owned_index(cfg, store, "s1_val", mx)
        keep = idx["cls"] >= 0
        idx = {k: v[keep] for k, v in idx.items()}
        score = z[np.searchsorted(r_va, idx["row"]), idx["i"], idx["j"]]
        m = band_masks(idx)
        info["val_ap_per_band"] = {k: round(average_precision(score[s], idx["cls"][s]), 4) for k, s in m.items()}
        info["val_samples"] = int(keep.sum())
        if qg1 is not None:
            info["linear_probe_val_ap_qg1"] = qg1[vid]["val_ap"]
            info["check_val_samples_equal_qg1"] = bool(info["val_samples"] == qg1[vid]["samples_train_val_test"][1])
        info["seconds_loading_training"] = [round(load_s, 1), round(time.perf_counter() - t0 - load_s, 1)]
        out[vid] = info
        print(f"QH1 {vid}: best epoch {info['best_epoch']}, S1-val AP {info['val_ap']:.4f}", flush=True)
        if not smoke:
            d = stage1_dir(cfg)
            d.mkdir(parents=True, exist_ok=True)
            torch.save({"variant": vid, "state": {k: v.cpu() for k, v in model.state_dict().items()}, "mu": mu.cpu(),
                        "sd": sd.cpu(), "in_dim": int(x_tr.shape[-1]), "hidden": int(cfg["stage1"]["hidden"]), "info": info},
                       d / f"{vid}.pt")
            written(d / f"{vid}.pt")
        del x_tr, x_va
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    report = {"variants": out, "hyper_parameters": cfg["stage1"],
              "note": "loss and S1-val AP on the labelled owned sub-tiles (central 11 x 11); AP per band = positives of the band + all negatives"}
    if not smoke:
        save_report(cfg, "QH1", report)
    return report


# ---------------------------------------------------------------------------
# QH2 - Threshold for the target recall, candidate fraction, trigger of QB2.5
# ---------------------------------------------------------------------------


def load_stage1(cfg: dict, vid: str, device="cpu"):
    """Saved stage-1 head of ``vid`` (QH1): (model in eval mode, mu, sd)."""
    import torch

    ck = torch.load(stage1_dir(cfg) / f"{vid}.pt", map_location="cpu", weights_only=False)
    model = build_stage1(ck["in_dim"], ck["hidden"])
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), ck["mu"].to(device), ck["sd"].to(device)


def owned_scores(cfg: dict, store: TokenStore, vid: str, set_name: str, device, max_tiles: int | None = None):
    """Stage-1 logits of the owned sub-tiles of ``set_name`` (all of them, ignore cells included), in the
    order of ``owned_index``: (scores, index)."""
    model, mu, sd = load_stage1(cfg, vid, device)
    maps, rows, _ = set_maps(cfg, store, vid, set_name, device, max_tiles)
    z = stage1_logits(model, maps, mu, sd).cpu().numpy()
    idx = owned_index(cfg, store, set_name, max_tiles)
    return z[np.searchsorted(rows, idx["row"]), idx["i"], idx["j"]], idx


def confusion_zones(cfg: dict) -> dict:
    """Cells of R where centre-based labels can confuse stage 1 (risk R8, trigger of QB2.5):
    ``adjacent`` = 8-neighbours of a positive (target-centre) cell that are not positive themselves;
    ``in_circle`` = cells whose centre lies inside the circle (an ellipse on the equirectangular grid) of a
    target crater with ``trigger_min_diam_km`` <= D < 8 km and are not its centre cell."""
    import pandas as pd
    from scipy.ndimage import binary_dilation

    from qc_figures import pixel_km
    from qc_data import region_layout

    lab = np.load(cfg["paths"]["output"] / "cell_labels.npz")
    cls = lab["cls"]
    cell = int(cfg["tiling"]["cell_px"])
    pos = cls == 1
    adjacent = binary_dilation(pos, np.ones((3, 3), bool)) & ~pos
    cr = pd.read_csv(cfg["paths"]["output"] / "craters_R.csv")
    d_lo = float(cfg["stage1"]["trigger_min_diam_km"])
    d_hi = float(cfg["labels"]["diam_range_km"][1])
    cr = cr[cr["diam_km"].between(d_lo, d_hi, inclusive="left")]
    px = pixel_km(region_layout(cfg, "wac").grid)
    in_circle = np.zeros(cls.shape, bool)
    for k in cr.itertuples():
        ry = k.diam_km / 2 / px                                          # radius in pixels, N-S
        rx = ry / np.cos(np.radians(k.lat))                              # wider E-W
        i0, i1 = int((k.row - ry) // cell), int((k.row + ry) // cell) + 1
        j0, j1 = int((k.col - rx) // cell), int((k.col + rx) // cell) + 1
        i0, j0, i1, j1 = max(i0, 0), max(j0, 0), min(i1, cls.shape[0] - 1), min(j1, cls.shape[1] - 1)
        if i0 > i1 or j0 > j1:                                            # crater of the margin, outside R
            continue
        ci, cj = np.mgrid[i0:i1 + 1, j0:j1 + 1]
        yc, xc = (ci + 0.5) * cell - 0.5, (cj + 0.5) * cell - 0.5          # cell centres, pixels
        inside = ((yc - k.row) / ry) ** 2 + ((xc - k.col) / rx) ** 2 <= 1
        centre = (ci == int(k.row // cell)) & (cj == int(k.col // cell))
        in_circle[ci[inside & ~centre], cj[inside & ~centre]] = True
    return {"adjacent": adjacent, "in_circle": in_circle & ~pos, "craters_in_circle_rule": int(len(cr))}


def run_threshold(cfg: dict, device: str = "cpu", smoke: bool = False) -> dict:
    """QH2: per variant, the stage-1 threshold on S1-val for ``stage1.recall`` of the owned positive
    sub-tiles; at that threshold the candidate fraction (of all owned and of the labelled sub-tiles), the
    recall per band, the precision, and the trigger of QB2.5: share of the false positives in the
    confusion zones (``confusion_zones``) against the share of such cells among all the negatives."""
    target = float(cfg["stage1"]["recall"])
    variants = list(VARIANTS)[:2] if smoke else list(VARIANTS)
    mx = 40 if smoke else None
    store = TokenStore(cfg)
    zones = confusion_zones(cfg)
    out = {}
    for vid in variants:
        z, idx = owned_scores(cfg, store, vid, "s1_val", device, mx)
        lab = idx["cls"] >= 0
        y = idx["cls"] == 1
        thr = float(np.quantile(z[y], 1 - target, method="lower"))      # recall >= target
        cand = z >= thr
        neg, fp = idx["cls"] == 0, cand & (idx["cls"] == 0)
        zone = {k: zones[k][idx["ci"], idx["cj"]] for k in ("adjacent", "in_circle")}
        zone["either"] = zone["adjacent"] | zone["in_circle"]
        r = {"threshold_logit": round(thr, 4), "owned_subtiles": int(z.size), "labelled": int(lab.sum()),
             "positives": int(y.sum()),
             "recall": round(float(cand[y].mean()), 4),
             "recall_per_band_1_2_4_8_km": [round(float(cand[y & (idx["band"] == b)].mean()), 4) for b in range(3)],
             "candidates": int(cand.sum()),
             "candidate_fraction_of_owned": round(float(cand.mean()), 4),
             "candidate_fraction_of_labelled": round(float(cand[lab].mean()), 4),
             "ignore_cells_among_candidates": int((cand & (idx["cls"] == -1)).sum()),
             "precision_at_threshold": round(float(y[cand].sum() / cand[lab].sum()), 4),
             "negatives_removed_fraction": round(float(1 - cand[neg].mean()), 4),
             "trigger_qb25": {k: {"share_of_false_positives": round(float(v[fp].mean()), 4),
                                  "share_of_all_negatives": round(float(v[neg].mean()), 4),
                                  "ratio": round(float(v[fp].mean() / max(v[neg].mean(), 1e-12)), 2)}
                              for k, v in zone.items()}}
        r["trigger_fires"] = bool(r["trigger_qb25"]["either"]["share_of_false_positives"] > 0.30)
        curve = {}
        for rc in (float(v) for v in cfg["stage1"]["curve_recalls"]):          # trade-off recall vs candidates
            t_rc = float(np.quantile(z[y], 1 - rc, method="lower"))
            c = z >= t_rc
            curve[f"{rc:.2f}"] = {"threshold_logit": round(t_rc, 4), "candidate_fraction_of_owned": round(float(c.mean()), 4),
                                  "precision": round(float(y[c].sum() / c[lab].sum()), 4),
                                  "recall_per_band_1_2_4_8_km": [round(float(c[y & (idx["band"] == b)].mean()), 4) for b in range(3)]}
        r["recall_curve"] = curve
        out[vid] = r
        print(f"QH2 {vid}: recall {r['recall']:.3f}, candidates {r['candidate_fraction_of_owned']:.3f} of the owned, "
              f"precision {r['precision_at_threshold']:.3f}, FP in zones {r['trigger_qb25']['either']['share_of_false_positives']:.3f} "
              f"(negatives {r['trigger_qb25']['either']['share_of_all_negatives']:.3f})", flush=True)
    report = {"variants": out, "target_recall": target, "set": "s1_val",
              "confusion_zone_rule": {"adjacent": "8-neighbours of a positive cell, not positive",
                                      "in_circle": f"centre inside the circle of a target crater with {cfg['stage1']['trigger_min_diam_km']} <= D < 8 km, "
                                                   "not its centre cell, not positive (larger craters, up to basins, left out: their circles cover whole regions)",
                                      "craters_in_circle_rule": zones["craters_in_circle_rule"]},
              "trigger_rule": "fires if the share of the false positives in either zone is > 30 % (QB2.5; also compare with the share among all negatives)"}
    if not smoke:
        save_report(cfg, "QH2", report)
    return report


# ---------------------------------------------------------------------------
# QH3 - Stage 1 on QK-pool and Test: the candidates
# ---------------------------------------------------------------------------

CANDIDATE_SETS = ("qk_pool", "test")


def candidates_path(cfg: dict) -> Path:
    return stage1_dir(cfg) / "stage1_scores.npz"


def load_candidates(cfg: dict, set_name: str, vid: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """QH3 output for one set and variant: (stage-1 logits of all the owned sub-tiles, candidate flag, index
    {row, i, j, ci, cj, cls, band, block, tercile})."""
    z = np.load(candidates_path(cfg))
    idx = {k: z[f"{set_name}_{k}"] for k in ("row", "i", "j", "ci", "cj", "cls", "band", "block", "tercile")}
    return z[f"{set_name}_score_{vid}"], z[f"{set_name}_cand_{vid}"], idx


def run_candidates(cfg: dict, device: str = "cpu", smoke: bool = False) -> dict:
    """QH3: stage 1 of every variant on the owned sub-tiles of QK-pool and Test with the threshold of QH2.
    Saves the logits of all the owned sub-tiles and the candidate flags (``stage1_scores.npz``); reports per
    set and variant the candidates, recall per band, precision, the test AP of stage 1 alone (QJ2 reference,
    block-bootstrap CI) and, for QK-pool, the labelled candidates per class, band and density tercile (QH4)."""
    import pandas as pd

    from qc_data import cell_blocks, region_grid
    from qc_probes import block_bootstrap, bootstrap_ap, ci95

    variants = list(VARIANTS)[:2] if smoke else list(VARIANTS)
    mx = 40 if smoke else None
    thr = load_report(cfg, "QH2")["variants"]
    store = TokenStore(cfg)
    lab = np.load(cfg["paths"]["output"] / "cell_labels.npz")
    cb = cell_blocks(region_grid(cfg, "wac"), lab["cls"].shape, int(cfg["tiling"]["cell_px"]), cfg["region"],
                     float(cfg["split"]["block_deg"]))
    blocks_csv = pd.read_csv(cfg["paths"]["output"] / "split_blocks.csv")
    n_c = int(blocks_csv["bj"].max()) + 1
    stratum = dict(zip(blocks_csv["bi"] * n_c + blocks_csv["bj"], blocks_csv["stratum"]))
    arrays, out = {}, {}
    n_boot = 50 if smoke else int(cfg["probes"]["bootstrap"])
    for set_name in CANDIDATE_SETS:
        out[set_name] = {}
        idx = None
        for vid in variants:
            z, ix = owned_scores(cfg, store, vid, set_name, device, mx)
            if idx is None:
                idx = ix
                idx["block"] = cb[idx["ci"], idx["cj"]]
                idx["tercile"] = np.array([stratum[b] for b in idx["block"]], dtype=np.int8)
                for k in ("row", "i", "j", "ci", "cj", "cls", "band", "block", "tercile"):
                    arrays[f"{set_name}_{k}"] = idx[k]
                lab_m = idx["cls"] >= 0
                y = idx["cls"] == 1
                wts = block_bootstrap(idx["block"][lab_m], n_boot, int(cfg["seed"]))
            elif not np.array_equal(ix["row"], idx["row"]):
                raise ValueError("the owned sub-tiles differ between variants")
            cand = z >= float(thr[vid]["threshold_logit"])
            arrays[f"{set_name}_score_{vid}"] = z.astype(np.float32)
            arrays[f"{set_name}_cand_{vid}"] = cand
            r = {"owned_subtiles": int(z.size), "candidates": int(cand.sum()), "candidate_fraction": round(float(cand.mean()), 4),
                 "recall": round(float(cand[y].mean()), 4),
                 "recall_per_band_1_2_4_8_km": [round(float(cand[y & (idx["band"] == b)].mean()), 4) for b in range(3)],
                 "precision": round(float(y[cand].sum() / cand[lab_m].sum()), 4),
                 "labelled_candidates": {"positive": int((cand & y).sum()), "negative": int((cand & (idx["cls"] == 0)).sum()),
                                         "ignore": int((cand & (idx["cls"] == -1)).sum())}}
            if set_name == "test":
                ap_b = bootstrap_ap(z[lab_m], y[lab_m].astype(int), wts)
                r["stage1_alone_test_ap"] = round(average_precision(z[lab_m], y[lab_m].astype(int)), 4)
                r["stage1_alone_test_ap_ci95"] = ci95(ap_b)
            else:                                                        # what QH4 can sample from
                c = cand & lab_m
                r["candidates_per_tercile_class_band"] = {
                    f"tercile_{t}": {"negative": int((c & (idx["tercile"] == t) & (idx["cls"] == 0)).sum()),
                                     "positive_per_band_1_2_4_8_km": [int((c & (idx["tercile"] == t) & y & (idx["band"] == b)).sum())
                                                                      for b in range(3)]} for t in range(3)}
            out[set_name][vid] = r
            print(f"QH3 {set_name} {vid}: {r['candidates']} candidates ({r['candidate_fraction']:.3f}), recall {r['recall']:.3f}, "
                  f"precision {r['precision']:.3f}", flush=True)
    report = {"sets": out, "thresholds_of_qh2": {v: thr[v]["threshold_logit"] for v in variants},
              "file": "Dataset/Moon/qcai2027/stage1/stage1_scores.npz",
              "note": "candidates = owned sub-tiles with stage-1 logit >= the QH2 threshold (95 % S1-val recall); "
                      "recall / precision on the labelled owned sub-tiles; stage-1-alone test AP = reference of QJ2"}
    if smoke:
        return report
    np.savez(candidates_path(cfg), **arrays)
    written(candidates_path(cfg))
    save_report(cfg, "QH3", report)
    return report


# ---------------------------------------------------------------------------
# QH4 - QK training samples
# ---------------------------------------------------------------------------


def allocate(total: int, weights) -> np.ndarray:
    """Integer split of ``total`` proportional to ``weights`` (largest remainder)."""
    w = np.asarray(weights, dtype=float)
    q = total * w / w.sum()
    a = np.floor(q).astype(int)
    a[np.argsort(-(q - a), kind="stable")[:total - a.sum()]] += 1
    return a


def band_shares(counts, floor: float) -> np.ndarray:
    """Shares proportional to ``counts`` with at least ``floor`` each (the others scaled down)."""
    p = np.asarray(counts, dtype=float) / np.sum(counts)
    fixed = np.zeros(p.size, bool)
    while True:
        low = (~fixed) & (p < floor)
        if not low.any():
            return p
        fixed |= low
        p[fixed] = floor
        rest = ~fixed
        p[rest] = p[rest] / p[rest].sum() * (1 - floor * fixed.sum())


def qk_strata(cls: np.ndarray, band: np.ndarray, terc: np.ndarray, sel: np.ndarray) -> dict:
    """Members (indices) of the strata among the selected samples: ("pos", band, tercile), ("neg", tercile)."""
    st = {("pos", b, t): np.flatnonzero(sel & (cls == 1) & (band == b) & (terc == t)) for b in range(3) for t in range(3)}
    st.update({("neg", t): np.flatnonzero(sel & (cls == 0) & (terc == t)) for t in range(3)})
    return st


def stratum_sizes(n: int, strata: dict, pos_frac: float, floor: float) -> dict:
    """QH4 allocation: n * pos_frac positives split over the bands (``band_shares``) and, inside a band,
    over the terciles in proportion to the candidates; the negatives over the terciles in proportion."""
    n_pos = int(round(n * pos_frac))
    per_band = allocate(n_pos, band_shares([sum(len(strata["pos", b, t]) for t in range(3)) for b in range(3)], floor))
    out = {}
    for b in range(3):
        for t, k in enumerate(allocate(per_band[b], [len(strata["pos", b, t]) for t in range(3)])):
            out["pos", b, t] = int(k)
    for t, k in enumerate(allocate(n - n_pos, [len(strata["neg", t]) for t in range(3)])):
        out["neg", t] = int(k)
    return out


def run_qk_samples(cfg: dict) -> dict:
    """QH4: for every variant, size and seed a stratified sample of the labelled QK-pool candidates (indices
    into the QK-pool arrays of ``stage1_scores.npz``), saved in ``stage1/qk_samples.npz`` (key
    ``<variant>_n<size>_s<seed>``); checks: candidates only, labelled, unique, balanced, allocation met,
    nested across sizes."""
    q = cfg["qk_samples"]
    sizes, seeds = [int(v) for v in q["sizes"]], [int(v) for v in q["seeds"]]
    pf, floor = float(q["positive_fraction"]), float(q["min_band_share"])
    arrays, out = {}, {}
    for vid in VARIANTS:
        _, cand, idx = load_candidates(cfg, "qk_pool", vid)
        strata = qk_strata(idx["cls"], idx["band"], idx["tercile"], cand & (idx["cls"] >= 0))
        alloc = {n: stratum_sizes(n, strata, pf, floor) for n in sizes}
        checks = {"candidates_only": True, "labelled": True, "unique": True, "allocation_met": True, "nested": True}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            order = {k: rng.permutation(v) for k, v in strata.items()}        # one random order per stratum
            prev = None
            for n in sizes:
                if any(alloc[n][k] > len(order[k]) for k in order):
                    raise ValueError(f"{vid}: not enough candidates for n = {n}")
                s_idx = np.sort(np.concatenate([order[k][:alloc[n][k]] for k in order]))
                arrays[f"{vid}_n{n}_s{seed}"] = s_idx.astype(np.int32)
                checks["candidates_only"] &= bool(cand[s_idx].all())
                checks["labelled"] &= bool((idx["cls"][s_idx] >= 0).all())
                checks["unique"] &= bool(np.unique(s_idx).size == n)
                checks["allocation_met"] &= bool(s_idx.size == n and (idx["cls"][s_idx] == 1).sum() == round(n * pf))
                if prev is not None:
                    checks["nested"] &= bool(np.isin(prev, s_idx).all())
                prev = s_idx
        comp = {}
        for n in (sizes[0], sizes[-1]):
            s_idx = arrays[f"{vid}_n{n}_s{seeds[0]}"]
            c, b, t = idx["cls"][s_idx], idx["band"][s_idx], idx["tercile"][s_idx]
            comp[f"n{n}_seed{seeds[0]}"] = {"positives": int((c == 1).sum()), "negatives": int((c == 0).sum()),
                                           "positives_per_band_1_2_4_8_km": [int(((c == 1) & (b == k)).sum()) for k in range(3)],
                                           "positives_per_tercile": [int(((c == 1) & (t == k)).sum()) for k in range(3)],
                                           "negatives_per_tercile": [int(((c == 0) & (t == k)).sum()) for k in range(3)]}
        big = [set(arrays[f"{vid}_n{sizes[-1]}_s{s}"]) for s in seeds]
        out[vid] = {"labelled_candidates": {"positive": int(sum(len(strata["pos", b, t]) for b in range(3) for t in range(3))),
                                            "negative": int(sum(len(strata["neg", t]) for t in range(3)))},
                    "band_shares_of_the_positives": [round(float(v), 4) for v in band_shares(
                        [sum(len(strata["pos", b, t]) for t in range(3)) for b in range(3)], floor)],
                    "composition": comp,
                    "mean_overlap_between_seeds_largest_size": round(float(np.mean(
                        [len(big[a] & big[b]) / sizes[-1] for a in range(len(seeds)) for b in range(a + 1, len(seeds))])), 4),
                    "checks": checks}
        print(f"QH4 {vid}: {len(seeds) * len(sizes)} samples, checks {all(checks.values())}", flush=True)
    path = stage1_dir(cfg) / "qk_samples.npz"
    np.savez(path, **arrays)
    written(path)
    report = {"variants": out, "sizes": sizes, "seeds": seeds, "positive_fraction": pf, "min_band_share": floor,
              "file": "Dataset/Moon/qcai2027/stage1/qk_samples.npz",
              "note": "indices into the QK-pool arrays of stage1_scores.npz (load_candidates); strata = positives per band x "
                      "tercile, negatives per tercile; nested: every size of a seed contains the smaller ones"}
    save_report(cfg, "QH4", report)
    return report


# ---------------------------------------------------------------------------
# QH5 - Fixed test candidates
# ---------------------------------------------------------------------------


def run_test_subset(cfg: dict) -> dict:
    """QH5: per variant a fixed subset of ``test_subset.size`` labelled test candidates, proportional to the
    candidates in every stratum (positives per band x tercile, negatives per tercile: natural class ratio),
    saved in ``stage1/test_subset.npz`` (key = variant, indices into the Test arrays of
    ``stage1_scores.npz``); composition against the full candidate set, checks."""
    t = cfg["test_subset"]
    n, seed = int(t["size"]), int(t["seed"])
    arrays, out = {}, {}
    for vid in VARIANTS:
        _, cand, idx = load_candidates(cfg, "test", vid)
        sel = cand & (idx["cls"] >= 0)
        strata = qk_strata(idx["cls"], idx["band"], idx["tercile"], sel)
        keys = list(strata)
        alloc = dict(zip(keys, allocate(n, [len(strata[k]) for k in keys])))
        rng = np.random.default_rng(seed)
        s_idx = np.sort(np.concatenate([rng.choice(strata[k], alloc[k], replace=False) for k in keys if alloc[k]]))
        arrays[vid] = s_idx.astype(np.int32)
        c, b, tr = idx["cls"], idx["band"], idx["tercile"]

        def comp(m):
            y = c[m] == 1
            return {"positive_fraction": round(float(y.mean()), 4),
                    "band_shares_of_the_positives": [round(float((b[m][y] == k).mean()), 4) for k in range(3)],
                    "tercile_shares": [round(float((tr[m] == k).mean()), 4) for k in range(3)]}

        m_sub = np.zeros(c.size, bool)
        m_sub[s_idx] = True
        out[vid] = {"labelled_candidates": int(sel.sum()), "subset": int(s_idx.size),
                    "positives": int((c[s_idx] == 1).sum()),
                    "positives_per_band_1_2_4_8_km": [int(((c[s_idx] == 1) & (b[s_idx] == k)).sum()) for k in range(3)],
                    "test_blocks_covered": int(np.unique(idx["block"][s_idx]).size),
                    "composition_subset": comp(m_sub), "composition_all_candidates": comp(sel),
                    "checks": {"candidates_only": bool(cand[s_idx].all()), "labelled": bool((c[s_idx] >= 0).all()),
                               "unique": bool(np.unique(s_idx).size == n), "size": bool(s_idx.size == n)}}
        print(f"QH5 {vid}: {out[vid]['positives']} positives of {n}, checks {all(out[vid]['checks'].values())}", flush=True)
    path = stage1_dir(cfg) / "test_subset.npz"
    np.savez(path, **arrays)
    written(path)
    report = {"variants": out, "size": n, "seed": seed, "file": "Dataset/Moon/qcai2027/stage1/test_subset.npz",
              "note": "indices into the Test arrays of stage1_scores.npz; natural class ratio, proportional allocation over "
                      "positives per band x tercile and negatives per tercile (largest remainder); one subset per variant, "
                      "the same for every stage-2 method"}
    save_report(cfg, "QH5", report)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train", "threshold", "candidates", "qk-samples", "test-subset"])
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="40 tiles per set, 2 variants, 2 epochs, nothing written")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "train":
        r = run_train(cfg, args.device, args.smoke)
        for v in r["variants"].values():
            v.pop("curve_train_loss_val_ap")
        print(json.dumps(r, indent=1))
    elif args.command == "test-subset":
        print(json.dumps(run_test_subset(cfg), indent=1))
    elif args.command == "qk-samples":
        print(json.dumps(run_qk_samples(cfg), indent=1))
    elif args.command == "candidates":
        print(json.dumps(run_candidates(cfg, args.device, args.smoke), indent=1))
    elif args.command == "threshold":
        print(json.dumps(run_threshold(cfg, args.device, args.smoke), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
