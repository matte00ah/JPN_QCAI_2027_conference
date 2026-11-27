#!/usr/bin/env python3
"""Probes of the frozen representations of the QCAI 2027 experiment (plan phase QG, RQ1).

A sample is a labelled owned sub-tile (QC7.2, class 1 / 0 of QB2.3; the ignore cells, -1, are left
out) with the vector of one variant of QF1 (``qc_fusion.VARIANTS``, read from the caches by
``qc_fusion.owned_samples``).  Probes are fitted on S1-train, their hyper-parameter is chosen on
S1-val, the test set is evaluated once.  Uncertainty: block bootstrap of the test blocks (QB3.1).

QG1 - Linear probe.  ``fit_logistic``: logistic regression with class-balanced sample weights and an
L2 penalty on the weights (the objective of scikit-learn ``LogisticRegression(class_weight="balanced",
C)``: sum_i s_i loss_i + |w|^2 / (2 C)), full-batch L-BFGS in PyTorch (GPU).  Features standardised
with the mean / std of S1-train.  C chosen per variant on the S1-val AP.

QG2 - MLP probe.  ``fit_mlp``: d -> 256 -> 1 (ReLU), the same class-balanced loss, AdamW, early stopping
on the S1-val AP; the weight decay chosen on the S1-val AP with the first seed, then 3 seeds at that
value (test AP = mean over the seeds).  Hyper-parameters of both probes in ``probes`` of
``qc_config.yaml``; ``run_probe`` is the common protocol.

QG3 - Compressed probe.  ``fit_pca_probe``: PCA of the centred S1-train features (covariance,
``pca_fit``), the first d components standardised on S1-train, the linear probe of QG1 on them, for
d in {4, 8, 12, 16, 20} (the qubit grid of QI); d = 16 is the Table I value.

QG4 - Missing-modality robustness.  ``run_missing``: the probes of QG1 and QG3 (d = 16), fitted on
complete data, applied to test features with one modality missing: F-X and F-CM (both trained with
modality drop-out) with the mask token, F-C with the missing half set to the S1-train mean (with a linear
probe any constant fill shifts every score by the same amount, so zeros give the same AP: checked).

QG6 - Results per terrain.  ``run_terrain``: the saved test scores of QG1 and QG3 per crater-density
tercile of the test blocks (QB3.1), block bootstrap inside each tercile, fusion gain per tercile and its
low-minus-high difference.  No fit: CPU, reads only ``probes/qg1_test_scores.npz`` and ``qg3_test_scores.npz``.

Commands
--------
linear       QG1 -> report section QG1 of ``results/qg_probes.json`` and
             ``Dataset/Moon/qcai2027/probes/qg1_test_scores.npz`` (probe score of every
             labelled test sub-tile per variant, with its cell, labels and block): per
             variant the S1-val AP per C, the chosen C, test AP and AUROC overall and per
             diameter band, block-bootstrap 95 % CI of the AP and of the AP difference
             F-X minus every other variant; check of ``fit_logistic`` against scikit-learn.
             ``--smoke``: 40 tiles per set, 2 variants, 50 resamples, nothing written.

mlp          QG2 -> report section QG2: per variant the S1-val AP per weight decay (first
             seed), the chosen weight decay, the S1-val AP and best epoch of each seed, test
             AP (mean and std over the seeds) and AUROC overall and per band, block-bootstrap
             95 % CI and the AP difference F-X minus every other variant.  ``--smoke`` as above
             (3 epochs, 2 weight decays).

pca          QG3 -> report section QG3 and ``probes/qg3_test_scores.npz`` (scores at d = 16,
             for QG6): per variant and d the explained variance on S1-train, S1-val AP per C,
             test AP / AUROC overall and per band, CI, and per d the AP difference F-X minus
             every other variant.

missing      QG4 -> report section QG4: for the linear and the PCA-16 probe, test AP per band
             with CI of F-C, F-CM and F-X with nothing, the DEM or the WAC
             missing, retention, F-X minus each other row; checks: complete-data AP equal to
             QG1 / QG3, recomputed fused maps equal to the cache.

terrain      QG6 -> report section QG6: per density tercile of the test blocks, test AP
             with CI of every variant (linear and PCA-16), fusion gain F-X minus M-W /
             best single modality / F-C with CI, and the low-minus-high tercile difference
             of each gain.  ``--smoke``: 50 resamples, nothing written.

Usage
-----
    python Experiments/qcai2027/qc_probes.py linear --device cuda       (GPU job: run_qcai_gpu.sh)
    python Experiments/qcai2027/qc_probes.py mlp --device cuda          (GPU job: run_qcai_gpu.sh)
    python Experiments/qcai2027/qc_probes.py pca --device cuda          (GPU job: run_qcai_gpu.sh)
    python Experiments/qcai2027/qc_probes.py missing --device cuda      (GPU job: run_qcai_gpu.sh)
    python Experiments/qcai2027/qc_probes.py terrain                    (CPU job: run_qcai_cpu.sh)
    python Experiments/qcai2027/qc_probes.py linear --smoke --device cpu
    python Experiments/qcai2027/qc_probes.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_data import CONFIG_PATH, load_config, load_report, region_cell_blocks, save_report, written  # noqa: E402
from qc_fusion import VARIANTS, TokenStore, owned_samples, variant_ids  # noqa: E402

BANDS = ("1_2_km", "2_4_km", "4_8_km")


def probes_dir(cfg: dict) -> Path:
    """Probe outputs that later steps reuse: ``<output>/probes/``."""
    return cfg["paths"]["output"] / "probes"


def labelled_samples(cfg: dict, store: TokenStore, vid: str, set_name: str, max_tiles: int | None = None):
    """Vectors and index (``owned_samples``) of the labelled owned sub-tiles (class 0 / 1) of ``set_name``."""
    x, idx = owned_samples(cfg, store, vid, set_name, max_tiles=max_tiles)
    keep = idx["cls"] >= 0
    return x[keep], {k: v[keep] for k, v in idx.items()}


# ---------------------------------------------------------------------------
# Metrics and bootstrap
# ---------------------------------------------------------------------------


def average_precision(score: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None) -> float:
    """AP = sum over the positives, in decreasing score, of the precision at their rank / number of
    positives; with ``weight`` every sample counts ``weight`` times (block bootstrap)."""
    o = np.argsort(-score, kind="stable")
    y = y[o].astype(np.float64)
    w = np.ones_like(y) if weight is None else weight[o].astype(np.float64)
    tp = np.cumsum(w * y)
    n_pos = tp[-1]
    if n_pos == 0:
        return float("nan")
    return float(np.sum(w * y * tp / np.maximum(np.cumsum(w), 1e-12)) / n_pos)   # leading zero weights: term 0


def band_masks(idx: dict) -> dict:
    """Samples of each evaluation: all, and per band the positives of that band with all the negatives."""
    neg = idx["cls"] == 0
    out = {"all": np.ones(idx["cls"].size, bool)}
    for b, name in enumerate(BANDS):
        out[name] = neg | ((idx["cls"] == 1) & (idx["band"] == b))
    return out


def block_bootstrap(blocks: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_samples) weight of every sample in each resample: the test blocks drawn with replacement,
    a sample counts as many times as its block was drawn."""
    ub, inv = np.unique(blocks, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, ub.size, ub.size), minlength=ub.size) for _ in range(n_boot)])
    return counts[:, inv].astype(np.float32)


def bootstrap_ap(score: np.ndarray, y: np.ndarray, wts: np.ndarray, chunk: int = 100) -> np.ndarray:
    """AP (``average_precision``) of every resample of ``wts`` (n_boot, n): one sort, resamples in chunks."""
    o = np.argsort(-score, kind="stable")
    y = y[o].astype(np.float32)
    out = []
    for s in range(0, wts.shape[0], chunk):
        w = wts[s:s + chunk][:, o]
        tp = np.cumsum(w * y, axis=1, dtype=np.float64)
        out.append(np.sum(w * y * tp / np.maximum(np.cumsum(w, axis=1, dtype=np.float64), 1e-12), axis=1) / tp[:, -1])
    return np.concatenate(out)


def ci95(v: np.ndarray) -> list[float]:
    return [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]


# ---------------------------------------------------------------------------
# QG1 - Linear probe
# ---------------------------------------------------------------------------


def fit_logistic(x, y, C: float, max_iter: int = 1000):
    """Class-balanced L2 logistic regression on the tensors ``x`` (n, d), ``y`` (n,) in {0, 1}, full-batch
    L-BFGS (strong Wolfe).  Minimises mean_i s_i loss_i + |w|^2 / (2 C n), i.e. the objective of scikit-learn
    divided by n (same solution); s_i = n / (2 n_class).  Returns (w, b, info)."""
    import torch
    import torch.nn.functional as F

    n, d = x.shape
    n_pos = float(y.sum())
    sw = torch.where(y > 0, n / (2 * n_pos), n / (2 * (n - n_pos)))
    w = torch.zeros(d, device=x.device, dtype=x.dtype, requires_grad=True)
    b = torch.zeros(1, device=x.device, dtype=x.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, history_size=20, tolerance_grad=1e-7,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def objective():
        return (sw * F.binary_cross_entropy_with_logits(x @ w + b, y, reduction="none")).mean() + (w @ w) / (2 * C * n)

    def closure():
        opt.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    opt.step(closure)
    loss = objective()
    g = torch.autograd.grad(loss, [w, b])
    it = int(opt.state[w]["n_iter"])
    info = {"iterations": it, "converged": it < max_iter, "loss": loss.item(),
            "grad_norm": float(torch.cat([t.reshape(-1) for t in g]).norm())}
    return w.detach(), b.detach(), info


def standardise_stats(train):
    """(mean, std clamped at 1e-6) per feature of ``train`` (tensor)."""
    return train.mean(0), train.std(0).clamp(min=1e-6)


def standardise(train, *others):
    """Features standardised with the mean / std of ``train`` (tensors)."""
    mu, sd = standardise_stats(train)
    return [(t - mu) / sd for t in (train, *others)]


def sklearn_check(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray, C: float, device) -> dict:
    """``fit_logistic`` against scikit-learn on the same (standardised) data."""
    import torch
    from sklearn.linear_model import LogisticRegression

    t = lambda a: torch.from_numpy(a).to(device, torch.float64)
    w, b, info = fit_logistic(t(x_tr), t(y_tr.astype(np.float64)), C)
    sk = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, tol=1e-10).fit(x_tr, y_tr)
    z_t = (t(x_va) @ w + b).cpu().numpy()
    z_s = sk.decision_function(x_va)
    return {"samples_train_val": [int(y_tr.size), int(y_va.size)], "C": C, "torch_iterations": info["iterations"],
            "max_abs_diff_weights": float(f"{np.abs(w.cpu().numpy() - sk.coef_[0]).max():.1e}"),
            "max_abs_diff_val_logits": float(f"{np.abs(z_t - z_s).max():.1e}"),
            "ap_val_torch_sklearn": [round(average_precision(z_t, y_va), 5), round(average_precision(z_s, y_va), 5)],
            "check_equal_to_sklearn": bool(np.abs(z_t - z_s).max() < 1e-3 * max(1.0, np.abs(z_s).max()))}


def fit_mlp(x_tr, y_tr, x_va, y_va: np.ndarray, wd: float, seed: int, p: dict):
    """QG2: MLP probe d -> ``hidden`` -> 1 (ReLU), class-balanced BCE (sample weights as ``fit_logistic``),
    AdamW (``lr``, weight decay ``wd``), mini-batches of ``batch``; early stopping on the S1-val AP
    (``patience`` epochs), the weights of the best epoch kept.  Returns (model, info)."""
    import copy

    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    n, d = x_tr.shape
    model = torch.nn.Sequential(torch.nn.Linear(d, int(p["hidden"])), torch.nn.ReLU(), torch.nn.Linear(int(p["hidden"]), 1)).to(x_tr.device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(p["lr"]), weight_decay=wd)
    n_pos = float(y_tr.sum())
    sw = torch.where(y_tr > 0, n / (2 * n_pos), n / (2 * (n - n_pos)))
    g = torch.Generator(device=x_tr.device).manual_seed(seed)
    batch = int(p["batch"])
    best, curve = (-1.0, 0, None), []
    for epoch in range(int(p["epochs"])):
        model.train()
        perm = torch.randperm(n, generator=g, device=x_tr.device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            loss = (sw[idx] * F.binary_cross_entropy_with_logits(model(x_tr[idx])[:, 0], y_tr[idx], reduction="none")).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        ap = average_precision(predict(model, x_va), y_va)
        curve.append(round(ap, 4))
        if ap > best[0]:
            best = (ap, epoch + 1, copy.deepcopy(model.state_dict()))
        if epoch + 1 - best[1] >= int(p["patience"]):
            break
    model.load_state_dict(best[2])
    return model.eval(), {"val_ap": round(best[0], 4), "best_epoch": best[1], "epochs_run": len(curve), "val_ap_curve": curve}


def predict(model, x, batch: int = 65536) -> np.ndarray:
    """Logits (n,) of ``model`` on ``x`` in batches."""
    import torch

    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[s:s + batch])[:, 0] for s in range(0, x.shape[0], batch)]).cpu().numpy()


def fit_linear_probe(cfg: dict, x_tr, y_tr, x_va, y_va: np.ndarray) -> dict:
    """QG1: one logistic fit per C, the best S1-val AP kept.  Returns {"": ([predict function], details)}."""
    per_c, best = {}, None
    for C in (float(c) for c in cfg["probes"]["linear"]["Cs"]):
        w, b, info = fit_logistic(x_tr, y_tr, C)
        ap = average_precision((x_va @ w + b).cpu().numpy(), y_va)
        per_c[f"{C:g}"] = {"val_ap": round(ap, 4), **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in info.items()}}
        if best is None or ap > best[0]:
            best = (ap, C, w, b)
    w, b = best[2], best[3]
    return {"": ([lambda x: (x @ w + b).cpu().numpy()], {"per_C": per_c, "chosen_C": best[1], "val_ap": round(best[0], 4)})}


def fit_mlp_probe(cfg: dict, x_tr, y_tr, x_va, y_va: np.ndarray) -> dict:
    """QG2: weight decay chosen on the S1-val AP with the first seed, then the other seeds at that value.
    Returns {"": ([predict function per seed], details)}."""
    p = cfg["probes"]["mlp"]
    seeds = [int(s) for s in p["seeds"]]
    per_wd, models = {}, {}
    for wd in (float(v) for v in p["weight_decays"]):
        models[wd], per_wd[f"{wd:g}"] = fit_mlp(x_tr, y_tr, x_va, y_va, wd, seeds[0], p)
    wd = max(models, key=lambda v: per_wd[f"{v:g}"]["val_ap"])
    runs = [(seeds[0], models[wd], per_wd[f"{wd:g}"])] + [(s, *fit_mlp(x_tr, y_tr, x_va, y_va, wd, s, p)) for s in seeds[1:]]
    val = [r[2]["val_ap"] for r in runs]
    return {"": ([lambda x, m=m: predict(m, x) for _, m, _ in runs],
                 {"per_weight_decay_first_seed": per_wd, "chosen_weight_decay": wd, "seeds": seeds,
                  "val_ap_per_seed": val, "val_ap": round(float(np.mean(val)), 4),
                  "best_epoch_per_seed": [r[2]["best_epoch"] for r in runs]})}


def pca_fit(x_tr, d_max: int):
    """PCA of the centred features of ``x_tr`` (covariance, float64): (mean, first ``d_max`` components as
    columns, explained-variance fraction of each component)."""
    import torch

    mu = x_tr.mean(0)
    xc = (x_tr - mu).double()
    ev, vec = torch.linalg.eigh(xc.T @ xc / (x_tr.shape[0] - 1))
    ev, vec = ev.flip(0), vec.flip(1)
    return mu, vec[:, :d_max].to(x_tr.dtype), (ev / ev.sum())[:d_max].cpu().numpy()


def fit_pca_probe(cfg: dict, x_tr, y_tr, x_va, y_va: np.ndarray) -> dict:
    """QG3: PCA of the centred S1-train features (``pca_fit``), then for every d of ``probes.pca.dims`` the
    first d components, standardised on S1-train, and the linear probe of QG1 (C on the S1-val AP).
    Returns {"d<d>": ([predict function], details)}."""
    dims = [int(d) for d in cfg["probes"]["pca"]["dims"]]
    mu, vec, frac = pca_fit(x_tr, max(dims))
    proj = lambda x: (x - mu) @ vec
    p_tr, p_va = proj(x_tr), proj(x_va)
    out = {}
    for d in dims:
        a_tr, a_va = standardise(p_tr[:, :d], p_va[:, :d])
        mu_d, sd_d = standardise_stats(p_tr[:, :d])
        (pred,), det = fit_linear_probe(cfg, a_tr, y_tr, a_va, y_va)[""]
        out[f"d{d}"] = ([lambda x, d=d, m=mu_d, s=sd_d, f=pred: f((proj(x)[:, :d] - m) / s)],
                        {"explained_variance_s1_train": round(float(frac[:d].sum()), 4), **det})
    return out


# kind: (report step, fit function, features standardised before the fit)
PROBES = {"linear": ("QG1", fit_linear_probe, True), "mlp": ("QG2", fit_mlp_probe, True), "pca": ("QG3", fit_pca_probe, False)}


def run_probe(cfg: dict, kind: str, device: str = "cuda", smoke: bool = False) -> dict:
    """Probe ``kind`` (``PROBES``) of every variant: fitted on S1-train, hyper-parameter on S1-val, test AP and
    AUROC overall and per band (mean over the seeds of a stochastic probe, with their std), block-bootstrap
    95 % CI of the AP and of the AP difference F-X minus each variant (on the seed-averaged AP of every
    resample).  QG1 also saves the test scores (for QG6) and checks ``fit_logistic`` against scikit-learn."""
    import torch
    from sklearn.metrics import roc_auc_score

    step, fit, std_first = PROBES[kind]
    torch.backends.cuda.matmul.allow_tf32 = False                 # exact float32 for the probes
    variants = variant_ids(smoke)                                 # order of Table I: M-W, M-D, F-E, F-C, F-CM, F-X
    mx = 40 if smoke else None
    n_boot = 50 if smoke else int(cfg["probes"]["bootstrap"])
    if smoke and kind == "mlp":
        cfg = {**cfg, "probes": {**cfg["probes"], "mlp": {**cfg["probes"]["mlp"], "epochs": 3, "weight_decays": [1e-3, 1e-2]}}}
    store = TokenStore(cfg)
    lab = np.load(cfg["paths"]["output"] / "cell_labels.npz")
    cb = region_cell_blocks(cfg, lab["cls"].shape)
    out, scores, ref_idx = {}, {}, None                          # keyed by (variant, setting)
    for vid in variants:
        t0 = time.perf_counter()
        (x_tr, i_tr), (x_va, i_va), (x_te, i_te) = (labelled_samples(cfg, store, vid, s, mx) for s in ("s1_train", "s1_val", "test"))
        load_s = time.perf_counter() - t0
        if ref_idx is None:
            ref_idx = i_te
        elif not all(np.array_equal(ref_idx[k], i_te[k]) for k in ("ci", "cj")):
            raise ValueError("the test samples differ between variants")
        t = lambda a: torch.from_numpy(a).to(device)
        x_tr, x_va, x_te = (standardise if std_first else lambda *a: list(a))(t(x_tr), t(x_va), t(x_te))
        y_tr = t(i_tr["cls"].astype(np.float32))
        m = band_masks(i_te)
        for name, (predictors, details) in fit(cfg, x_tr, y_tr, x_va, i_va["cls"]).items():
            zs = [f(x_te).astype(np.float32) for f in predictors]                # one test score per seed
            scores[vid, name] = zs
            ap = {k: [average_precision(z[s], i_te["cls"][s]) for z in zs] for k, s in m.items()}
            r = {"dim": VARIANTS[vid].dim, "samples_train_val_test": [int(y_tr.numel()), int(i_va["cls"].size), int(zs[0].size)],
                 **details,
                 "test_ap": {k: round(float(np.mean(v)), 4) for k, v in ap.items()},
                 "test_auroc": {k: round(float(np.mean([roc_auc_score(i_te["cls"][s], z[s]) for z in zs])), 4) for k, s in m.items()}}
            if len(zs) > 1:
                r["test_ap_std_over_seeds"] = {k: round(float(np.std(v)), 4) for k, v in ap.items()}
            out[vid, name] = r
            print(f"{step} {vid} {name}: val AP {details['val_ap']:.4f}, test AP {r['test_ap']['all']:.4f}, "
                  f"AUROC {r['test_auroc']['all']:.4f}", flush=True)
        for name in [k[1] for k in out if k[0] == vid]:
            out[vid, name]["seconds_loading_fitting"] = [round(load_s, 1), round(time.perf_counter() - t0 - load_s, 1)]
        del x_tr, x_va, x_te, y_tr
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # block bootstrap of the test blocks: CI of the AP and of the AP difference F-X minus each variant
    blocks = cb[ref_idx["ci"], ref_idx["cj"]]
    wts = block_bootstrap(blocks, n_boot, int(cfg["seed"]))
    m = band_masks(ref_idx)
    y = ref_idx["cls"]
    boot = {key: {k: np.mean([bootstrap_ap(z[s], y[s], wts[:, s]) for z in zs], axis=0) for k, s in m.items()}
            for key, zs in scores.items()}
    for key in out:
        out[key]["test_ap_ci95"] = {k: ci95(v) for k, v in boot[key].items()}
    ref = "F-X" if "F-X" in variants else variants[-1]
    names = list(dict.fromkeys(k[1] for k in out))
    diff = {name: {vid: {k: {"ap_difference": round(out[ref, name]["test_ap"][k] - out[vid, name]["test_ap"][k], 4),
                             "ci95": ci95(boot[ref, name][k] - boot[vid, name][k]),
                             "fraction_resamples_ref_better": round(float(np.mean(boot[ref, name][k] > boot[vid, name][k])), 3)}
                         for k in m} for vid in variants if vid != ref} for name in names}
    if names == [""]:                                                # one setting: flat layout (QG1, QG2)
        out, diff = {vid: out[vid, ""] for vid in variants}, diff[""]
    else:                                                            # several settings (QG3: one per d)
        out = {vid: {name: out[vid, name] for name in names} for vid in variants}
    report = {"variants": out, "reference": ref, f"difference_{ref}_minus": diff,
              "test_blocks": int(np.unique(blocks).size), "bootstrap_resamples": n_boot, "bootstrap_seed": int(cfg["seed"]),
              "hyper_parameters": cfg["probes"][kind],
              "note": "AP per band = positives of the band + all negatives; ignore cells (-1) left out"}
    if kind == "linear":
        report["fit_logistic_vs_sklearn"] = linear_sklearn_check(cfg, store, device, mx)
    if smoke:
        return report
    if kind in ("linear", "pca"):                                  # test scores for QG6 (QG3: the Table I d)
        name = "" if kind == "linear" else f"d{int(cfg['probes']['pca']['table_dim'])}"
        d = probes_dir(cfg)
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{step.lower()}_test_scores.npz"
        np.savez(f, **{f"score_{v}": scores[v, name][0] for v in variants}, ci=ref_idx["ci"], cj=ref_idx["cj"],
                 cls=ref_idx["cls"], band=ref_idx["band"], block=blocks, tile_row=ref_idx["row"])
        written(f)
    save_report(cfg, step, report)
    return report


def linear_sklearn_check(cfg: dict, store: TokenStore, device, mx) -> dict:
    """``fit_logistic`` against scikit-learn: M-W, 20,000 S1-train and 20,000 S1-val samples, C = 1e-2."""
    rng = np.random.default_rng(int(cfg["seed"]))
    (x_tr, i_tr), (x_va, i_va) = (labelled_samples(cfg, store, "M-W", s, mx) for s in ("s1_train", "s1_val"))
    a = rng.choice(i_tr["cls"].size, min(20000, i_tr["cls"].size), replace=False)
    c = rng.choice(i_va["cls"].size, min(20000, i_va["cls"].size), replace=False)
    mu, sd = x_tr[a].mean(0), x_tr[a].std(0) + 1e-6
    return sklearn_check(((x_tr[a] - mu) / sd).astype(np.float64), i_tr["cls"][a], ((x_va[c] - mu) / sd).astype(np.float64),
                         i_va["cls"][c], 1e-2, device)


# ---------------------------------------------------------------------------
# QG4 - Missing-modality robustness
# ---------------------------------------------------------------------------

MISSING = ("none", "dem", "wac")                                   # modality missing at test time


def fused_samples(cfg: dict, store: TokenStore, vid: str, idx: dict, missing: str, device, batch: int = 256) -> np.ndarray:
    """Vectors (n, 512) of the trained fusion ``vid`` for the samples ``idx`` (``labelled_samples``), computed
    from the tokens with the modality ``missing`` fully masked (mask token of QF3.1), float32."""
    import torch

    from qc_fusion import MODALITIES, load_fusion

    enc = load_fusion(cfg, vid, device=device)
    rows = np.unique(idx["row"])
    n_tok = enc.grid ** 2
    maps = np.empty((rows.size, enc.grid, enc.grid, int(cfg["fusion"]["out_dim"])), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, rows.size, batch):
            r = rows[s:s + batch]
            tok = store.modality_tokens(r, device)
            masks = {m: torch.ones(r.size, n_tok, dtype=torch.bool, device=device) if m == missing else None for m in MODALITIES}
            maps[s:s + r.size] = enc(tok, masks).cpu().numpy()
    return maps[np.searchsorted(rows, idx["row"]), idx["i"], idx["j"]]


def run_missing(cfg: dict, device: str = "cuda", smoke: bool = False) -> dict:
    """QG4: the linear probe (QG1) and the PCA-16 probe (QG3), fitted on complete S1-train data with the C
    chosen there, applied to test features with the DEM or the WAC missing: F-X and F-CM with the mask token
    of the missing modality, F-C with the missing half set to the S1-train mean (zeros: same AP).  Test AP per
    band with block-bootstrap CI, retention (AP missing / AP complete), F-X minus each other row."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    mx, n_boot = (40, 50) if smoke else (None, int(cfg["probes"]["bootstrap"]))
    d16 = f"d{int(cfg['probes']['pca']['table_dim'])}"
    qg1, qg3 = load_report(cfg, "QG1")["variants"], load_report(cfg, "QG3")["variants"]
    store = TokenStore(cfg)
    lab = np.load(cfg["paths"]["output"] / "cell_labels.npz")
    cb = region_cell_blocks(cfg, lab["cls"].shape)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    half = int(cfg["fusion"]["in_dim"])                               # F-C = [WAC 384, DEM 384]
    scores, checks, idx_te = {}, {}, None
    for vid in ("F-C", "F-CM", "F-X"):
        (x_tr, i_tr), (x_va, i_va), (x_te, idx_te) = (labelled_samples(cfg, store, vid, s, mx) for s in ("s1_train", "s1_val", "test"))
        y_tr = t(i_tr["cls"].astype(np.float32))
        # probes fitted on the complete S1-train data, C of QG1 / QG3
        xt = t(x_tr)
        mu, sd = standardise_stats(xt)
        w1, b1, _ = fit_logistic((xt - mu) / sd, y_tr, float(qg1[vid]["chosen_C"]))
        pmu, vec, _ = pca_fit(xt, int(d16[1:]))
        p_tr = (xt - pmu) @ vec
        qmu, qsd = standardise_stats(p_tr)
        w3, b3, _ = fit_logistic((p_tr - qmu) / qsd, y_tr, float(qg3[vid][d16]["chosen_C"]))
        probe = {"linear": lambda x: (((x - mu) / sd) @ w1 + b1).cpu().numpy(),
                 "pca16": lambda x: ((((x - pmu) @ vec) - qmu) / qsd @ w3 + b3).cpu().numpy()}
        train_mean = x_tr.mean(0)
        del xt, p_tr
        rows = {}
        for miss in MISSING:
            if vid == "F-C":
                lo, hi = (half, 2 * half) if miss == "dem" else (0, half)
                for fill in ("zeros", "mean"):
                    x = x_te.copy()
                    if miss != "none":
                        x[:, lo:hi] = 0.0 if fill == "zeros" else train_mean[lo:hi]
                    rows["F-C" if fill == "mean" else "F-C zeros", miss] = x
            elif miss == "none":
                rows[vid, miss] = x_te                                    # cached fused maps, as in QG1 / QG3
                recomputed = fused_samples(cfg, store, vid, idx_te, "none", device)
                checks[f"{vid}_max_abs_diff_recomputed_vs_cache"] = float(f"{np.abs(recomputed - x_te).max():.1e}")
            else:
                rows[vid, miss] = fused_samples(cfg, store, vid, idx_te, miss, device)
        for (name, miss), x in rows.items():
            xx = t(x)
            for setting, f in probe.items():
                scores[name, setting, miss] = f(xx).astype(np.float32)
        del rows
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"QG4 {vid} done", flush=True)

    m = band_masks(idx_te)
    y = idx_te["cls"]
    blocks = cb[idx_te["ci"], idx_te["cj"]]
    wts = block_bootstrap(blocks, n_boot, int(cfg["seed"]))
    boot = {k: {b: bootstrap_ap(z[s], y[s], wts[:, s]) for b, s in m.items()} for k, z in scores.items()}
    ap = {k: {b: average_precision(z[s], y[s]) for b, s in m.items()} for k, z in scores.items()}
    # a linear probe (also after the PCA) turns any constant fill of the missing half into the same shift of
    # every score, so the AP (a ranking metric) is the same for zeros and for the S1-train mean: checked here
    checks["F-C_zeros_equal_mean_fill_ap"] = bool(all(abs(ap["F-C zeros", st, ms]["all"] - ap["F-C", st, ms]["all"]) < 1e-6
                                                      for st in ("linear", "pca16") for ms in MISSING))
    names = ["F-C", "F-CM", "F-X"]
    out = {}
    for setting in ("linear", "pca16"):
        out[setting] = {}
        for name in names:
            r = {}
            for miss in MISSING:
                k = (name, setting, miss)
                r[f"missing_{miss}"] = {"test_ap": {b: round(v, 4) for b, v in ap[k].items()},
                                        "test_ap_ci95": {b: ci95(v) for b, v in boot[k].items()}}
                if miss != "none":
                    base = (name, setting, "none")
                    r[f"missing_{miss}"]["retention"] = round(ap[k]["all"] / ap[base]["all"], 3)
                    r[f"missing_{miss}"]["retention_ci95"] = ci95(boot[k]["all"] / boot[base]["all"])
            out[setting][name] = r
        diffs = {}
        for miss in MISSING:
            diffs[f"missing_{miss}"] = {other: {"ap_difference": round(ap["F-X", setting, miss]["all"] - ap[other, setting, miss]["all"], 4),
                                                "ci95": ci95(boot["F-X", setting, miss]["all"] - boot[other, setting, miss]["all"]),
                                                "fraction_resamples_F-X_better": round(float(np.mean(boot["F-X", setting, miss]["all"] > boot[other, setting, miss]["all"])), 3)}
                                        for other in names[:2]}
        out[setting]["difference_F-X_minus"] = diffs
    # complete-data rows equal QG1 / QG3 (same probes, refitted deterministically)
    for name, vid in (("F-C", "F-C"), ("F-CM", "F-CM"), ("F-X", "F-X")):
        checks[f"{vid}_complete_equals_qg1_qg3"] = [round(ap[name, "linear", "none"]["all"], 4), qg1[vid]["test_ap"]["all"],
                                                     round(ap[name, "pca16", "none"]["all"], 4), qg3[vid][d16]["test_ap"]["all"]]
    report = {"results": out,
              "single_modality_reference_qg1_qg3": {v: {"linear": qg1[v]["test_ap"]["all"], "pca16": qg3[v][d16]["test_ap"]["all"]}
                                                    for v in ("M-W", "M-D")},
              "checks": checks, "test_blocks": int(np.unique(blocks).size), "bootstrap_resamples": n_boot,
              "note": ("probes fitted on complete S1-train data (C of QG1 / QG3); missing_dem = only the WAC available at test "
                       "time; F-X / F-CM: mask token of the missing modality; F-C: missing half = the S1-train mean (any constant "
                       "fill gives the same AP with a linear probe, checked with zeros); "
                       "retention = AP with the modality missing / AP complete; single-modality reference = probes trained "
                       "on that modality alone (QG1 / QG3)")}
    if not smoke:
        save_report(cfg, "QG4", report)
    return report


# ---------------------------------------------------------------------------
# QG6 - Results per terrain
# ---------------------------------------------------------------------------


def run_terrain(cfg: dict, n_boot: int | None = None, save: bool = True) -> dict:
    """QG6: the test AP of the linear probe (QG1) and of the PCA-16 probe (QG3) of every variant, from their
    saved test scores, per crater-density tercile of the test blocks (stratum of QB3.1: low ~ mare-like,
    high ~ highland), with a block bootstrap inside each tercile; the fusion gain (F-X minus M-W, minus the
    best single modality of the tercile, minus F-C) and its difference between the low and the high
    tercile (the terciles have disjoint blocks, so their resamples are independent)."""
    import pandas as pd

    n_boot = int(cfg["probes"]["bootstrap"]) if n_boot is None else n_boot
    blocks_csv = pd.read_csv(cfg["paths"]["output"] / "split_blocks.csv")
    n_c = int(blocks_csv["bj"].max()) + 1
    stratum = dict(zip(blocks_csv["bi"] * n_c + blocks_csv["bj"], blocks_csv["stratum"]))
    density = dict(zip(blocks_csv["bi"] * n_c + blocks_csv["bj"], blocks_csv["density_per_1000km2"]))
    d16 = int(cfg["probes"]["pca"]["table_dim"])
    files = {"linear": probes_dir(cfg) / "qg1_test_scores.npz", "pca16": probes_dir(cfg) / "qg3_test_scores.npz"}
    data = {k: np.load(f) for k, f in files.items()}
    z0 = data["linear"]
    if not np.array_equal(z0["block"], data["pca16"]["block"]) or not np.array_equal(z0["cls"], data["pca16"]["cls"]):
        raise ValueError("the QG1 and QG3 test scores are not on the same samples")
    y, blk = z0["cls"], z0["block"]
    terc = np.array([stratum[b] for b in blk])
    variants = list(VARIANTS)
    qd_a5 = load_report(cfg, "QD-A5")["correlation_per_subtile"]["wac_sun"]
    out = {"terciles": {}}
    boot_gain = {}
    for t_id, t_name in enumerate(("low", "medium", "high")):
        sel = terc == t_id
        ub = np.unique(blk[sel])
        wts = block_bootstrap(blk[sel], n_boot, int(cfg["seed"]) + t_id)
        r = {"stratum": t_id, "test_blocks": int(ub.size), "samples": int(sel.sum()), "positives": int((y[sel] == 1).sum()),
             "prevalence": round(float((y[sel] == 1).mean()), 4),
             "block_density_per_1000km2_min_max": [round(float(min(density[b] for b in ub)), 1), round(float(max(density[b] for b in ub)), 1)],
             "qd_a5_wac_dem_correlation_median": qd_a5[f"stratum_{t_id}"]["p50"]}
        for setting, z in data.items():
            ap = {v: average_precision(z[f"score_{v}"][sel], y[sel]) for v in variants}
            boot = {v: bootstrap_ap(z[f"score_{v}"][sel], y[sel], wts) for v in variants}
            best = max(("M-W", "M-D"), key=lambda v: ap[v])
            gains = {"F-X_minus_M-W": ("M-W", None), "F-X_minus_best_single": (best, None), "F-X_minus_F-C": ("F-C", None)}
            r[setting] = {"test_ap": {v: round(a, 4) for v, a in ap.items()},
                          "test_ap_ci95": {v: ci95(b) for v, b in boot.items()},
                          "best_single_modality": best,
                          "gain": {g: {"ap_difference": round(ap["F-X"] - ap[o], 4), "ci95": ci95(boot["F-X"] - boot[o]),
                                       "fraction_resamples_positive": round(float(np.mean(boot["F-X"] > boot[o])), 3)}
                                   for g, (o, _) in gains.items()},
                          "F-X_minus_each": {v: {"ap_difference": round(ap["F-X"] - ap[v], 4), "ci95": ci95(boot["F-X"] - boot[v])}
                                             for v in variants if v != "F-X"}}
            for g, (o, _) in gains.items():
                boot_gain[setting, g, t_name] = boot["F-X"] - boot[o]
        out["terciles"][t_name] = r
    out["gain_low_minus_high_tercile"] = {
        setting: {g: {"difference": round(out["terciles"]["low"][setting]["gain"][g]["ap_difference"]
                                          - out["terciles"]["high"][setting]["gain"][g]["ap_difference"], 4),
                      "ci95": ci95(boot_gain[setting, g, "low"] - boot_gain[setting, g, "high"]),
                      "fraction_resamples_low_larger": round(float(np.mean(boot_gain[setting, g, "low"] > boot_gain[setting, g, "high"])), 3)}
                  for g in ("F-X_minus_M-W", "F-X_minus_best_single", "F-X_minus_F-C")} for setting in data}
    out.update({"bootstrap_resamples": n_boot, "pca_dim": d16,
                "note": ("terciles = crater-density strata of QB3.1 (low ~ mare-like, high ~ highland: a proxy, no mare map); "
                         "block bootstrap inside each tercile (seed + stratum); scores of QG1 / QG3 (probes fitted on S1-train)")})
    if save:
        save_report(cfg, "QG6", out)
    return out


# ---------------------------------------------------------------------------


def selftest() -> None:
    """Weighted AP against scikit-learn; ``fit_logistic`` against scikit-learn on synthetic data."""
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.07).astype(int)
    s = rng.normal(size=5000) + 1.5 * y
    w = rng.integers(0, 3, 5000).astype(float)
    assert abs(average_precision(s, y) - average_precision_score(y, s)) < 1e-12
    assert abs(average_precision(s, y, w) - average_precision_score(y, s, sample_weight=w)) < 1e-12
    x = rng.normal(size=(4000, 10))
    yy = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(size=4000) > 1.8).astype(int)
    r = sklearn_check(x[:3000], yy[:3000], x[3000:], yy[3000:], 0.05, "cpu")
    assert r["check_equal_to_sklearn"], r
    wb = block_bootstrap(np.repeat(np.arange(50), 100), 7, 1)
    assert np.allclose(bootstrap_ap(s, y, wb, chunk=3), [average_precision(s, y, r) for r in wb], atol=1e-9)
    wts = block_bootstrap(np.array([3, 3, 7, 9, 9, 9]), 4, 0)
    assert wts.shape == (4, 6) and all(len(set(r[:2])) == 1 and len(set(r[3:])) == 1 for r in wts)
    print("qc_probes selftest: OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=[*PROBES, "missing", "terrain"])
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="40 tiles per set, 2 variants, 50 resamples, nothing written")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    cfg = load_config(args.config)
    if args.command == "terrain":
        print(json.dumps(run_terrain(cfg, 50 if args.smoke else None, save=not args.smoke), indent=1))
    elif args.command == "missing":
        print(json.dumps(run_missing(cfg, args.device, smoke=args.smoke), indent=1))
    elif args.command:
        print(json.dumps(run_probe(cfg, args.command, args.device, smoke=args.smoke), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
