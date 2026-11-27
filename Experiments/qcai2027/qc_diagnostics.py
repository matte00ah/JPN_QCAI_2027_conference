#!/usr/bin/env python3
"""Kernel diagnostics of the QCAI 2027 experiment (plan phase QI5, RQ3).

Separate from ``qc_kernels.py`` (kernels) and ``qc_experiments.py`` (classifier runs) for length, decided with the
user 2026-11-23; it reuses their functions by import.

QI5.1 - Concentration.  ``run_concentration``: on the training samples of the main grid (``stage2.main``: variant,
size n, every seed of ``qk_samples.seeds``; decided with the user) and for every d of ``quantum.dims`` and c of
``quantum.bandwidths``, the exact training kernel K_train of QK-A, QK-Z (r of ``quantum.zz_reps``) and QK-P (gamma =
the median gamma, s = 1), and of its off-diagonal values (``offdiag_stats``): mean, variance and
``var_over_shot`` = variance / mean binomial variance K(1 - K) / N of a value estimated with N shots (N =
``diagnostics.shots_reference``): < 1 means the spread of the kernel values is smaller than the shot noise of
one estimate, so the kernel cannot be resolved on hardware with N shots.  Reference: the off-diagonal values of
Haar-random states of d qubits (``haar_fidelity``: mean 1 / 2^d, variance (D - 1) / (D^2 (D + 1)), D = 2^d), the
fully concentrated limit.  QK-P rescales itself with the median gamma, so its own variance hides the concentration
of the states: as decided with the user, the spread of the single-qubit expectations <X>, <Y>, <Z> it is built
from is reported too (``expectation_stats``: variance across samples averaged over the 3d coordinates, mean Bloch
vector length, variance / shot variance (1 - <P>^2) / N), the concentration indicator of projected kernels
(Thanasilp et al. 2024).  Mean / std over the seeds.

QI5.2 - Kernel-target alignment.  ``run_alignment``: on the same samples, the alignment (``alignment``) of the exact
training kernel of every configuration of the QI4.2 grid except C (map, d, c, r, s; QK-P at every s of
``quantum.projected_gamma_scales``), plain (Cristianini et al. 2001) and centred (Cortes et al. 2012: invariant to a
constant added to the kernel, as the SVM with a bias; used for the choice, decided with the user), saved with the
best CV AP and F1 over C of ``qi4_cv.csv`` in ``results/qi5_alignment.csv``.  Report (decided with the user): KTA
per kernel, d and c; Spearman correlation with the CV scores over the configurations; the configuration with the
largest centred KTA against the CV choice of QI4.3 (agreement, loss of CV AP); at the per-c d of QI4.3 the test AP
of the c chosen by the KTA, read from the per-c runs (no new training).

QI5.3 - Geometric difference.  ``run_geometric``: g(K_C || K_Q) of Huang et al. 2021 (``geometric_difference``,
regularised, kernels normalised to trace N, every lambda of ``diagnostics.gd_lambdas``) for the training kernel of
the configuration chosen by the CV in QI4.3 (every map, d, seed), against the RBF kernel on the same standardised
components at every gamma of ``classical.rbf_gamma_scales`` x the median gamma (the grid of QJ1; g per gamma and its
minimum) and, for QK-Z / QK-P, against QK-A at the same c (decided with the user).

QI5.4 - Eigenvalue spectrum.  ``run_spectrum``: spectrum of the training kernels (``spectrum_stats``: participation
ratio, eigenvalues covering 90 / 99 % of the trace, share of the 20 largest) of the CV choice of QI4.3 (every map, d,
seed), of the RBF closest to it in QI5.3 and, at one d, of QK-A / QK-Z at the extreme bandwidths (decided with the
user); every spectrum also of the centred kernel H K H (``centre``), as the centred KTA of QI5.2.

Commands
--------
concentration   QI5.1 -> report section QI5.1 of ``results/qi_kernels.json``.  GPU job (run_qcai_gpu.sh,
                ``--device cuda``; Aer states on the CPUs), ~15 min.

geometric       QI5.3 -> report section QI5.3.  GPU job, ~5 min; needs the ``qi4_runs.csv`` of QI4.3.

spectrum        QI5.4 -> report section QI5.4.  GPU job, ~5 min; needs QI4.3 and the report section QI5.3.

alignment       QI5.2 -> ``results/qi5_alignment.csv`` and report section QI5.2.  GPU job, ~15 min; needs the
                ``qi4_cv.csv`` / ``qi4_runs.csv`` of QI4.3.  ``--from-csv``: the report only, from the saved table
                (seconds).

Usage
-----
    python Experiments/qcai2027/qc_diagnostics.py concentration --device cuda   (GPU job: run_qcai_gpu.sh)
    python Experiments/qcai2027/qc_diagnostics.py geometric --device cuda       (GPU job: run_qcai_gpu.sh)
    python Experiments/qcai2027/qc_diagnostics.py spectrum --device cuda        (GPU job: run_qcai_gpu.sh)
    python Experiments/qcai2027/qc_diagnostics.py alignment --device cuda       (GPU job: run_qcai_gpu.sh)
    python Experiments/qcai2027/qc_diagnostics.py --selftest                    (seconds)
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
from qc_experiments import zz_kernel_and_expectations  # noqa: E402
from qc_kernels import (feature_angles, features_path, kernel_projected, kernel_qk_a, load_qk,  # noqa: E402
                        median_gamma, reduce_features, rep_suffix)

g3 = lambda x: float(f"{x:.3g}")


# ---------------------------------------------------------------------------
# QI5.1 - Concentration
# ---------------------------------------------------------------------------


def haar_fidelity(d: int) -> tuple[float, float]:
    """Mean and variance of |<a|b>|^2 for two Haar-random states of d qubits (Beta(1, D - 1), D = 2^d)."""
    D = 2.0 ** d
    return 1 / D, (D - 1) / (D ** 2 * (D + 1))


def offdiag_stats(k: np.ndarray, shots: int) -> dict:
    """Off-diagonal values of a symmetric training kernel: mean, variance, variance / mean shot variance
    K (1 - K) / ``shots`` (meaningful for fidelity kernels, whose values are probabilities)."""
    v = k[np.triu_indices(k.shape[0], 1)]
    shot_var = float(np.mean(np.clip(v, 0, 1) * (1 - np.clip(v, 0, 1)))) / shots
    return {"mean": float(v.mean()), "var": float(v.var()), "var_over_shot": float(v.var()) / max(shot_var, 1e-300)}


def expectation_stats(e: np.ndarray, shots: int) -> dict:
    """Spread of the single-qubit expectations (n, 3d) [X_0, Y_0, Z_0, ...] behind QK-P: variance across samples
    averaged over the coordinates, mean length of the Bloch vectors, variance / mean shot variance (1 - <P>^2) / N."""
    var = e.var(0)
    shot_var = (1 - e ** 2).mean(0) / shots
    bloch = np.linalg.norm(e.reshape(e.shape[0], -1, 3), axis=2)
    return {"var_mean": float(var.mean()), "bloch_length_mean": float(bloch.mean()),
            "var_over_shot": float(np.mean(var / np.maximum(shot_var, 1e-300)))}


def sample_kernels(a_tr: np.ndarray, c: float, reps, device: str, scales=(1.0,)):
    """Exact training kernels of one sample at bandwidth ``c``: yields (name, s, K_train, expectations or None) for
    QK-A, and for every r QK-Z_r<r> and QK-P_r<r> (one per gamma = s x the median gamma, s in ``scales``; s = None for
    the fidelity maps); the QK-Z states are simulated once for QK-Z and QK-P (``zz_kernel_and_expectations``)."""
    th = feature_angles(a_tr, c)
    yield "QK-A", None, kernel_qk_a(th, th), None
    for r in reps:
        (k_z, _), (e, _) = zz_kernel_and_expectations(th, th[:1], int(r), device)
        yield f"QK-Z_r{r}", None, k_z, None
        g = median_gamma(e)
        for s_ in scales:
            yield f"QK-P_r{r}", float(s_), kernel_projected(e, e, float(s_) * g), e


def run_concentration(cfg: dict, device: str = "cuda") -> dict:
    """QI5.1 report (see the module docstring)."""
    main, q = cfg["stage2"]["main"], cfg["quantum"]
    vid, n = main["variant"], int(main["n"])
    seeds, dims = [int(s) for s in cfg["qk_samples"]["seeds"]], [int(d) for d in q["dims"]]
    shots = int(cfg["diagnostics"]["shots_reference"])
    z = np.load(features_path(cfg))
    per = {}                                                    # (kernel, d, c) -> [stats per seed]
    diag_dev, sym_dev = 0.0, 0.0
    for seed in seeds:
        x_tr = load_qk(cfg, vid, n, seed, z)[0]
        for d in dims:
            t0 = time.time()
            a_tr = reduce_features(x_tr, d)[0]
            for c in [float(v) for v in q["bandwidths"]]:
                for name, _, k, e in sample_kernels(a_tr, c, q["zz_reps"], device):
                    diag_dev = max(diag_dev, float(np.abs(np.diag(k) - 1).max()))
                    sym_dev = max(sym_dev, float(np.abs(k - k.T).max()))
                    per.setdefault((name, d, c), []).append(offdiag_stats(k, shots))
                    if e is not None:
                        per.setdefault((name.replace("QK-P", "expectations"), d, c), []).append(expectation_stats(e, shots))
            print(f"QI5.1 seed {seed} d {d}: {time.time() - t0:.0f} s", flush=True)
    ms = lambda v: [g3(np.mean(v)), g3(np.std(v, ddof=1)) if len(v) > 1 else None]
    out_k, out_e = {}, {}
    for (name, d, c), rows in per.items():
        tgt = out_e if name.startswith("expectations") else out_k
        tgt.setdefault(name, {}).setdefault(f"d{d}", {})[f"c{c:g}"] = {k: ms([r[k] for r in rows]) for k in rows[0]}
    out = {"variant": vid, "n": n, "seeds": seeds, "shots_reference": shots,
           "haar_floor": {f"d{d}": {"mean": g3(haar_fidelity(d)[0]), "var": g3(haar_fidelity(d)[1])} for d in dims},
           "kernels": out_k, "projected_expectations": out_e,
           "checks": {"max_abs_diag_minus_1": g3(diag_dev), "max_abs_asymmetry": g3(sym_dev)},
           "note": ("off-diagonal values of the exact training kernel, [mean, std] over the seeds; var_over_shot = "
                    "variance / mean K(1-K)/N (< 1: spread below the shot noise of one estimate with N shots); "
                    "haar_floor = Haar-random states (full concentration); QK-P with the median gamma (s = 1) rescales "
                    "itself: its concentration is read in projected_expectations (variance of <X>, <Y>, <Z> across "
                    "samples, mean Bloch length, variance / (1 - <P>^2)/N)")}
    save_report(cfg, "QI5.1", out)
    return out


# ---------------------------------------------------------------------------
# QI5.2 - Kernel-target alignment
# ---------------------------------------------------------------------------

ALIGN_COLS = ["step", "script", "variant", "n", "seed", "d", "map", "r", "s", "c", "kta", "kta_centered", "cv_ap_best_C",
              "cv_f1_best_C"]


def centre(k: np.ndarray) -> np.ndarray:
    """H K H, H = I - 11^T / n: the kernel without its constant part (to which the SVM with a bias is blind)."""
    return k - k.mean(0, keepdims=True) - k.mean(1, keepdims=True) + k.mean()


def alignment(k: np.ndarray, y: np.ndarray, centered: bool = False) -> float:
    """Kernel-target alignment <K, t t^T>_F / (||K||_F ||t t^T||_F), t = 2y - 1 (Cristianini et al. 2001); with
    ``centered`` K is first centred in feature space, H K H with H = I - 11^T / n (Cortes et al. 2012), which makes it
    invariant to a constant added to K, as the SVM with a bias.  NaN for a zero (e.g. constant, centred) kernel."""
    t = 2.0 * np.asarray(y, dtype=np.float64) - 1
    if centered:
        k = centre(k)
    nk = np.linalg.norm(k)
    return float(t @ k @ t / (nk * t.size)) if nk > 1e-300 else float("nan")


def kernel_label(m: str, r) -> str:
    return m + rep_suffix(m, r)


def alignment_table(cfg: dict, device: str) -> "pd.DataFrame":
    """KTA (plain and centred) of the exact training kernel of every configuration of the QI4.2 grid except C
    (map, d, c, r, s) on the 5 main-grid samples, joined with the best CV AP and F1 over C of ``qi4_cv.csv``."""
    import pandas as pd

    from qc_experiments import qi4_paths

    main, q = cfg["stage2"]["main"], cfg["quantum"]
    vid, n = main["variant"], int(main["n"])
    z = np.load(features_path(cfg))
    rows = []
    for seed in [int(s) for s in cfg["qk_samples"]["seeds"]]:
        x_tr, i_tr = load_qk(cfg, vid, n, seed, z)[:2]
        y = i_tr["cls"].astype(int)
        for d in [int(v) for v in q["dims"]]:
            t0 = time.time()
            a_tr = reduce_features(x_tr, d)[0]
            for c in [float(v) for v in q["bandwidths"]]:
                for name, s_, k, _ in sample_kernels(a_tr, c, q["zz_reps"], device, q["projected_gamma_scales"]):
                    m, r = (name.split("_r")[0], int(name.split("_r")[1])) if "_r" in name else (name, np.nan)
                    rows.append({"step": "QI5.2", "script": "qc_diagnostics.py", "variant": vid, "n": n, "seed": seed,
                                 "d": d, "map": m, "r": r, "s": np.nan if s_ is None else s_, "c": c,
                                 "kta": alignment(k, y), "kta_centered": alignment(k, y, centered=True)})
            print(f"QI5.2 seed {seed} d {d}: {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    cv = pd.read_csv(qi4_paths(cfg)["cv"])
    cv = cv[(cv["n"] == n) & (cv["variant"] == vid)]
    keys = ["seed", "d", "map", "r", "s", "c"]
    norm = lambda t: t.assign(r=t["r"].fillna(-1).astype(int), s=t["s"].fillna(-1).round(6), c=t["c"].round(6))
    best = norm(cv).groupby(keys)[["cv_ap", "cv_f1"]].max().reset_index()
    best = best.rename(columns={"cv_ap": "cv_ap_best_C", "cv_f1": "cv_f1_best_C"})
    df = norm(df).merge(best, on=keys, how="left")
    df[["r", "s"]] = df[["r", "s"]].replace(-1, np.nan)
    return df[ALIGN_COLS]


def alignment_summary(cfg: dict, df) -> dict:
    """QI5.2 report from the alignment table: centred / plain KTA per kernel, d and c (QK-P: best s by centred KTA);
    Spearman correlation of the KTA with the best CV F1 / AP over the configurations (c, r, s) per map and over all
    maps, per d; the configuration chosen by the largest centred KTA against the CV choice of QI4.3 (same c, same
    configuration, loss of mean CV AP) and, at ``stage2.main.per_c_dims``, the test AP of the per-c run at the c
    chosen by the KTA against the CV choice."""
    import pandas as pd
    from scipy.stats import spearmanr

    from qc_experiments import qi4_paths, run_id

    main = cfg["stage2"]["main"]
    n = int(main["n"])
    ms = lambda v: [round(float(np.nanmean(v)), 4), round(float(np.nanstd(v, ddof=1)), 4) if len(v) > 1 else None]
    df = df.assign(kernel=[kernel_label(m, r) for m, r in zip(df["map"], df["r"])])
    # (a) KTA per kernel, d, c (QK-P: per seed the s with the largest centred KTA)
    top = df.loc[df.groupby(["seed", "d", "c", "kernel"])["kta_centered"].idxmax()]
    kta = {k: {f"d{d}": {f"c{c:g}": {"centered": ms(g3_["kta_centered"]), "plain": ms(g3_["kta"]),
                                       **({"s_mode": float(g3_["s"].mode().iloc[0])} if k.startswith("QK-P") else {})}
                         for c, g3_ in g2.groupby("c")} for d, g2 in g.groupby("d")} for k, g in top.groupby("kernel")}
    # (b) rank correlation with the CV scores over the configurations
    groups = {"QK-A": df["map"] == "QK-A", "QK-Z": df["map"] == "QK-Z", "QK-P": df["map"] == "QK-P",
              "all": np.ones(len(df), bool)}
    corr = {}
    for gname, gm in groups.items():
        for d, g in df[gm].groupby("d"):
            vals = {"centered_vs_cv_f1": [], "centered_vs_cv_ap": [], "plain_vs_cv_f1": []}
            for _, gs in g.groupby("seed"):
                vals["centered_vs_cv_f1"].append(spearmanr(gs["kta_centered"], gs["cv_f1_best_C"]).statistic)
                vals["centered_vs_cv_ap"].append(spearmanr(gs["kta_centered"], gs["cv_ap_best_C"]).statistic)
                vals["plain_vs_cv_f1"].append(spearmanr(gs["kta"], gs["cv_f1_best_C"]).statistic)
            corr.setdefault(gname, {})[f"d{d}"] = {k: ms(v) for k, v in vals.items()}
    # (c) choice by the largest centred KTA against the CV choice of QI4.3; (d) test AP at the per-c dims
    runs = pd.read_csv(qi4_paths(cfg)["runs"])
    chosen = runs[(runs["role"] == "chosen") & (runs["n"] == n)]
    test_ap = runs.set_index("run_id")["test_ap"]
    per_c_dims = [int(v) for v in main["per_c_dims"]]
    choice, test = {}, {}
    for gname, gm in groups.items():
        for d, g in df[gm].groupby("d"):
            same_c, same_cfg, same_map, loss, loss_plain, ap_kta, ap_cv = [], [], [], [], [], [], []
            for seed, gs in g.groupby("seed"):
                gs = gs.sort_values(["c", "r", "s"])                            # ties -> smaller c, r, s
                pick, pick_plain = gs.loc[gs["kta_centered"].idxmax()], gs.loc[gs["kta"].idxmax()]
                ch = chosen[(chosen["d"] == d) & (chosen["seed"] == seed)]
                cvp = ch[ch["map"] == gname].iloc[0] if gname != "all" else ch.loc[ch["cv_ap"].idxmax()]
                same_c.append(float(np.isclose(pick["c"], cvp["c"])))
                same_map.append(float(pick["map"] == cvp["map"]))
                same_cfg.append(float(pick["map"] == cvp["map"] and np.isclose(pick["c"], cvp["c"])
                                      and (pd.isna(cvp["r"]) or pick["r"] == cvp["r"])
                                      and (pd.isna(cvp["s"]) or np.isclose(pick["s"], cvp["s"]))))
                loss.append(gs["cv_ap_best_C"].max() - pick["cv_ap_best_C"])
                loss_plain.append(gs["cv_ap_best_C"].max() - pick_plain["cv_ap_best_C"])
                if d in per_c_dims and gname != "all":
                    ap_kta.append(test_ap[run_id(gname, d, n, seed, float(pick["c"]))])
                    ap_cv.append(test_ap[run_id(gname, d, n, seed)])
            choice.setdefault(gname, {})[f"d{d}"] = {
                "same_c_share": round(float(np.mean(same_c)), 2), "same_configuration_share": round(float(np.mean(same_cfg)), 2),
                **({"same_map_share": round(float(np.mean(same_map)), 2)} if gname == "all" else {}),
                "cv_ap_loss_centered": ms(loss), "cv_ap_loss_plain": ms(loss_plain)}
            if ap_kta:
                test.setdefault(gname, {})[f"d{d}"] = {"test_ap_kta_c": ms(ap_kta), "test_ap_cv_choice": ms(ap_cv)}
    ident_y = np.r_[np.ones(n // 2), np.zeros(n - n // 2)]          # balanced labels of the QK samples (QH4)
    return {"variant": main["variant"], "n": n, "kta": kta, "spearman": corr, "kta_choice_vs_cv": choice,
            "test_ap_at_kta_c": test,
            "identity_kernel_kta": {"plain": round(alignment(np.eye(n), ident_y), 4),
                                    "centered": round(alignment(np.eye(n), ident_y, centered=True), 4),
                                    "one_over_sqrt_n": round(float(1 / np.sqrt(n)), 4)},
            "checks": {"rows": int(len(df)), "rows_without_cv_match": int(df["cv_ap_best_C"].isna().sum()),
                       "nan_kta": int(df["kta_centered"].isna().sum())},
            "note": ("KTA of the exact training kernel of the 5 main-grid samples, [mean, std] over the seeds; centered = "
                     "Cortes et al. 2012 (used for the choice), plain = Cristianini et al. 2001; QK-P: s with the largest "
                     "centred KTA; spearman over the configurations (c, r, s) of a map or of all maps, against the best "
                     "CV score over C (qi4_cv.csv); kta_choice_vs_cv: configuration with the largest centred KTA against "
                     "the CV choice of QI4.3 (cv_ap_loss = best mean CV AP - CV AP of the KTA choice); test_ap_at_kta_c: "
                     "per-c run of QI4.3 at the c chosen by the KTA (other hyper-parameters by CV at that c); every value "
                     "in results/qi5_alignment.csv; identity_kernel_kta = KTA of the fully concentrated kernel K = I on n balanced "
                     "samples (the limit of a concentrated fidelity kernel)")}


def alignment_path(cfg: dict) -> Path:
    return cfg["paths"]["results"] / "qi5_alignment.csv"


def run_alignment(cfg: dict, device: str = "cuda", from_csv: bool = False) -> dict:
    """QI5.2: the alignment table (``results/qi5_alignment.csv``) and the report section QI5.2; ``from_csv``: the
    report only, from the saved table (seconds)."""
    import pandas as pd

    from qc_data import written

    if from_csv:
        out = alignment_summary(cfg, pd.read_csv(alignment_path(cfg)))
        save_report(cfg, "QI5.2", out)
        return out
    df = alignment_table(cfg, device)
    df.to_csv(alignment_path(cfg), index=False, float_format="%.6g")
    written(alignment_path(cfg))
    out = alignment_summary(cfg, df)
    save_report(cfg, "QI5.2", out)
    return out


# ---------------------------------------------------------------------------
# QI5.3 - Geometric difference
# ---------------------------------------------------------------------------


def geometric_difference(k_c: np.ndarray, k_q: np.ndarray, lambdas) -> dict:
    """Regularised geometric difference g(K_C || K_Q) of Huang et al. 2021 (Nat. Commun. 12, 2631), kernels normalised
    to trace N: g = sqrt(|| sqrt(K_Q) sqrt(K_C) (K_C + lambda I)^-2 sqrt(K_C) sqrt(K_Q) ||_inf) for every lambda of
    ``lambdas`` ({lambda: g}).  g ~ 1: the classical kernel can learn whatever the quantum one learns with comparable
    data; large g (up to ~sqrt(N)): room for a quantum advantage on some labelling.  Negative eigenvalues (round-off)
    clipped to 0."""
    n = k_c.shape[0]
    kc, kq = k_c * n / np.trace(k_c), k_q * n / np.trace(k_q)
    wc, vc = np.linalg.eigh((kc + kc.T) / 2)
    wc = np.clip(wc, 0, None)
    wq, vq = np.linalg.eigh((kq + kq.T) / 2)
    sq = (vq * np.sqrt(np.clip(wq, 0, None))) @ vq.T
    out = {}
    for lam in lambdas:
        m = sq @ ((vc * (wc / (wc + lam) ** 2)) @ vc.T) @ sq
        out[float(lam)] = float(np.sqrt(max(np.linalg.eigvalsh((m + m.T) / 2).max(), 0.0)))
    return out


def run_geometric(cfg: dict, device: str = "cuda") -> dict:
    """QI5.3: for every map, d and seed of the main grid, the training kernel of the configuration chosen by the CV
    in QI4.3 (``stage2_kernels``) against (decided with the user) the RBF kernel exp(-gamma ||a - a'||^2) on the same
    standardised components, gamma = s x the median gamma for every s of ``classical.rbf_gamma_scales`` (g per gamma
    and its minimum = the most favourable classical kernel), and for QK-Z / QK-P against QK-A at the same c (a product
    kernel, classically computable: does the entanglement add geometry?); every lambda of ``diagnostics.gd_lambdas``.
    Report: [mean, std] over the seeds."""
    import pandas as pd

    from qc_experiments import config_rs, qi4_paths, stage2_kernels

    main = cfg["stage2"]["main"]
    vid, n = main["variant"], int(main["n"])
    lams = [float(v) for v in cfg["diagnostics"]["gd_lambdas"]]
    scales = [float(v) for v in cfg["classical"]["rbf_gamma_scales"]]
    runs = pd.read_csv(qi4_paths(cfg)["runs"])
    chosen = runs[(runs["role"] == "chosen") & (runs["n"] == n) & (runs["variant"] == vid)]
    z = np.load(features_path(cfg))
    per = {}                                                    # (map, d, what) -> [{lam: g} per seed]
    check_self = 0.0
    for seed in [int(s) for s in cfg["qk_samples"]["seeds"]]:
        x_tr = load_qk(cfg, vid, n, seed, z)[0]
        for d in [int(v) for v in cfg["quantum"]["dims"]]:
            t0 = time.time()
            a_tr = reduce_features(x_tr, d)[0]
            g_med = median_gamma(a_tr)
            rbf = {sc: kernel_projected(a_tr, a_tr, sc * g_med) for sc in scales}
            for row in chosen[(chosen["d"] == d) & (chosen["seed"] == seed)].itertuples():
                r, gs = config_rs(row.r, row.s)
                k_q = stage2_kernels(row.map, a_tr, a_tr[:1], float(row.c), r, gs, device=device)[0]
                check_self = max(check_self, abs(geometric_difference(k_q, k_q, [1e-9])[1e-9] - 1))
                for sc, k_c in rbf.items():
                    per.setdefault((row.map, d, f"rbf_s{sc:g}"), []).append(geometric_difference(k_c, k_q, lams))
                if row.map != "QK-A":
                    k_a = stage2_kernels("QK-A", a_tr, a_tr[:1], float(row.c), device=device)[0]
                    per.setdefault((row.map, d, "qk_a_same_c"), []).append(geometric_difference(k_a, k_q, lams))
            print(f"QI5.3 seed {seed} d {d}: {time.time() - t0:.0f} s", flush=True)
    ms = lambda v: [round(float(np.mean(v)), 3), round(float(np.std(v, ddof=1)), 3) if len(v) > 1 else None]
    out = {}
    for m in sorted({k[0] for k in per}):
        for d in sorted({k[1] for k in per if k[0] == m}):
            e = {}
            for lam in lams:
                gs_ = np.array([[g[lam] for g in per[(m, d, f"rbf_s{sc:g}")]] for sc in scales])   # (gamma, seed)
                e[f"lambda{lam:g}"] = {"g_rbf_per_gamma_mean": [round(float(v), 3) for v in gs_.mean(1)],
                                       "g_rbf_min_over_gamma": ms(gs_.min(0)),
                                       "argmin_gamma_scale_per_seed": [scales[i] for i in gs_.argmin(0)],
                                       **({"g_vs_qk_a_same_c": ms([g[lam] for g in per[(m, d, "qk_a_same_c")]])}
                                          if (m, d, "qk_a_same_c") in per else {})}
            out.setdefault(m, {})[f"d{d}"] = e
    rep_ = {"variant": vid, "n": n, "rbf_gamma_scales": scales, "lambdas": lams, "sqrt_n": round(float(np.sqrt(n)), 2),
            "g": out, "checks": {"max_abs_g_self_minus_1": float(f"{check_self:.3g}")},
            "note": ("g(K_C || K_Q) of Huang et al. 2021, kernels normalised to trace N, [mean, std] over the seeds; K_Q = "
                     "the CV choice of QI4.3 per map, d, seed; K_C = RBF on the same standardised PCA components with "
                     "gamma = s x median gamma (per gamma: mean over the seeds; min over gamma per seed), and QK-A at the "
                     "same c for QK-Z / QK-P; g ~ 1: the classical kernel matches the quantum one on any labelling; "
                     "upper scale ~ sqrt_n; check: g(K || K) at lambda 1e-9")}
    save_report(cfg, "QI5.3", rep_)
    return rep_


# ---------------------------------------------------------------------------
# QI5.4 - Eigenvalue spectrum
# ---------------------------------------------------------------------------


def spectrum_stats(k: np.ndarray, top: int = 20) -> dict:
    """Spectrum of a kernel normalised to trace N (eigenvalues clipped at 0): participation ratio (sum l)^2 / sum l^2
    (1 = one eigenvalue dominates, a nearly constant kernel; N = flat spectrum, the identity, memorisation), the number
    of the largest eigenvalues that cover 90 % and 99 % of the trace, and the share of the trace of the ``top``
    largest ones.  Also used on ``centre(k)``, normalised the same way."""
    n = k.shape[0]
    w = np.clip(np.linalg.eigvalsh((k + k.T) / 2), 0, None)[::-1] * n / np.trace(k)
    cum = np.cumsum(w) / w.sum()
    return {"participation_ratio": float(w.sum() ** 2 / np.sum(w ** 2)),
            "n_eig_90": int(np.searchsorted(cum, 0.90) + 1), "n_eig_99": int(np.searchsorted(cum, 0.99) + 1),
            "top_share": (w[:top] / w.sum()).tolist()}


def run_spectrum(cfg: dict, device: str = "cuda") -> dict:
    """QI5.4 (decided with the user): spectrum (``spectrum_stats``) of the training kernel of the CV choice of QI4.3
    for every map, d and seed, of the RBF with the gamma closest to it in QI5.3 (smallest g at lambda =
    ``diagnostics.spectrum_rbf_lambda``, per seed), and at d = ``diagnostics.spectrum_extreme_d`` of QK-A and QK-Z
    (r = 1) at the extreme bandwidths (smallest and largest c of the grid: the nearly constant and the nearly
    diagonal kernel).  Report: [mean, std] over the seeds; top shares as means."""
    import pandas as pd

    from qc_data import load_report
    from qc_experiments import config_rs, qi4_paths, stage2_kernels

    main, dg = cfg["stage2"]["main"], cfg["diagnostics"]
    vid, n = main["variant"], int(main["n"])
    seeds = [int(v) for v in cfg["qk_samples"]["seeds"]]
    lam, d_ext = float(dg["spectrum_rbf_lambda"]), int(dg["spectrum_extreme_d"])
    c_ext = [min(map(float, cfg["quantum"]["bandwidths"])), max(map(float, cfg["quantum"]["bandwidths"]))]
    gd = load_report(cfg, "QI5.3")["g"]
    runs = pd.read_csv(qi4_paths(cfg)["runs"])
    chosen = runs[(runs["role"] == "chosen") & (runs["n"] == n) & (runs["variant"] == vid)]
    z = np.load(features_path(cfg))
    per = {}
    for i, seed in enumerate(seeds):
        x_tr = load_qk(cfg, vid, n, seed, z)[0]
        for d in [int(v) for v in cfg["quantum"]["dims"]]:
            t0 = time.time()
            a_tr = reduce_features(x_tr, d)[0]
            g_med = median_gamma(a_tr)
            for row in chosen[(chosen["d"] == d) & (chosen["seed"] == seed)].itertuples():
                r, gs = config_rs(row.r, row.s)
                k_q = stage2_kernels(row.map, a_tr, a_tr[:1], float(row.c), r, gs, device=device)[0]
                sc = gd[row.map][f"d{d}"][f"lambda{lam:g}"]["argmin_gamma_scale_per_seed"][i]
                k_r = kernel_projected(a_tr, a_tr, sc * g_med)
                for what, k_ in (("cv_choice", k_q), ("rbf_closest", k_r)):
                    per.setdefault((row.map, f"d{d}", what), []).append(spectrum_stats(k_))
                    per.setdefault((row.map, f"d{d}", what + "_centred"), []).append(spectrum_stats(centre(k_)))
            if d == d_ext:
                for m in ("QK-A", "QK-Z"):
                    for c in c_ext:
                        k = stage2_kernels(m, a_tr, a_tr[:1], c, 1, device=device)[0]
                        per.setdefault((m, f"d{d}", f"c{c:g}"), []).append(spectrum_stats(k))
                        per.setdefault((m, f"d{d}", f"c{c:g}_centred"), []).append(spectrum_stats(centre(k)))
            print(f"QI5.4 seed {seed} d {d}: {time.time() - t0:.0f} s", flush=True)
    ms = lambda v: [round(float(np.mean(v)), 2), round(float(np.std(v, ddof=1)), 2) if len(v) > 1 else None]
    out = {}
    for (m, d, what), rows in sorted(per.items()):
        out.setdefault(m, {}).setdefault(d, {})[what] = {
            **{k: ms([r_[k] for r_ in rows]) for k in ("participation_ratio", "n_eig_90", "n_eig_99")},
            "top_share_mean": [round(float(v), 4) for v in np.mean([r_["top_share"] for r_ in rows], 0)]}
    rep_ = {"variant": vid, "n": n, "rbf_lambda": lam, "extreme_d": d_ext, "extreme_c": c_ext, "spectra": out,
            "note": ("training kernels normalised to trace N, [mean, std] over the seeds; participation_ratio from 1 "
                     "(one eigenvalue: nearly constant kernel) to N (flat: identity, memorisation); n_eig_90 / 99 = "
                     "largest eigenvalues covering 90 / 99 % of the trace; top_share_mean = share of the trace of the "
                     "20 largest; cv_choice = QI4.3 choice; rbf_closest = RBF with the smallest g of QI5.3 at rbf_lambda; "
                     "c<..> = QK-A / QK-Z (r = 1) at the extreme bandwidths, d = extreme_d; *_centred = the same on H K H "
                     "(constant part removed, as the centred KTA of QI5.2: the SVM with a bias is blind to it)")}
    save_report(cfg, "QI5.4", rep_)
    return rep_


# ---------------------------------------------------------------------------


def selftest() -> None:
    """Haar reference against sampled random states, the statistics on hand-made kernels and expectations, and
    the limits of QK-Z (c -> 0: no concentration; large c at d = 10: close to the Haar floor)."""
    rng = np.random.default_rng(0)
    d, m = 6, 400
    psi = rng.normal(size=(m, 2 ** d)) + 1j * rng.normal(size=(m, 2 ** d))
    psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    st = offdiag_stats(np.abs(psi @ psi.conj().T) ** 2, 1024)
    mu, var = haar_fidelity(d)
    assert abs(st["mean"] - mu) < 0.05 * mu and abs(st["var"] - var) < 0.1 * var
    k = np.array([[1, 0.5, 0.1], [0.5, 1, 0.3], [0.1, 0.3, 1]])
    st = offdiag_stats(k, 100)
    v = np.array([0.5, 0.1, 0.3])
    assert np.isclose(st["mean"], v.mean()) and np.isclose(st["var"], v.var())
    assert np.isclose(st["var_over_shot"], v.var() / (np.mean(v * (1 - v)) / 100))
    e = np.array([[1, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, -1]], float)
    es = expectation_stats(e, 10)
    assert np.isclose(es["bloch_length_mean"], 1) and np.isclose(es["var_mean"], np.mean(e.var(0)))
    a = rng.normal(size=(60, 10))
    ks = {nm: kk for nm, _, kk, _ in sample_kernels(a, 1e-3, [1], "cpu")}
    assert offdiag_stats(ks["QK-Z_r1"], 1024)["mean"] > 0.99                    # tiny angles: all values ~ 1
    ks = {nm: kk for nm, _, kk, _ in sample_kernels(a, 3.0, [2], "cpu")}
    st = offdiag_stats(ks["QK-Z_r2"], 1024)
    assert st["mean"] < 20 * haar_fidelity(10)[0]                             # large angles: near the Haar floor
    assert np.allclose(np.diag(ks["QK-P_r2"]), 1) and np.allclose(ks["QK-A"], ks["QK-A"].T)
    yb = np.array([1, 1, 0, 0])
    kp = np.outer(2 * yb - 1, 2 * yb - 1).astype(float)
    assert np.isclose(alignment(kp, yb), 1) and np.isclose(alignment(kp, yb, centered=True), 1)
    assert np.isclose(alignment(np.ones((4, 4)), yb), 0) and np.isnan(alignment(np.ones((4, 4)), yb, centered=True))
    kr = rng.random((7, 7))
    kr = kr @ kr.T
    y7 = np.array([1, 1, 1, 0, 0, 0, 0])
    assert np.isclose(alignment(kr, y7, centered=True), alignment(kr + 3.0, y7, centered=True))   # shift invariant
    assert not np.isclose(alignment(kr, y7), alignment(kr + 3.0, y7))
    sk = list(sample_kernels(a, 0.2, [1], "cpu", scales=(0.5, 2.0)))
    assert [(nm, s_) for nm, s_, _, _ in sk] == [("QK-A", None), ("QK-Z_r1", None), ("QK-P_r1", 0.5), ("QK-P_r1", 2.0)]
    kk = ks["QK-A"]
    assert abs(geometric_difference(kk, kk, [1e-9])[1e-9] - 1) < 1e-4                # same kernel: g = 1
    assert abs(geometric_difference(kk, 5.0 * kk, [1e-9])[1e-9] - 1) < 1e-4          # trace normalisation
    g_id = geometric_difference(np.ones((60, 60)) * 0.99 + 0.01 * np.eye(60), np.eye(60), [1e-3, 1e-1])
    assert g_id[1e-3] > 5 and g_id[1e-3] > g_id[1e-1]          # near-constant vs identity: g >> 1, shrunk by a large lambda
    st = spectrum_stats(np.eye(50))
    assert np.isclose(st["participation_ratio"], 50) and st["n_eig_90"] == 45 and st["n_eig_99"] == 50
    st = spectrum_stats(np.ones((50, 50)))
    assert np.isclose(st["participation_ratio"], 1) and st["n_eig_90"] == 1 and np.isclose(st["top_share"][0], 1)
    assert np.allclose(centre(np.ones((5, 5))), 0) and spectrum_stats(centre(np.eye(50)))["n_eig_99"] >= 49
    print("qc_diagnostics selftest: OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["concentration", "alignment", "geometric", "spectrum"])
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--from-csv", action="store_true", help="alignment: rebuild the report from qi5_alignment.csv")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if args.command is None:
        ap.error("a command or --selftest is required")
    cfg = load_config(args.config)
    if args.command == "concentration":
        r = run_concentration(cfg, args.device)
        print(json.dumps({"haar_floor": r["haar_floor"], "checks": r["checks"],
                          "offdiag_var_c0.2": {k: {dd: v[dd]["c0.2"]["var"] for dd in v} for k, v in r["kernels"].items()}},
                         indent=1))
    elif args.command == "geometric":
        r = run_geometric(cfg, args.device)
        print(json.dumps({"checks": r["checks"], **{m: {d: {k: v.get("g_rbf_min_over_gamma") for k, v in e.items()}
                                                         for d, e in md.items()} for m, md in r["g"].items()}}, indent=1))
    elif args.command == "spectrum":
        r = run_spectrum(cfg, args.device)
        print(json.dumps({m: {d: {w: v["participation_ratio"] for w, v in e.items()} for d, e in md.items()}
                          for m, md in r["spectra"].items()}, indent=1))
    elif args.command == "alignment":
        r = run_alignment(cfg, args.device, args.from_csv)
        print(json.dumps({"checks": r["checks"], "spearman_all": r["spearman"]["all"],
                          "kta_choice_vs_cv": r["kta_choice_vs_cv"], "test_ap_at_kta_c": r["test_ap_at_kta_c"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
