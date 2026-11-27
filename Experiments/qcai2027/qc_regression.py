#!/usr/bin/env python3
"""Support vector regression of the crater parameters on the stage-2 kernels (plan step QI7.1).

Every true crater among the candidates gets three SVRs (``sklearn.svm.SVR(kernel="precomputed")``), one per target of
QB2.2: ``dx`` and ``dy`` (offset of the crater centre from the cell centre, east / south, cell units, in [-0.5, 0.5))
and ``log_r`` (natural log of the radius in km).  Decided with the user (2026-11-25):

- **the same kernel as the classifier** (pipeline section 3.1, point 6: no extra quantum cost): for every chosen QSVC
  run of QI4.3 (F-X; map, d, n, seed) its configuration (c, r, s) and its PCA of QI1.1 fitted on the whole training
  sample; the SVR uses the rows and columns of the positives of that kernel (``positive_kernels``).  The RBF-SVR uses
  the RBF kernel of the chosen QJ1 RBF run of the same unit (gamma = s x the median gamma of the whole sample), the
  classical twin.  Only C and epsilon of the SVR are chosen, per target, by the spatial CV of QI4.2 on the positives
  (``spatial_folds``, 5 deg blocks, folds stratified by diameter band; criterion: mean fold MAE of the standardised
  target; the kernel and its PCA are not refitted per fold: the PCA uses no labels);
- **test = the true craters among the 2,000 fixed test candidates of QH5** (pure regression, as if the classification
  were perfect: the same craters for every method and seed);
- **units = the QI4.3 grid**: every d of ``quantum.dims`` at the grid size and every size of ``qk_samples.sizes`` at
  d = ``regression.curve_d``, every seed; methods QK-A, QK-Z, QK-P, RBF and the trivial prediction (cell centre,
  median training radius).

Metrics (``regression_metrics``) on the test craters: centre error in km (distance between the predicted and the true
centre, east-west cell size scaled by cos(latitude) of the equirectangular WAC grid; mean and median), relative error
of the radius |R_hat - R| / R (mean and median), MAE and R^2 of every target, centre and radius error per diameter
band.  One row per run (unit x method) in ``results/qi7_runs.csv`` (completion marker: a unit already there is
skipped); ``regression_summary`` -> report section QI7.1 of ``results/qi_kernels.json``: mean / std over the seeds per
method and unit, paired difference of every quantum map against the RBF-SVR, learning curve.

Commands
--------
run        every unit, then the report.  GPU job (run_qcai_gpu.sh, ``--device cuda``: Aer states on the CPUs, kernel
           products on the GPU, as QI4), ~30 min; resumes after an interruption.
summary    report section QI7.1 from ``results/qi7_runs.csv``.  Seconds.

Usage
-----
    python Experiments/qcai2027/qc_regression.py run --device cuda     (GPU job: run_qcai_gpu.sh)
    python Experiments/qcai2027/qc_regression.py summary
    python Experiments/qcai2027/qc_regression.py --selftest
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_data import CONFIG_PATH, load_config, region_grid, save_report, written  # noqa: E402
from qc_experiments import (chosen_quantum_runs, config_rs, spatial_folds, stage2_kernels, stage2_paths,  # noqa: E402
                            upsert_runs)
from qc_kernels import features_path, kernel_projected, load_qk, median_gamma, projected_kernels, reduce_features  # noqa: E402

TARGETS = ("dx", "dy", "log_r")
QUANTUM = ("QK-A", "QK-Z", "QK-P")
METHODS = QUANTUM + ("RBF", "trivial")
RUN_COLS = ["step", "script", "run_id", "variant", "method", "d", "n", "seed", "c", "r", "s", "n_train_pos", "n_test_pos",
            *[f"{t}_{k}" for t in TARGETS for k in ("C", "epsilon", "cv_mae")],
            "centre_err_km_mean", "centre_err_km_median", "radius_rel_err_mean", "radius_rel_err_median",
            *[f"{t}_{k}" for t in TARGETS for k in ("mae", "r2")],
            *[f"{k}_band{b}" for k in ("centre_err_km", "radius_rel_err") for b in range(3)], "seconds", "timestamp"]


def runs_path(cfg: dict) -> Path:
    return stage2_paths(cfg, "qi7")["runs"]


# ---------------------------------------------------------------------------
# Targets and geometry
# ---------------------------------------------------------------------------


def cell_scale_km(cfg: dict) -> tuple[float, float]:
    """(km per cell north-south, degrees of latitude per pixel) of the WAC grid of R; east-west the cell is
    cos(latitude) times shorter (equirectangular grid)."""
    from qc_data import MOON_RADIUS_M

    g = region_grid(cfg, "wac")
    lat, _ = g.rc_to_latlon(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    deg_px = abs(float(lat[0] - lat[1]))
    return np.radians(deg_px) * MOON_RADIUS_M / 1000 * int(cfg["tiling"]["cell_px"]), deg_px


def crater_targets(cfg: dict, vid: str, n: int, seed: int) -> tuple[dict, dict]:
    """Targets of the training sample (``<vid>_n<n>_s<seed>`` of QH4, rows in the order of ``load_qk``) and of the
    2,000 test candidates (QH5): {dx, dy, log_r, lat} from ``cell_labels.npz`` at the cell of every sub-tile (NaN for
    the negatives); lat = latitude of the cell centre."""
    from qc_stage1 import candidates_path, stage1_dir

    out_dir, cell_px = cfg["paths"]["output"], int(cfg["tiling"]["cell_px"])
    lab = np.load(out_dir / "cell_labels.npz")
    z1 = np.load(candidates_path(cfg))
    grid = region_grid(cfg, "wac")
    sel = np.load(stage1_dir(cfg) / "qk_samples.npz")[f"{vid}_n{n}_s{seed}"]
    te = np.load(features_path(cfg))[f"{vid}_test_index"]
    res = []
    for set_name, idx in (("qk_pool", sel), ("test", te)):
        ci, cj = z1[f"{set_name}_ci"][idx], z1[f"{set_name}_cj"][idx]
        lat, _ = grid.rc_to_latlon((ci + 0.5) * cell_px - 0.5, (cj + 0.5) * cell_px - 0.5)
        res.append({**{t: lab[t][ci, cj].astype(float) for t in TARGETS}, "lat": np.asarray(lat, float)})
    return res[0], res[1]


# ---------------------------------------------------------------------------
# Kernels, fit, metrics
# ---------------------------------------------------------------------------


def positive_kernels(method: str, p: dict, x_tr: np.ndarray, x_te: np.ndarray, pos_tr: np.ndarray, pos_te: np.ndarray,
                     d: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """(K[train positives, train positives], K[test positives, train positives]) of the kernel of the classifier: PCA
    of QI1.1 fitted on the whole training sample, configuration ``p`` (c, r, s; RBF: s).  Fidelity kernels are
    computed on the positives only (the same values); QK-P and RBF take their gamma from the whole sample."""
    a_tr, a_te = reduce_features(x_tr, d, x_te)
    if method == "RBF":
        g = float(p["s"]) * median_gamma(a_tr)
        return projected_kernels(a_tr[pos_tr], a_te[pos_te], g)
    r, s = config_rs(p.get("r"), p.get("s"))
    if method == "QK-P":
        k_tr, k_te = stage2_kernels(method, a_tr, a_te[pos_te], float(p["c"]), r, s, device=device)
        return k_tr[np.ix_(pos_tr, pos_tr)], k_te[:, pos_tr]
    return stage2_kernels(method, a_tr[pos_tr], a_te[pos_te], float(p["c"]), r, s, device=device)


def svr_grid(cfg: dict) -> list[dict]:
    rg = cfg["regression"]
    return [{"C": float(C), "epsilon": float(e)} for C, e in itertools.product(cfg["stage2"]["svm_C"], rg["epsilon"])]


def fit_svr(k_tr: np.ndarray, y: np.ndarray, C: float, epsilon: float):
    """SVR on the precomputed kernel, target standardised with its training mean / std (epsilon in std units)."""
    from sklearn.svm import SVR

    mu, sd = float(np.mean(y)), float(np.std(y)) or 1.0
    return SVR(kernel="precomputed", C=C, epsilon=epsilon).fit(k_tr, (y - mu) / sd), mu, sd


def predict_svr(fitted, k_te: np.ndarray) -> np.ndarray:
    model, mu, sd = fitted
    return model.predict(k_te) * sd + mu


def cv_svr(cfg: dict, k_tr: np.ndarray, y: np.ndarray, folds: list) -> dict:
    """Spatial CV of C and epsilon for one target on the positive kernel ``k_tr``: the configuration with the smallest
    mean fold MAE of the standardised target (ties: the first of ``svr_grid``, smallest C)."""
    sd = float(np.std(y)) or 1.0
    best = None
    for p in svr_grid(cfg):
        mae = []
        for tr, va in folds:
            f = fit_svr(k_tr[np.ix_(tr, tr)], y[tr], p["C"], p["epsilon"])
            mae.append(np.mean(np.abs(predict_svr(f, k_tr[np.ix_(va, tr)]) - y[va])) / sd)
        if best is None or np.mean(mae) < best["cv_mae"] - 1e-12:
            best = {**p, "cv_mae": float(np.mean(mae))}
    return best


def regression_metrics(pred: dict, true: dict, band: np.ndarray, cell_km: float) -> dict:
    """Centre error (km), relative radius error and per-target MAE / R^2 of the predictions ``pred`` {dx, dy, log_r}
    against ``true`` {dx, dy, log_r, lat}; centre and radius error also per diameter band (0/1/2 = 1-2/2-4/4-8 km)."""
    ex = (pred["dx"] - true["dx"]) * cell_km * np.cos(np.radians(true["lat"]))
    ey = (pred["dy"] - true["dy"]) * cell_km
    centre = np.hypot(ex, ey)
    rad = np.abs(np.exp(pred["log_r"]) - np.exp(true["log_r"])) / np.exp(true["log_r"])
    out = {"centre_err_km_mean": centre.mean(), "centre_err_km_median": np.median(centre),
           "radius_rel_err_mean": rad.mean(), "radius_rel_err_median": np.median(rad)}
    for t in TARGETS:
        res = pred[t] - true[t]
        out[f"{t}_mae"] = np.mean(np.abs(res))
        out[f"{t}_r2"] = 1 - np.sum(res ** 2) / np.sum((true[t] - true[t].mean()) ** 2)
    for b in range(3):
        m = band == b
        out[f"centre_err_km_band{b}"] = centre[m].mean() if m.any() else np.nan
        out[f"radius_rel_err_band{b}"] = rad[m].mean() if m.any() else np.nan
    return {k: float(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def qi7_units(cfg: dict) -> list[tuple[int, int]]:
    """(d, n): the QI4.3 grid (every d at the grid size) and the learning curve at ``regression.curve_d``."""
    n0, d0 = int(cfg["stage2"]["main"]["n"]), int(cfg["regression"]["curve_d"])
    units = [(int(d), n0) for d in cfg["quantum"]["dims"]]
    return units + [(d0, int(n)) for n in cfg["qk_samples"]["sizes"] if (d0, int(n)) not in units]


def chosen_configs(cfg: dict, d: int, n: int, seed: int) -> dict:
    """{method: configuration} of the classifier of the unit: the chosen QSVC runs of QI4.3 (c, r, s) and the chosen RBF
    run of QJ1 (s)."""
    import pandas as pd

    vid = cfg["stage2"]["main"]["variant"]
    q = chosen_quantum_runs(cfg)
    q = q[(q["d"] == d) & (q["n"] == n) & (q["seed"] == seed)]
    out = {row["map"]: {"c": row["c"], "r": row["r"], "s": row["s"]} for _, row in q.iterrows()}
    j = pd.read_csv(cfg["paths"]["results"] / "qj1_runs.csv")
    j = j[(j["map"] == "RBF") & (j["variant"] == vid) & (j["d"] == d) & (j["n"] == n) & (j["seed"] == seed)]
    if len(q) != len(QUANTUM) or len(j) != 1:
        raise ValueError(f"missing chosen QI4.3 / QJ1 runs for d {d} n {n} seed {seed}")
    out["RBF"] = {"c": np.nan, "r": np.nan, "s": float(j["s"].iloc[0])}
    return out


def run_unit(cfg: dict, d: int, n: int, seed: int, z, cell_km: float, device: str) -> list[dict]:
    """One unit (d, n, seed): the SVRs of every method on the positives of the training sample, scored on the test
    craters; the trivial prediction (cell centre, median training log radius)."""
    vid = cfg["stage2"]["main"]["variant"]
    x_tr, i_tr, x_te, i_te = load_qk(cfg, vid, n, seed, z)
    t_tr, t_te = crater_targets(cfg, vid, n, seed)
    pos_tr, pos_te = np.flatnonzero(i_tr["cls"] == 1), np.flatnonzero(i_te["cls"] == 1)
    true = {k: v[pos_te] for k, v in t_te.items()}
    y = {t: t_tr[t][pos_tr] for t in TARGETS}
    if any(np.isnan(v).any() for v in (*y.values(), *(true[t] for t in TARGETS))):
        raise ValueError("a positive without crater parameters")
    folds = spatial_folds(i_tr["band"][pos_tr], i_tr["block"][pos_tr], int(cfg["stage2"]["cv_folds"]), seed)
    band = i_te["band"][pos_te]
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    cfgs = chosen_configs(cfg, d, n, seed)
    rows = []
    for m in METHODS:
        t0 = time.time()
        row = {"step": "QI7.1", "script": "qc_regression.py", "run_id": f"{vid}_{m}_d{d}_n{n}_s{seed}", "variant": vid,
               "method": m, "d": d, "n": n, "seed": seed, "n_train_pos": len(pos_tr), "n_test_pos": len(pos_te)}
        if m == "trivial":
            pred = {"dx": np.zeros(len(pos_te)), "dy": np.zeros(len(pos_te)),
                    "log_r": np.full(len(pos_te), np.median(y["log_r"]))}
        else:
            p = cfgs[m]
            row.update({k: p[k] for k in ("c", "r", "s")})
            k_tr, k_te = positive_kernels(m, p, x_tr, x_te, pos_tr, pos_te, d, device)
            pred = {}
            for t in TARGETS:
                best = cv_svr(cfg, k_tr, y[t], folds)
                row.update({f"{t}_C": best["C"], f"{t}_epsilon": best["epsilon"], f"{t}_cv_mae": best["cv_mae"]})
                pred[t] = predict_svr(fit_svr(k_tr, y[t], best["C"], best["epsilon"]), k_te)
        row.update(regression_metrics(pred, true, band, cell_km))
        row.update({"seconds": round(time.time() - t0, 2), "timestamp": stamp})
        rows.append(row)
        print(f"QI7.1 {row['run_id']}: centre {row['centre_err_km_mean']:.3f} km, radius {row['radius_rel_err_mean']:.3f}, "
              f"R2 dx / dy / log_r {row['dx_r2']:.3f} / {row['dy_r2']:.3f} / {row['log_r_r2']:.3f}", flush=True)
    return rows


def save_rows(cfg: dict, rows: list[dict]) -> None:
    import pandas as pd

    upsert_runs(runs_path(cfg), pd.DataFrame(rows).reindex(columns=RUN_COLS))


def run_regression(cfg: dict, device: str = "cuda") -> None:
    """Every unit of ``qi7_units`` and seed of ``qk_samples.seeds``; units already in ``qi7_runs.csv`` are skipped."""
    import pandas as pd

    z = np.load(features_path(cfg))
    cell_km, _ = cell_scale_km(cfg)
    done = set(pd.read_csv(runs_path(cfg))["run_id"]) if runs_path(cfg).exists() else set()
    vid = cfg["stage2"]["main"]["variant"]
    for d, n in qi7_units(cfg):
        for seed in [int(s) for s in cfg["qk_samples"]["seeds"]]:
            if all(f"{vid}_{m}_d{d}_n{n}_s{seed}" in done for m in METHODS):
                continue
            t0 = time.time()
            save_rows(cfg, run_unit(cfg, d, n, seed, z, cell_km, device))
            print(f"QI7.1 unit d{d}_n{n}_s{seed}: {time.time() - t0:.0f} s", flush=True)
    written(runs_path(cfg))
    regression_summary(cfg)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

SUMMARY_COLS = ("centre_err_km_mean", "centre_err_km_median", "radius_rel_err_mean", "radius_rel_err_median",
                "dx_r2", "dy_r2", "log_r_r2", "centre_err_km_band0", "centre_err_km_band1", "centre_err_km_band2",
                "radius_rel_err_band0", "radius_rel_err_band1", "radius_rel_err_band2")


def regression_summary(cfg: dict) -> dict:
    """Report section QI7.1: per unit (d, n) and method [mean, std] over the seeds of the metrics, the most chosen C /
    epsilon per target, and for every quantum map the paired difference against the RBF-SVR (mean, std, seeds in which
    the quantum map is better) of the centre error, the radius error and the R^2 of every target."""
    import pandas as pd

    df = pd.read_csv(runs_path(cfg))
    units = {}
    for (d, n), g in df.groupby(["d", "n"]):
        u = {"seeds": int(g["seed"].nunique()), "test_craters": int(g["n_test_pos"].iloc[0]),
             "train_craters": int(g["n_train_pos"].iloc[0])}
        for m in METHODS:
            gm = g[g["method"] == m]
            if gm.empty:
                continue
            u[m] = {k: [round(float(gm[k].mean()), 4), round(float(gm[k].std(ddof=0)), 4)] for k in SUMMARY_COLS}
            if m != "trivial":
                u[m]["chosen_C_epsilon"] = {t: gm[[f"{t}_C", f"{t}_epsilon"]].astype(str).agg("/".join, axis=1).mode().iloc[0]
                                            for t in TARGETS}
        ref = g[g["method"] == "RBF"].set_index("seed")
        for m in QUANTUM:
            q = g[g["method"] == m].set_index("seed")
            if q.empty or ref.empty:
                continue
            diff = {}
            for k, better in (("centre_err_km_mean", "lower"), ("radius_rel_err_mean", "lower"), ("dx_r2", "higher"),
                              ("dy_r2", "higher"), ("log_r_r2", "higher")):
                dlt = (q[k] - ref[k]).dropna()
                wins = int((dlt < 0).sum() if better == "lower" else (dlt > 0).sum())
                diff[k] = {"mean": round(float(dlt.mean()), 4), "std": round(float(dlt.std(ddof=0)), 4),
                           "seeds_quantum_better": wins}
            u[f"{m}_minus_RBF"] = diff
        units[f"d{d}_n{n}"] = u
    cell_km, _ = cell_scale_km(cfg)
    rep = {"variant": cfg["stage2"]["main"]["variant"], "targets": list(TARGETS), "methods": list(METHODS),
           "svr_grid": {"C": [float(v) for v in cfg["stage2"]["svm_C"]], "epsilon_std_units": [float(v) for v in cfg["regression"]["epsilon"]]},
           "cell_km_north_south": round(float(cell_km), 4), "units": units,
           "note": ("[mean, std] over the seeds; test = the true craters among the 2,000 fixed test candidates (QH5); "
                    "kernel = that of the chosen QSVC of QI4.3 (QK maps) or of the chosen QJ1 RBF (RBF) of the same unit, "
                    "restricted to the positives; C, epsilon per target by the spatial CV of QI4.2 on the training "
                    "positives (mean fold MAE of the standardised target); trivial = cell centre and median training "
                    "radius; centre error with the east-west cell size x cos(lat); R^2 against the test mean "
                    "(0 = no better than the mean, < 0 worse); runs in results/qi7_runs.csv")}
    save_report(cfg, "QI7.1", rep)
    return rep


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def selftest() -> None:
    """Metrics on known errors; the SVR recovers a smooth target from an RBF kernel and beats the trivial prediction;
    the CV picks a configuration of the grid."""
    rng = np.random.default_rng(0)
    true = {"dx": np.array([0.1, -0.2]), "dy": np.array([0.0, 0.3]), "log_r": np.log([1.0, 2.0]), "lat": np.array([0.0, 60.0])}
    pred = {"dx": np.array([0.4, -0.2]), "dy": np.array([0.4, 0.3]), "log_r": np.log([1.5, 2.0])}
    m = regression_metrics(pred, true, np.array([0, 1]), 1.6)
    assert np.isclose(m["centre_err_km_mean"], np.hypot(0.3 * 1.6, 0.4 * 1.6) / 2)
    assert np.isclose(m["radius_rel_err_mean"], 0.25) and np.isclose(m["centre_err_km_band1"], 0.0)
    pred2 = {**pred, "dx": np.array([0.1, -0.2 + 0.1])}                 # east error at lat 60 counts half
    assert np.isclose(regression_metrics(pred2, true, np.array([0, 1]), 1.6)["centre_err_km_band1"], 0.1 * 1.6 * 0.5)
    x = rng.normal(size=(200, 3))
    y = np.sin(x[:, 0]) + 0.05 * rng.normal(size=200)
    k = kernel_projected(x, x, median_gamma(x))
    cfg = {"stage2": {"svm_C": [0.1, 1, 10]}, "regression": {"epsilon": [0.05, 0.2]}}
    folds = [(np.arange(200)[np.arange(200) % 4 != f], np.arange(200)[np.arange(200) % 4 == f]) for f in range(4)]
    best = cv_svr(cfg, k[:150, :150], y[:150], [(tr[tr < 150], va[va < 150]) for tr, va in folds])
    assert best["C"] in (0.1, 1, 10) and best["epsilon"] in (0.05, 0.2)
    p = predict_svr(fit_svr(k[:150, :150], y[:150], best["C"], best["epsilon"]), k[150:, :150])
    assert np.mean(np.abs(p - y[150:])) < 0.5 * np.mean(np.abs(np.median(y[:150]) - y[150:]))
    print(f"qc_regression selftest: OK (SVR MAE {np.mean(np.abs(p - y[150:])):.3f} vs trivial "
          f"{np.mean(np.abs(np.median(y[:150]) - y[150:])):.3f}; best C {best['C']} epsilon {best['epsilon']})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["run", "summary"])
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if args.command is None:
        ap.error("a command or --selftest is required")
    cfg = load_config(args.config)
    if args.command == "run":
        run_regression(cfg, args.device)
    else:
        print(json.dumps(regression_summary(cfg)["units"].get(f"d{cfg['regression']['curve_d']}_n{cfg['stage2']['main']['n']}", {}), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
