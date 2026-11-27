#!/usr/bin/env python3
"""Noise and resources of the quantum kernel (plan phase QK, RQ4): device noise model and readout mitigation.

QK1 - Device noise model.  The QK-Z fidelity kernel (r = ``noise.reps``) is estimated as on hardware: one
compute-uncompute circuit U(x) U(x')^+ per pair (``qc_kernels.feature_map``), transpiled **once** per device and d
with all qubits measured (``device_circuit``: ``transpile`` at ``noise.optimization_level`` on the fake backend, the
angles left as parameters, so every pair runs on the same physical qubits), then bound to the angles of every pair
and run on ``AerSimulator.from_backend`` (the noise model of the calibration data of the fake backend: gate errors,
thermal relaxation, readout errors) with ``noise.shots`` shots, method ``noise.method``.  K(x, x') = share of the
all-zero outcomes.  Devices ``noise.backends`` (``qiskit_ibm_runtime.fake_provider``), d in ``noise.dims``.
Data (``qk1_data``): the QK training sample of ``noise.variant`` of size ``noise.n_train`` (first seed of
``qk_samples``) and ``noise.n_test`` test candidates of QH5 drawn class-balanced (half positives, seed ``seed``), PCA
of QI1.1 fitted on the training sample.  Hyper-parameters per d (``choose_hparams``): c and C by the spatial CV of
QI4.2 (``qc_experiments.cv_search``) on the exact kernel of the training sample, QK-Z with r = ``noise.reps`` only;
the same c and C for every kernel estimate.  Circuits per device and d (``pair_lists``): the n(n-1)/2 training pairs,
the n_test x n training-test pairs and the n diagonal pairs (measured for the report only: the training diagonal is
set to 1, known on hardware, as in QI3.3).  The share of zeros of every circuit is kept in
``Dataset/Moon/qcai2027/noise/qk_noise.npz`` (one array per device, d and estimate, NaN = not run yet), written after
every batch of ``noise.batch`` circuits, so a resubmitted job continues where it stopped.

QK2 - Readout-error mitigation (``mthree``), in the same run on the same counts: M3 calibrated once per device and d on
the measured physical qubits of the transpiled circuit (``cals_from_system`` on the same noisy simulator, 10,000
shots; the calibration matrices are kept in the npz and reused on a restart), then ``apply_correction`` and K = the
quasi-probability of the all-zero outcome, clipped to [0, 1].

Report (``noise_summary``, sections QK1 and QK2 of ``results/qk_noise.json``), per device and d: the transpiled
circuit (depth, two-qubit gates, physical qubits), the check of the transpiled circuit without noise against the exact
kernel (QI3.1), and for every estimate of the kernel - exact, shots only (Binomial(N, K) / N, QI3.3), device noise,
device noise + M3 - the relative Frobenius error against the exact kernel (training matrix after the PSD projection
of QI3.3, test matrix as estimated), the mean off-diagonal value against the uniform floor 1 / 2^d, the fit
K_est = a K + b (a global depolarising channel gives b = (1 - a) / 2^d), the mean measured diagonal, and the SVM
(``qc_experiments.fit_svc``, Platt probability >= ``stage2.default_threshold``) trained on the estimated training
kernel and scored on the estimated test kernel: F1, precision, recall, AP, agreement of the predictions with the exact
SVM.

Commands
--------
run        QK1 + QK2 simulation -> ``Dataset/Moon/qcai2027/noise/qk_noise.npz``, then the report.  Hours (d = 12
           density matrices): CPU job (run_qcai_cpu.sh); resubmit to continue.
summary    QK1 + QK2 report from the saved npz -> sections QK1, QK2 of ``results/qk_noise.json``.  Seconds.

Usage
-----
    python Experiments/qcai2027/qc_noise.py run        (CPU job: run_qcai_cpu.sh)
    python Experiments/qcai2027/qc_noise.py summary
    python Experiments/qcai2027/qc_noise.py --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_data import CONFIG_PATH, load_config, save_report, written  # noqa: E402
from qc_experiments import cv_search, cv_table, f1, fit_svc, select_best, svc_scores  # noqa: E402
from qc_kernels import feature_angles, feature_map, fidelity_kernels, load_qk, psd_project, reduce_features, shot_kernel  # noqa: E402
from qc_probes import average_precision  # noqa: E402

ESTIMATES = ("raw", "m3")                     # device noise, device noise + M3 readout mitigation


def noise_path(cfg: dict) -> Path:
    return cfg["paths"]["output"] / "noise" / "qk_noise.npz"


def fake_backend(name: str):
    from qiskit_ibm_runtime import fake_provider

    return getattr(fake_provider, name)()


# ---------------------------------------------------------------------------
# Circuits on the device
# ---------------------------------------------------------------------------


def device_circuit(name: str, d: int, reps: int, backend, opt_level: int, seed: int) -> dict:
    """Compute-uncompute circuit U(theta) U(phi)^+ of the feature map ``name``, all qubits measured, transpiled once
    on ``backend`` with the angles as parameters: {circuit, theta, phi (parameter lists), mapping (clbit -> physical
    qubit), summary (depth, two-qubit gates, gates, physical qubits)}."""
    from mthree.utils import final_measurement_mapping
    from qiskit import transpile
    from qiskit.circuit import ParameterVector

    u = feature_map(name, d, reps)
    theta = sorted(u.parameters, key=lambda p: p.index)
    phi = list(ParameterVector("phi", d))
    qc = u.compose(u.assign_parameters(dict(zip(theta, phi))).inverse())
    qc.measure_all()
    tq = transpile(qc, backend, optimization_level=opt_level, seed_transpiler=seed)
    mapping = final_measurement_mapping(tq)
    ops = tq.count_ops()
    two_q = sum(v for k, v in ops.items() if k in ("ecr", "cz", "cx", "rzz"))
    return {"circuit": tq, "theta": theta, "phi": phi, "mapping": mapping,
            "summary": {"depth": tq.depth(), "two_qubit_gates": int(two_q),
                        "gates": int(sum(v for k, v in ops.items() if k not in ("measure", "barrier"))),
                        "ops": {k: int(v) for k, v in ops.items() if k != "barrier"},
                        "physical_qubits": [int(mapping[k]) for k in sorted(mapping)]}}


def bind(dev: dict, a: np.ndarray, b: np.ndarray) -> list:
    """The transpiled circuit of ``dev`` bound to every row pair (a_k, b_k) of angles: U(a_k) U(b_k)^+."""
    keys = dev["theta"] + dev["phi"]
    return [dev["circuit"].assign_parameters(dict(zip(keys, np.concatenate([x, y]).astype(float)))) for x, y in zip(a, b)]


def ideal_zero_probability(dev: dict, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """P(|0...0>) of the bound transpiled circuits without noise (Aer, exact probabilities of the measured physical
    qubits): checks that transpilation, layout and binding keep the kernel of QI3.1."""
    from qiskit_aer import AerSimulator

    qubits = [dev["mapping"][k] for k in sorted(dev["mapping"])]
    circs = []
    for qc in bind(dev, a, b):
        c = qc.remove_final_measurements(inplace=False)
        c.save_probabilities(qubits=qubits)
        circs.append(c)
    res = AerSimulator(method="statevector").run(circs).result()
    return np.array([res.data(k)["probabilities"][0] for k in range(len(circs))])


def pair_lists(n_tr: int, n_te: int) -> dict:
    """Pairs of one device run, as (first, second) row indices into the angles: ``train`` = training pairs i < j,
    ``test`` = (test p, training q) for every p, q, ``diag`` = (i, i).  The order is the order of the saved arrays."""
    iu = np.triu_indices(n_tr, 1)
    p, q = np.meshgrid(np.arange(n_te), np.arange(n_tr), indexing="ij")
    return {"train": (iu[0], iu[1]), "test": (p.ravel(), q.ravel()), "diag": (np.arange(n_tr), np.arange(n_tr))}


def all_pairs(th_tr: np.ndarray, th_te: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """(first angles, second angles, slices per part) of all the circuits of one device run, parts in the order
    train, test, diag."""
    pl = pair_lists(len(th_tr), len(th_te))
    a = np.concatenate([th_tr[pl["train"][0]], th_te[pl["test"][0]], th_tr[pl["diag"][0]]])
    b = np.concatenate([th_tr[pl["train"][1]], th_tr[pl["test"][1]], th_tr[pl["diag"][1]]])
    sizes = [len(pl[k][0]) for k in ("train", "test", "diag")]
    edges = np.cumsum([0] + sizes)
    return a, b, {k: slice(edges[i], edges[i + 1]) for i, k in enumerate(("train", "test", "diag"))}


def assemble(p: np.ndarray, n_tr: int, n_te: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(K_train with diagonal 1, K_test (n_te, n_tr), measured diagonal) from the values ``p`` of ``all_pairs``."""
    _, _, sl = all_pairs(np.zeros((n_tr, 1)), np.zeros((n_te, 1)))
    k_tr = np.eye(n_tr)
    iu = np.triu_indices(n_tr, 1)
    k_tr[iu] = p[sl["train"]]
    k_tr[(iu[1], iu[0])] = p[sl["train"]]
    return k_tr, p[sl["test"]].reshape(n_te, n_tr), p[sl["diag"]]


# ---------------------------------------------------------------------------
# Data and hyper-parameters
# ---------------------------------------------------------------------------


def qk1_data(cfg: dict) -> dict:
    """Training sample (``noise.variant``, ``noise.n_train``, first seed) and ``noise.n_test`` class-balanced test
    candidates of QH5: {x_tr, y_tr, blocks, x_te, y_te, te_rows (rows of the 2,000 test candidates), seed}."""
    nz = cfg["noise"]
    seed = int(cfg["qk_samples"]["seeds"][0])
    x_tr, i_tr, x_te, i_te = load_qk(cfg, nz["variant"], int(nz["n_train"]), seed)
    y_te = i_te["cls"].astype(int)
    rng = np.random.default_rng([int(cfg["seed"]), int(nz["n_test"])])
    half = int(nz["n_test"]) // 2
    rows = np.sort(np.concatenate([rng.choice(np.flatnonzero(y_te == v), half, replace=False) for v in (1, 0)]))
    return {"x_tr": x_tr, "y_tr": i_tr["cls"].astype(int), "blocks": i_tr["block"], "x_te": x_te[rows],
            "y_te": y_te[rows], "te_rows": rows, "seed": seed}


def choose_hparams(cfg: dict, data: dict, d: int) -> dict:
    """QI4.2 CV (``cv_search``) on the training sample, QK-Z with r = ``noise.reps`` only, exact kernels: the chosen
    {c, C, cv_ap}."""
    nz = cfg["noise"]
    cfg_r = {**cfg, "quantum": {**cfg["quantum"], "zz_reps": [int(nz["reps"])]}}
    rows, _ = cv_search(data["x_tr"], data["y_tr"], data["blocks"], d, cfg_r, maps=(nz["map"],), seed=data["seed"],
                        device="cpu")
    best = select_best(cv_table(rows), nz["map"])
    return {"c": float(best["c"]), "C": float(best["C"]), "cv_ap": round(float(best["cv_ap"]), 4)}


def angles(cfg: dict, data: dict, d: int, c: float) -> tuple[np.ndarray, np.ndarray]:
    a_tr, a_te = reduce_features(data["x_tr"], d, data["x_te"])
    return feature_angles(a_tr, c), feature_angles(a_te, c)


# ---------------------------------------------------------------------------
# QK1 + QK2 simulation
# ---------------------------------------------------------------------------


def load_store(cfg: dict) -> dict:
    p = noise_path(cfg)
    return dict(np.load(p)) if p.exists() else {}


def save_store(cfg: dict, store: dict) -> None:
    """Atomic rewrite of the npz (a job killed while writing leaves the previous file)."""
    p = noise_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npz")
    np.savez(tmp, **store)
    os.replace(tmp, p)
    written(p)


def m3_mitigator(sim, dev: dict, store: dict, key: str):
    """M3 calibrated on the measured physical qubits of ``dev`` (on ``sim``, 10,000 shots), or from the matrices
    saved under ``key`` in ``store`` (restart); the matrices are added to ``store``."""
    import mthree

    m3 = mthree.M3Mitigation(sim)
    qubits = [dev["mapping"][k] for k in sorted(dev["mapping"])]
    if key in store:
        mats = [None] * m3.num_qubits
        for q, m in zip(qubits, store[key]):
            mats[q] = m
        m3.cals_from_matrices(mats)
    else:
        m3.cals_from_system(qubits, shots=10000, async_cal=False)
        store[key] = np.stack([m3.single_qubit_cals[q] for q in qubits])
    return m3


def run_device(cfg: dict, store: dict, bname: str, d: int, th_tr: np.ndarray, th_te: np.ndarray) -> dict:
    """QK1 + QK2 circuits of one device and d, batch by batch, skipping the batches already in ``store``."""
    from qiskit_aer import AerSimulator

    nz = cfg["noise"]
    backend = fake_backend(bname)
    dev = device_circuit(nz["map"], d, int(nz["reps"]), backend, int(nz["optimization_level"]), int(cfg["seed"]))
    sim = AerSimulator.from_backend(backend, method=nz["method"], max_parallel_experiments=0)
    a, b, _ = all_pairs(th_tr, th_te)
    key = f"{bname}_d{d}"
    for e in ESTIMATES:
        store.setdefault(f"{key}_{e}", np.full(len(a), np.nan))
    m3 = m3_mitigator(sim, dev, store, f"{key}_m3cals")
    shots, batch, zero = int(nz["shots"]), int(nz["batch"]), "0" * d
    t0, todo = time.time(), np.flatnonzero(np.isnan(store[f"{key}_raw"]))
    for s in range(0, todo.size, batch):
        k = todo[s:s + batch]
        res = sim.run(bind(dev, a[k], b[k]), shots=shots, seed_simulator=int(cfg["seed"]) + int(k[0])).result()
        counts = [res.get_counts(i) for i in range(len(k))]
        quasi = m3.apply_correction(counts, [dev["mapping"]] * len(k))
        store[f"{key}_raw"][k] = [c.get(zero, 0) / shots for c in counts]
        store[f"{key}_m3"][k] = np.clip([q.get(zero, 0.0) for q in quasi], 0, 1)
        save_store(cfg, store)
        done = s + len(k)
        print(f"QK1 {key}: {done} / {todo.size} circuits, {time.time() - t0:.0f} s", flush=True)
    return dev


def run_noise(cfg: dict) -> None:
    """QK1 + QK2: hyper-parameters per d, then every device and d; the report when everything is done."""
    nz = cfg["noise"]
    data = qk1_data(cfg)
    store = load_store(cfg)
    store["test_rows"] = data["te_rows"]
    for d in [int(v) for v in nz["dims"]]:
        if f"hp_d{d}" not in store:
            hp = choose_hparams(cfg, data, d)
            store[f"hp_d{d}"] = np.array([hp["c"], hp["C"], hp["cv_ap"]])
            save_store(cfg, store)
        c = float(store[f"hp_d{d}"][0])
        print(f"QK1 d {d}: c {c}, C {store[f'hp_d{d}'][1]}, CV AP {store[f'hp_d{d}'][2]}", flush=True)
        th_tr, th_te = angles(cfg, data, d, c)
        for bname in nz["backends"]:
            run_device(cfg, store, bname, d, th_tr, th_te)
    noise_summary(cfg)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def svm_metrics(k_tr: np.ndarray, k_te: np.ndarray, y_tr: np.ndarray, y_te: np.ndarray, C: float, thr: float) -> dict:
    model = fit_svc(k_tr, y_tr, C)
    sc = svc_scores(model, k_te)
    pred = (sc["proba"] >= thr).astype(int)
    tp = int(np.sum(pred[y_te == 1]))
    return {"pred": pred, "f1": f1(pred, y_te), "precision": tp / max(int(pred.sum()), 1), "recall": tp / int((y_te == 1).sum()),
            "ap": average_precision(sc["proba"], y_te)}


def estimate_metrics(k_tr_est, k_te_est, diag, exact: tuple, data: dict, hp: dict, thr: float, d: int,
                     pred_exact: np.ndarray | None) -> dict:
    """Error, concentration and SVM metrics of one kernel estimate against the exact kernel ``exact``."""
    k_tr, k_te = exact
    n = len(k_tr)
    off = ~np.eye(n, dtype=bool)
    k_psd, info = psd_project(k_tr_est)
    rel = lambda e, x: round(float(np.linalg.norm(e - x) / np.linalg.norm(x)), 4)
    x_all = np.concatenate([k_tr[off], k_te.ravel()])
    y_all = np.concatenate([k_tr_est[off], k_te_est.ravel()])
    slope, icpt = np.polyfit(x_all, y_all, 1)
    svm = svm_metrics(k_psd, k_te_est, data["y_tr"], data["y_te"], hp["C"], thr)
    out = {"rel_frobenius_error_train_psd": rel(k_psd, k_tr), "rel_frobenius_error_test": rel(k_te_est, k_te),
           "offdiag_mean_train": round(float(k_tr_est[off].mean()), 4), "mean_test": round(float(k_te_est.mean()), 4),
           "fit_slope_a": round(float(slope), 4), "fit_intercept_b": round(float(icpt), 5),
           "depolarising_b_expected": round(float((1 - slope) / 2 ** d), 5),
           "min_eigenvalue_before_psd": float(f"{info['min_eigenvalue']:.3g}"), "eigenvalues_clipped": info["eigenvalues_clipped"],
           **{k: round(float(v), 4) for k, v in svm.items() if k != "pred"}}
    if diag is not None:
        out["measured_diagonal_mean"] = round(float(np.mean(diag)), 4)
    if pred_exact is not None:
        out["prediction_agreement_with_exact"] = round(float(np.mean(svm["pred"] == pred_exact)), 4)
    return out, svm["pred"]


def noise_summary(cfg: dict) -> dict:
    """QK1 and QK2 report sections from the saved npz (only the devices and d whose circuits are all done)."""

    nz = cfg["noise"]
    store, data = load_store(cfg), qk1_data(cfg)
    if not np.array_equal(store.get("test_rows", data["te_rows"]), data["te_rows"]):
        raise ValueError("test candidates differ from the saved run: rerun qc_noise.py run")
    thr, shots, n, m = float(cfg["stage2"]["default_threshold"]), int(nz["shots"]), len(data["y_tr"]), len(data["y_te"])
    qk1, qk2, checks = {}, {}, {}
    for d in [int(v) for v in nz["dims"]]:
        if f"hp_d{d}" not in store:
            continue
        c, C, cv_ap = (float(v) for v in store[f"hp_d{d}"])
        hp = {"c": c, "C": C}
        th_tr, th_te = angles(cfg, data, d, c)
        exact = fidelity_kernels(nz["map"], th_tr, th_te, int(nz["reps"]), "cpu")
        ex, pred_ex = estimate_metrics(exact[0], exact[1], None, exact, data, hp, thr, d, None)
        rng = np.random.default_rng([int(cfg["seed"]), shots, d])
        k_sh = (shot_kernel(exact[0], shots, rng, symmetric=True), shot_kernel(exact[1], shots, rng))
        sh, _ = estimate_metrics(k_sh[0], k_sh[1], None, exact, data, hp, thr, d, pred_ex)
        unit = {"c": c, "C": C, "cv_ap": cv_ap, "uniform_floor": float(f"{2.0 ** -d:.3g}"), "exact": ex, f"shots_N{shots}": sh}
        for bname in nz["backends"]:
            key = f"{bname}_d{d}"
            if key + "_raw" not in store or np.isnan(store[key + "_raw"]).any():
                continue
            dev = device_circuit(nz["map"], d, int(nz["reps"]), fake_backend(bname), int(nz["optimization_level"]), int(cfg["seed"]))
            rng = np.random.default_rng([int(cfg["seed"]), d])
            i, j = rng.integers(0, n, 10), rng.integers(0, m, 10)
            dif = np.abs(ideal_zero_probability(dev, th_te[j], th_tr[i]) - exact[1][j, i]).max()
            checks[f"{key}_transpiled_vs_exact_max_abs_diff"] = float(f"{dif:.1e}")
            res = {}
            for e in ESTIMATES:
                k_tr, k_te, diag = assemble(store[f"{key}_{e}"], n, m)
                res[e], _ = estimate_metrics(k_tr, k_te, diag, exact, data, hp, thr, d, pred_ex)
            cals = store[key + "_m3cals"]                       # (d, 2, 2): columns = prepared state
            qk1[f"{key}"] = {"circuit": dev["summary"], "device_noise": res["raw"]}
            qk2[f"{key}"] = {"readout_error_mean": round(float(np.mean([1 - np.trace(x) / 2 for x in cals])), 4),
                             "mitigated": res["m3"]}
            print(f"QK1 {key}: F1 exact / shots / noise / M3 {ex['f1']:.3f} / {sh['f1']:.3f} / {res['raw']['f1']:.3f} / "
                  f"{res['m3']['f1']:.3f}; rel. error test noise / M3 {res['raw']['rel_frobenius_error_test']} / "
                  f"{res['m3']['rel_frobenius_error_test']}; off-diag mean exact {ex['offdiag_mean_train']} noise "
                  f"{res['raw']['offdiag_mean_train']}", flush=True)
        qk1[f"d{d}"] = unit
    checks["transpiled_below_1e-8"] = all(v < 1e-8 for k, v in checks.items() if k.endswith("diff"))
    common = {"variant": nz["variant"], "map": nz["map"], "reps": int(nz["reps"]), "n_train": n,
              "n_test": f"{m} class-balanced from the QH5 test candidates", "shots": shots, "method": nz["method"],
              "optimization_level": int(nz["optimization_level"]), "threshold": thr}
    note = ("K = share of all-zero outcomes of U(x) U(x')^+ transpiled once per device and d (angles as parameters) on "
            "AerSimulator.from_backend(fake backend); training diagonal set to 1 (measured_diagonal_mean = the measured "
            "one); training matrix projected on the PSD cone before the SVM, test matrix as estimated; c, C from the "
            "QI4.2 CV on the exact kernel, the same for every estimate; fit = least squares K_est = a K + b on the "
            "off-diagonal training and the test values, depolarising_b_expected = (1 - a) / 2^d; no idle-time noise "
            "(circuits not scheduled)")
    save_report(cfg, "QK1", {**common, "results": qk1, "checks": checks, "note": note})
    if qk2:
        save_report(cfg, "QK2", {**common, "results": qk2,
                                 "note": "M3 (mthree) calibrated on the measured physical qubits with 10,000 shots on the "
                                         "same noisy simulator (calibration not seeded), apply_correction on the same "
                                         "counts as QK1, all-zero quasi-probability clipped to [0, 1]; "
                                         "readout_error_mean = mean over the qubits of 1 - trace / 2 of the calibration "
                                         "matrix"})
    return {"QK1": qk1, "QK2": qk2}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def selftest() -> None:
    """Pair enumeration and kernel assembly; the transpiled circuit without noise against the exact kernel; the noisy
    estimate close to the exact kernel on a 3-qubit device run, M3 closer than the raw counts on average."""
    from qiskit_aer import AerSimulator

    rng = np.random.default_rng(0)
    th_tr, th_te = rng.normal(0, 0.3, (5, 3)), rng.normal(0, 0.3, (4, 3))
    a, b, sl = all_pairs(th_tr, th_te)
    assert len(a) == 5 * 4 // 2 + 4 * 5 + 5
    k_tr, k_te = fidelity_kernels("QK-Z", th_tr, th_te, 1, "cpu")
    exact_pairs = np.concatenate([k_tr[np.triu_indices(5, 1)], k_te.ravel(), np.ones(5)])
    kt, ke, dg = assemble(exact_pairs, 5, 4)
    assert np.allclose(kt, k_tr) and np.allclose(ke, k_te) and np.allclose(dg, 1)
    backend = fake_backend("FakeFez")
    dev = device_circuit("QK-Z", 3, 1, backend, 3, 0)
    assert np.abs(ideal_zero_probability(dev, a, b) - exact_pairs).max() < 1e-8
    sim = AerSimulator.from_backend(backend, method="density_matrix")
    res = sim.run(bind(dev, a, b), shots=4096, seed_simulator=1).result()
    counts = [res.get_counts(i) for i in range(len(a))]
    raw = np.array([cnt.get("000", 0) / 4096 for cnt in counts])
    store: dict = {}
    m3 = m3_mitigator(sim, dev, store, "k")
    mit = np.clip([q.get("000", 0.0) for q in m3.apply_correction(counts, [dev["mapping"]] * len(a))], 0, 1)
    assert np.abs(raw - exact_pairs).max() < 0.2 and np.mean(np.abs(mit - exact_pairs)) < np.mean(np.abs(raw - exact_pairs))
    m3b = m3_mitigator(sim, dev, store, "k")                       # restart from the saved matrices
    mit_b = np.clip([q.get("000", 0.0) for q in m3b.apply_correction(counts, [dev["mapping"]] * len(a))], 0, 1)
    assert np.allclose(mit, mit_b)
    print(f"qc_noise selftest: OK (3 qubits FakeFez: mean |K_est - K| raw {np.mean(np.abs(raw - exact_pairs)):.3f}, "
          f"M3 {np.mean(np.abs(mit - exact_pairs)):.3f})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["run", "summary"])
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if args.command is None:
        ap.error("a command or --selftest is required")
    cfg = load_config(args.config)
    if args.command == "run":
        run_noise(cfg)
    else:
        print(json.dumps(noise_summary(cfg)["QK1"].get("d4", {}), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
