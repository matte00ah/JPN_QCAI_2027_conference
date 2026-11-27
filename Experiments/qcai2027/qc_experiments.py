#!/usr/bin/env python3
"""Stage-2 classifier and experiments of the QCAI 2027 experiment (plan phases QI4, QI6, QJ1).

The stage-2 classifier separates craters from non-craters among the stage-1 candidates (QH3) of a representation,
from a kernel matrix between samples: the quantum kernels of ``qc_kernels.py`` (QI2-QI3, exact or with shot noise)
or the classical baselines of QJ1 on the same reduced features.

QI4.1 - Classifier.  ``fit_svc``: ``sklearn.svm.SVC(kernel="precomputed", class_weight="balanced")`` on K_train
(n, n); ``svc_scores``: on K_test (m, n), the kernel between the test candidates and the training samples (rows =
test, columns = training samples in the training order).  Score of a candidate (decided with the user): the Platt
probability P(crater), used for AP and ranking.  It is computed with
``CalibratedClassifierCV(SVC(...), method="sigmoid", ensemble=False, cv=5)``, the replacement that scikit-learn 1.9
indicates for the deprecated ``SVC(probability=True)``: the SVC is fitted on the whole sample, the sigmoid on its
out-of-fold margins of a deterministic stratified 5-fold split (the precomputed kernel is sliced on rows and columns).
The sigmoid is monotone in the margin, so the ranking (and AP) equals that of ``decision_function``; the margin and
the SVC prediction ``predict`` (margin > 0) are kept too, and can disagree with proba >= 0.5 on a few candidates.
``class_weight="balanced"``
weights every class by n / (2 n_class): it has no effect on the balanced QK samples (QH4) but matters in the
cross-validation folds of QI4.2.  ``stage2_kernels``: the (K_train, K_test) of one feature map, d, c, r from the
standardised components of QI1.1, exact (QI3.1) or with N shots and the PSD projection (QI3.3).

``run_svm_check``: the end-to-end smoke run of QI4.1 on the F-X sample of the main grid size (first seed) and its
2,000 test candidates, fixed (not tuned) C, c, r, s of ``stage2.svm_check``: per map (QK-A, QK-Z, QK-P), d and
shots (exact, N): test AP of the Platt score and of the margin, F1 of ``predict`` and of proba >= 0.5, share of
candidates predicted positive, training accuracy, support vectors, agreement of the two predictions, time.
Indicative only: the tuned results are QI4.2-QI4.3.

QI4.2 - Hyper-parameters.  ``cv_search``: 5-fold cross-validation **inside** the QK training sample, folds from
``spatial_folds`` (``StratifiedGroupKFold``, groups = the 5 deg blocks, so a block is never split between training
and validation; classes balanced across folds; shuffled with the sample seed; decided with the user).  In every
fold the PCA of QI1.1 is refitted on the training fold only, then every combination of c (``quantum.bandwidths``),
r (``quantum.zz_reps``, QK-Z / QK-P), s (``quantum.projected_gamma_scales``, QK-P) and C (``stage2.svm_C``) is
scored on the validation fold with an SVC on the exact kernels (``grid_kernels``; the QK-Z states are simulated
once and give both the QK-Z kernel and the QK-P expectations, ``zz_kernel_and_expectations``).  Criterion (decided
with the user): mean fold AP of the SVM margin (Platt is monotone, it does not change the AP), ties -> the simpler
model (``select_best``); the mean fold F1 is kept for QI5.2.  ``run_cv_check``: the search on the F-X sample of the
main grid size (first seed) for every map and d, and the chosen model refitted on the whole sample and scored on
the test candidates.

QI4.3 - Main grid and learning curve.  A *unit* = one QK sample (n, seed) at d qubits: the CV of QI4.2 for all the
maps together (``run_unit``), then the chosen configuration of every map refitted on the whole sample and scored on
the 2,000 test candidates (``refit_and_score``: Platt AP, F1 of ``predict``).  ``run_grid``: the maps of
``stage2.main`` x every d x every seed of ``qk_samples.seeds`` at the grid size ``stage2.main.n``; at the d of
``stage2.main.per_c_dims`` also the best configuration at every c, refitted and scored (Fig. 3b, ``role = per_c``).
``run_curve``: every size of ``qk_samples.sizes`` x seed at the best d of each map (``best_dims``: argmax of the
mean CV AP over the seeds of the grid, ties -> smaller d; decided with the user).  Exact kernels.  Outputs (decided
with the user), written after every unit so that a stopped job resumes where it was: ``results/qi4_runs.csv`` (one
row per run: chosen hyper-parameters, CV and test metrics; the input of the figures), ``results/qi4_cv.csv`` (one
row per run x configuration: fold-mean AP and F1, for QI5.2), ``<output>/stage2/qi4_test_scores.npz`` (per run the
Platt probability and the margin of the 2,000 test candidates, with their index, class, band, block and tercile; for
QI4.4 and QJ3); ``qi4_summary`` writes the report section QI4.3.

QI4.4 - Metrics.  ``run_metrics``: for the chosen runs of QI4.3 at the best d (every n and seed), on the 2,000 test
candidates: AP overall and per band, and at the default threshold (Platt probability >= ``stage2.default_threshold``
= 0.5, decided with the user: the same rule for every method of QJ) precision, recall, F1 overall and per band
(``threshold_metrics``; per band = the positives of the band + all the negatives).  End to end on all the labelled
Test sub-tiles (decided with the user: estimated from the proportional subset of QH5, no stage-2 run on the ~110k
candidates): recall = stage-1 recall at the QH2 threshold x stage-2 recall, precision = stage-2 precision.  Precision
gain over stage 1 alone: at its QH2 threshold, and at the same end-to-end recall (``precision_at_recall``).

QI6.1 - Every representation.  ``run_representations``: on every other variant (``stage2.representations``), with
its own stage 1, QK samples and test candidates, every map at its best d of QI4.3, at the grid size and every seed:
the CV of QI4.2 on its own sample and the refit of the chosen configuration (``run_unit`` with ``variant``, decided with
the user), saved in ``results/qi6_runs.csv``, ``qi6_cv.csv``, ``<output>/stage2/qi6_<variant>_test_scores.npz``
(``stage2_paths``); ``qi6_summary``: the QI4.4 metrics (``stage2_metrics``) per representation and map, with F-X from
QI4.3 (Table I).

QJ1 - Classical baselines.  Same inputs (the standardised PCA components of the same QK samples), same protocol
(``spatial_folds``, PCA refitted per fold, mean fold AP, refit on the whole sample, the 2,000 test candidates) and a
comparable search budget (decided with the user: 20-40 configurations per method, ``classical_grid``): RBF SVM (gamma
= s x the median gamma, the 7 values of ``classical.rbf_gamma_scales``, the same relative grid as QK-P), linear SVM,
random Fourier features + logistic regression, MLP with one hidden layer, gradient-boosted trees; RBF-wide = the RBF
with the gamma grid extended to 1/64 x the median, a sensitivity check of the lower edge (decided with the user, not
part of the main comparison).  ``classical_scores``: the SVMs on precomputed kernels with the Platt score of QI4.1,
the others with their probability.  ``run_baselines`` on the units of ``qj1_units`` (decided with the user: the mirror
of QI4.3 grid, QI6.1 and the QI4.3 learning curve), saved in ``results/qj1_runs.csv``, ``qj1_cv.csv``,
``<output>/stage2/qj1_<variant>_test_scores.npz``; ``qj1_summary`` writes the report section QJ1.

QJ2 - Reference lines.  Stage 1 alone (AP of its logit on the same candidates; its precision at the recall of every
stage-2 run is already in QI4.4 / QI6.1 / QJ1), and without compression (decided with the user: both): the linear
probe of QG1 (full vectors, all S1-train) on the 2,000 candidates (``qg1_on_candidates``), and ``LIN-full``, the
linear SVM of QJ1 on the full standardised vectors of the same QK samples (``standardize``, scaled by
sqrt(``classical.full_reference_dim`` / D) so that the C grid of LIN keeps its meaning; same CV / refit / Platt;
``run_references`` on ``qj2_units``), so that LIN-full - LIN is the cost of the PCA at equal data; ``qj2_summary``.

QJ3 - Statistics.  ``qj3_statistics`` (decided with the user): per unit and method the mean / std over the seeds and
the block-bootstrap 95 % CI of the seed-averaged test AP; every quantum map against the RBF and the linear SVM
(``QJ3_REFERENCES``) on the same samples and candidates: paired AP difference with CI, exact McNemar test on the errors
per seed (``mcnemar_p``), majority over the seeds (median p), the plan rule and the Holm-adjusted version (``holm``).

QJ4 - Costs.  ``run_costs`` (decided with the user): wall-clock time of the refit + scoring of the chosen
configuration of every method on the same node (seed of the first sample, F-X units of ``qj4_units``), the QSVC with
the engines of ``QJ4_ENGINES`` (CPU, the GPU pipeline of QI4, all GPU); ``hardware_counts``: circuits and shots on a QPU
(depth, gates and QPU time: QK3).

Commands
--------
svm-check    QI4.1 -> report section QI4.1 of ``results/qi_kernels.json``.  GPU job (run_qcai_gpu.sh,
             ``--device cuda``; Aer states on the CPUs), ~5 min.

cv-check     QI4.2 -> report section QI4.2 of ``results/qi_kernels.json``: folds, chosen hyper-parameters,
             fold AP / F1, best fold AP per c, test AP of the refitted choice, per map and d.  GPU job
             (run_qcai_gpu.sh), ~30 min.

grid         QI4.3 main grid -> ``results/qi4_runs.csv``, ``qi4_cv.csv``, ``<output>/stage2/qi4_test_scores.npz``,
             report section QI4.3.  GPU job, ~2 h; resumes after an interruption.

curve        QI4.3 learning curve at the best d of every map (needs the complete grid) -> same files.  GPU
             job, 1-5 h depending on the best d.

summary      rewrites the report section QI4.3 from ``qi4_runs.csv``.  Seconds.

representations   QI6.1 runs + report section QI6.1.  GPU job, ~45 min; resumes after an interruption.

representations-summary   rewrites the report section QI6.1 from the saved runs.  Seconds.

baselines    QJ1 classical baselines (RBF, LIN, RFF, MLP, GBT, RBF-wide) on the units of ``qj1_units`` ->
             ``results/qj1_runs.csv``, ``qj1_cv.csv``, ``<output>/stage2/qj1_<variant>_test_scores.npz``, report
             section QJ1 of ``results/qj_classical.json``.  CPU job (run_qcai_cpu.sh); resumes after an interruption.

baselines-summary   rewrites the report section QJ1 from the saved runs.  Seconds.

references   QJ2 LIN-full runs -> ``results/qj2_runs.csv``, ``qj2_cv.csv``, ``<output>/stage2/qj2_<variant>_test_scores.npz``
             + report section QJ2 of ``results/qj_classical.json`` (reference lines).  CPU, minutes.

references-summary   rewrites the report section QJ2.  Seconds.

statistics   QJ3 -> report section QJ3 of ``results/qj_classical.json`` from the saved test scores.  CPU, ~1 min.

costs        QJ4 -> report section QJ4 of ``results/qj_classical.json``.  GPU job (run_qcai_gpu.sh, CPU and GPU
             timings on the same node), ~40 min.

metrics      QI4.4 -> report section QI4.4 of ``results/qi_kernels.json`` from the QI4.3 test scores and the stage-1
             scores of QH3.  Seconds.

Usage
-----
    python Experiments/qcai2027/qc_experiments.py svm-check --device cuda    (GPU job: run_qcai_gpu.sh)
    python Experiments/qcai2027/qc_experiments.py cv-check --device cuda     (GPU job: run_qcai_gpu.sh)
    python Experiments/qcai2027/qc_experiments.py grid --device cuda         (GPU job: run_qcai_gpu.sh)
    python Experiments/qcai2027/qc_experiments.py curve --device cuda        (GPU job: run_qcai_gpu.sh)
    python Experiments/qcai2027/qc_experiments.py summary
    python Experiments/qcai2027/qc_experiments.py metrics
    python Experiments/qcai2027/qc_experiments.py representations --device cuda   (GPU job: run_qcai_gpu.sh)
    python Experiments/qcai2027/qc_experiments.py baselines        (CPU job: run_qcai_cpu.sh)
    python Experiments/qcai2027/qc_experiments.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_data import CONFIG_PATH, load_config, load_report, save_report  # noqa: E402
from qc_kernels import (expectations_torch, feature_angles, features_path, fidelity_kernels, kernel_projected,  # noqa: E402
                        kernel_qk_a, load_qk, median_gamma, noisy_kernels, projected_features, reduce_features, state_chunks)
from qc_probes import BANDS, average_precision, band_masks  # noqa: E402


# ---------------------------------------------------------------------------
# QI4.1 - SVM on a precomputed kernel
# ---------------------------------------------------------------------------


def fit_svc(k_tr: np.ndarray, y_tr: np.ndarray, C: float = 1.0):
    """QI4.1 ``SVC(kernel="precomputed", class_weight="balanced")`` fitted on the training kernel ``k_tr`` (n, n)
    and labels ``y_tr`` (0 / 1), wrapped in the Platt calibration (sigmoid on the out-of-fold margins of a
    stratified 5-fold split, ``ensemble=False``: one SVC fitted on the whole sample).  Deterministic."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import SVC

    svc = SVC(kernel="precomputed", C=C, class_weight="balanced")
    return CalibratedClassifierCV(svc, method="sigmoid", ensemble=False, cv=5).fit(k_tr, y_tr)


def fitted_svc(model):
    """The SVC of a ``fit_svc`` model (fitted on the whole training sample)."""
    return model.calibrated_classifiers_[0].estimator


def svc_scores(model, k_te: np.ndarray) -> dict:
    """Scores of the test kernel ``k_te`` (m, n_train): {proba = Platt P(crater) (the score of QI4), margin =
    ``decision_function`` of the SVC, pred = its ``predict`` (margin > 0)}."""
    pos, svc = list(model.classes_).index(1), fitted_svc(model)
    return {"proba": model.predict_proba(k_te)[:, pos], "margin": svc.decision_function(k_te), "pred": svc.predict(k_te)}


def stage2_kernels(name: str, a_tr: np.ndarray, a_te: np.ndarray, c: float, reps: int = 1, gamma_scale: float = 1.0,
                   shots: int | None = None, rng=None, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """(K_train, K_test) of the feature map ``name`` (QK-A, QK-Z, QK-Z-qiskit, QK-P) from the standardised components
    ``a_tr``, ``a_te`` of QI1.1 at bandwidth ``c``: exact (QI3.1) if ``shots`` is None, else with ``shots``
    measurements per value and K_train projected on the PSD cone (QI3.3, ``noisy_kernels``).  QK-P: gamma =
    ``gamma_scale`` x the median gamma of the (noisy) training expectations."""
    th_tr, th_te = feature_angles(a_tr, c), feature_angles(a_te, c)
    if name == "QK-P":
        exact = (projected_features(th_tr, reps, device), projected_features(th_te, reps, device))
    else:
        exact = fidelity_kernels(name, th_tr, th_te, reps, device)
    if shots is not None:
        return noisy_kernels(name, exact, shots, rng, gamma_scale)[:2]
    if name == "QK-P":
        g = gamma_scale * median_gamma(exact[0])
        return kernel_projected(exact[0], exact[0], g), kernel_projected(exact[1], exact[0], g)
    return exact


def f1(pred: np.ndarray, y: np.ndarray) -> float:
    tp = int(np.sum((pred == 1) & (y == 1)))
    return 2 * tp / max(int(np.sum(pred == 1)) + int(np.sum(y == 1)), 1)


def run_svm_check(cfg: dict, device: str = "cuda") -> dict:
    """QI4.1 report: smoke run of ``fit_svc`` / ``svc_scores`` on the real kernels (see the module docstring)."""
    q, chk = cfg["quantum"], cfg["stage2"]["svm_check"]
    n, seed = int(q["bandwidth_check_size"]), int(cfg["qk_samples"]["seeds"][0])
    C, c, r, s, N = float(chk["C"]), float(chk["c"]), int(chk["reps"]), float(chk["gamma_scale"]), int(chk["shots"])
    x_tr, i_tr, x_te, i_te = load_qk(cfg, "F-X", n, seed)
    y_tr, y_te = i_tr["cls"].astype(int), i_te["cls"].astype(int)
    out, checks = {}, {"test_kernel_shape": True, "proba_in_0_1": True, "ap_proba_equal_ap_margin": True}
    for d in [int(v) for v in q["dims"]]:
        a_tr, a_te = reduce_features(x_tr, d, x_te)
        for m_id, name in enumerate(("QK-A", "QK-Z", "QK-P")):
            for shots in (None, N):
                t0 = time.time()
                rng = np.random.default_rng([int(cfg["seed"]), d, m_id, shots or 0])
                k_tr, k_te = stage2_kernels(name, a_tr, a_te, c, r, s, shots, rng, device)
                t_k = time.time() - t0
                model = fit_svc(k_tr, y_tr, C)
                sc = svc_scores(model, k_te)
                checks["test_kernel_shape"] &= k_te.shape == (len(y_te), n)
                checks["proba_in_0_1"] &= bool(np.all((sc["proba"] >= 0) & (sc["proba"] <= 1)))
                ap_p, ap_m = average_precision(sc["proba"], y_te), average_precision(sc["margin"], y_te)
                checks["ap_proba_equal_ap_margin"] &= abs(ap_p - ap_m) < 1e-9
                key = f"{name}_d{d}" + ("" if name == "QK-A" else f"_r{r}") + ("_exact" if shots is None else f"_N{shots}")
                out[key] = {"test_ap_proba": round(ap_p, 4), "test_ap_margin": round(ap_m, 4),
                            "test_f1_predict": round(f1(sc["pred"], y_te), 4),
                            "test_f1_proba05": round(f1((sc["proba"] >= 0.5).astype(int), y_te), 4),
                            "share_predicted_positive": round(float(sc["pred"].mean()), 4),
                            "predict_vs_proba05_agreement": round(float(np.mean(sc["pred"] == (sc["proba"] >= 0.5))), 4),
                            "train_accuracy": round(float(np.mean(fitted_svc(model).predict(k_tr) == y_tr)), 4),
                            "support_vectors": int(fitted_svc(model).n_support_.sum()),
                            "kernel_s": round(t_k, 1), "svm_s": round(time.time() - t0 - t_k, 2)}
                print(f"QI4.1 {key}: AP proba {ap_p:.3f} margin {ap_m:.3f}, F1 {out[key]['test_f1_predict']:.3f}, "
                      f"train acc {out[key]['train_accuracy']:.3f}, SV {out[key]['support_vectors']}, "
                      f"kernel {t_k:.0f} s", flush=True)
    report = {"configs": out, "checks": checks, "variant": "F-X", "sample": f"n{n}_s{seed} (QH4) + 2,000 test candidates (QH5)",
              "test_positives": int(y_te.sum()), "fixed": {"C": C, "c": c, "reps": r, "gamma_scale_QK-P": s, "shots": N},
              "note": "smoke run of the QI4.1 classifier with fixed, not tuned, hyper-parameters (tuned results: QI4.2-QI4.3); "
                      "SVC(kernel='precomputed', class_weight='balanced') in CalibratedClassifierCV(method='sigmoid', ensemble=False, cv=5); "
                      "proba = Platt P(crater) (score of QI4), margin = decision_function, predict = margin > 0; "
                      "noisy kernels as QI3.3 (K_train PSD-projected, diagonal 1); AP over the 2,000 test candidates"}
    save_report(cfg, "QI4.1", report)
    return report


# ---------------------------------------------------------------------------
# QI4.2 - Hyper-parameters by spatially grouped cross-validation
# ---------------------------------------------------------------------------


def spatial_folds(y: np.ndarray, blocks: np.ndarray, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """QI4.2 folds of a QK training sample: ``StratifiedGroupKFold`` (groups = 5 deg blocks, so no block is split
    between training and validation; classes balanced across folds), shuffled with ``seed``.  [(train, val)]."""
    from sklearn.model_selection import StratifiedGroupKFold

    return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, blocks))


def zz_kernel_and_expectations(th_tr: np.ndarray, th_va: np.ndarray, reps: int, device: str = "cuda",
                               chunk: int = 256) -> tuple[tuple, tuple]:
    """QK-Z (product pair) states of the training and validation angles computed once (Aer, ``state_chunks``) and
    used twice: ((K_tr, K_va) fidelity kernel of QK-Z, (E_tr, E_va) expectations of QK-P).  Equal to
    ``fidelity_kernels("QK-Z", ...)`` and ``projected_features`` (selftest), at half the simulation cost."""
    import torch

    p_tr = torch.cat(list(state_chunks("QK-Z", th_tr, reps, device, chunk)))
    gram = lambda a: ((a @ p_tr.conj().T).abs() ** 2).double().cpu().numpy()
    k_tr = np.concatenate([gram(p_tr[s:s + chunk]) for s in range(0, len(p_tr), chunk)])
    e_tr = np.concatenate([expectations_torch(p_tr[s:s + chunk]).cpu().numpy() for s in range(0, len(p_tr), chunk)])
    k_va, e_va = [], []
    for a in state_chunks("QK-Z", th_va, reps, device, chunk):
        k_va.append(gram(a))
        e_va.append(expectations_torch(a).cpu().numpy())
    del p_tr
    return (k_tr, np.concatenate(k_va)), (e_tr, np.concatenate(e_va))


def grid_kernels(a_tr: np.ndarray, a_va: np.ndarray, c: float, maps, reps, scales, device: str = "cuda"):
    """Exact (K_tr, K_va) of every map of ``maps`` at bandwidth ``c``: yields (map, r, s, K_tr, K_va), r = None for
    QK-A, s = None except for QK-P (gamma = s x the median gamma of the training expectations)."""
    th_tr, th_va = feature_angles(a_tr, c), feature_angles(a_va, c)
    if "QK-A" in maps:
        yield "QK-A", None, None, kernel_qk_a(th_tr, th_tr), kernel_qk_a(th_va, th_tr)
    if "QK-Z" not in maps and "QK-P" not in maps:
        return
    for r in reps:
        kz, ez = zz_kernel_and_expectations(th_tr, th_va, r, device)
        if "QK-Z" in maps:
            yield "QK-Z", r, None, kz[0], kz[1]
        if "QK-P" in maps:
            g0 = median_gamma(ez[0])
            for s in scales:
                yield "QK-P", r, s, kernel_projected(ez[0], ez[0], s * g0), kernel_projected(ez[1], ez[0], s * g0)


def cv_search(x: np.ndarray, y: np.ndarray, blocks: np.ndarray, d: int, cfg: dict, maps=("QK-A", "QK-Z", "QK-P"),
              seed: int = 0, device: str = "cuda") -> tuple[list[dict], list]:
    """QI4.2 grid search on one QK training sample (raw vectors ``x``, labels, blocks) at d qubits: for every fold of
    ``spatial_folds`` the PCA of QI1.1 is refitted on the training fold only, then for every c (QI1.2), r, s (QK-P)
    and C (``stage2.svm_C``) an ``SVC(kernel="precomputed", class_weight="balanced")`` is trained on the training
    fold and scored on the validation fold (AP of the margin, F1 of ``predict``).  Platt is not needed here: it is
    monotone and does not change the AP.  Returns (one row per configuration and fold, the folds)."""
    from sklearn.svm import SVC

    q = cfg["quantum"]
    cs, reps = [float(v) for v in q["bandwidths"]], [int(v) for v in q["zz_reps"]]
    scales, Cs = [float(v) for v in q["projected_gamma_scales"]], [float(v) for v in cfg["stage2"]["svm_C"]]
    folds = spatial_folds(y, blocks, int(cfg["stage2"]["cv_folds"]), seed)
    rows = []
    for f, (tr, va) in enumerate(folds):
        a_tr, a_va = reduce_features(x[tr], d, x[va])
        for c in cs:
            for m, r, s, k_tr, k_va in grid_kernels(a_tr, a_va, c, maps, reps, scales, device):
                for C in Cs:
                    svc = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(k_tr, y[tr])
                    rows.append({"map": m, "r": r, "s": s, "c": c, "C": C, "fold": f,
                                 "ap": average_precision(svc.decision_function(k_va), y[va]), "f1": f1(svc.predict(k_va), y[va])})
    return rows, folds


def cv_table(rows: list[dict]) -> list[dict]:
    """Mean / std over the folds of every configuration (map, r, s, c, C)."""
    groups: dict = {}
    for row in rows:
        groups.setdefault((row["map"], row["r"], row["s"], row["c"], row["C"]), []).append(row)
    out = []
    for (m, r, s, c, C), g in groups.items():
        ap, f = np.array([v["ap"] for v in g]), np.array([v["f1"] for v in g])
        out.append({"map": m, "r": r, "s": s, "c": c, "C": C, "cv_ap": float(ap.mean()), "cv_ap_std": float(ap.std(ddof=1)),
                    "cv_f1": float(f.mean()), "folds": len(g)})
    return out


def select_best(table: list[dict], name: str) -> dict:
    """QI4.2 choice for the map ``name``: highest mean fold AP; ties (1e-9) -> the simpler model (smaller C, r, s, c)."""
    cand = [t for t in table if t["map"] == name]
    return min(cand, key=lambda t: (-round(t["cv_ap"], 9), t["C"], t["r"] or 0, t["s"] or 0, t["c"]))


def fold_summary(folds: list, y: np.ndarray, blocks: np.ndarray) -> dict:
    """Checks of the folds: blocks never split, every sample validated once; sizes and positive share."""
    val = np.concatenate([va for _, va in folds])
    return {"blocks_disjoint": all(not set(blocks[tr]) & set(blocks[va]) for tr, va in folds),
            "each_sample_validated_once": bool(np.array_equal(np.sort(val), np.arange(len(y)))),
            "val_sizes": [int(len(va)) for _, va in folds],
            "val_positive_share": [round(float(y[va].mean()), 3) for _, va in folds],
            "val_blocks": [int(len(set(blocks[va]))) for _, va in folds]}


def run_cv_check(cfg: dict, device: str = "cuda") -> dict:
    """QI4.2 report: the cross-validation of ``cv_search`` on the F-X sample of the main grid size (first seed), every
    map and d: folds, chosen hyper-parameters, mean / std fold AP and F1 of the choice, best fold AP for every c
    (the other hyper-parameters chosen), configurations tried; then the chosen model refitted on the whole sample
    (``stage2_kernels``, ``fit_svc``) and its AP on the 2,000 test candidates (indicative: the first seed of QI4.3)."""
    q = cfg["quantum"]
    n, seed = int(q["bandwidth_check_size"]), int(cfg["qk_samples"]["seeds"][0])
    x_tr, i_tr, x_te, i_te = load_qk(cfg, "F-X", n, seed)
    y_tr, y_te = i_tr["cls"].astype(int), i_te["cls"].astype(int)
    out, folds_info = {}, None
    for d in [int(v) for v in q["dims"]]:
        t0 = time.time()
        rows, folds = cv_search(x_tr, y_tr, i_tr["block"], d, cfg, seed=seed, device=device)
        folds_info = folds_info or fold_summary(folds, y_tr, i_tr["block"])
        table, t_cv = cv_table(rows), time.time() - t0
        a_tr, a_te = reduce_features(x_tr, d, x_te)
        for name in ("QK-A", "QK-Z", "QK-P"):
            b = select_best(table, name)
            k_tr, k_te = stage2_kernels(name, a_tr, a_te, b["c"], b["r"] or 1, b["s"] or 1.0, device=device)
            ap_te = average_precision(svc_scores(fit_svc(k_tr, y_tr, b["C"]), k_te)["proba"], y_te)
            per_c = {f"c{c:g}": round(max(t["cv_ap"] for t in table if t["map"] == name and t["c"] == c), 4)
                     for c in sorted({t["c"] for t in table})}
            key = f"{name}_d{d}"
            out[key] = {"chosen": {k: b[k] for k in ("c", "r", "s", "C")}, "cv_ap": round(b["cv_ap"], 4),
                        "cv_ap_std": round(b["cv_ap_std"], 4), "cv_f1": round(b["cv_f1"], 4), "best_cv_ap_per_c": per_c,
                        "configs_tried": sum(t["map"] == name for t in table), "test_ap_refit": round(ap_te, 4)}
            print(f"QI4.2 {key}: chosen {out[key]['chosen']}, CV AP {b['cv_ap']:.3f} +- {b['cv_ap_std']:.3f}, "
                  f"test AP {ap_te:.3f}", flush=True)
        out[f"time_cv_d{d}_s"] = round(t_cv, 1)
        print(f"QI4.2 d = {d}: CV {t_cv:.0f} s", flush=True)
    report = {"configs": out, "folds": folds_info, "variant": "F-X", "sample": f"n{n}_s{seed} (QH4) + 2,000 test candidates (QH5)",
              "grid": {"C": cfg["stage2"]["svm_C"], "c": q["bandwidths"], "r": q["zz_reps"], "s_QK-P": q["projected_gamma_scales"]},
              "note": "StratifiedGroupKFold (groups = 5 deg blocks, shuffled with the sample seed); PCA of QI1.1 refitted on "
                      "every training fold; exact kernels; SVC(kernel='precomputed', class_weight='balanced') per fold; "
                      "criterion = mean fold AP of the margin, ties -> smaller C, r, s, c; cv_f1 = mean fold F1 of predict; "
                      "best_cv_ap_per_c = best mean fold AP at each c over the other hyper-parameters; test_ap_refit = "
                      "Platt score of the chosen model refitted on the whole sample (indicative, first seed of QI4.3)"}
    save_report(cfg, "QI4.2", report)
    return report


# ---------------------------------------------------------------------------
# QI4.3 - Main grid and learning curve
# ---------------------------------------------------------------------------

RUN_COLS = ["step", "script", "run_id", "role", "variant", "map", "d", "n", "seed", "c", "r", "s", "C", "cv_ap", "cv_ap_std",
            "cv_f1", "test_ap", "test_f1", "support_vectors", "seconds", "timestamp"]


def stage2_dir(cfg: dict) -> Path:
    """Test scores of the stage-2 runs: ``<output>/stage2/``."""
    return cfg["paths"]["output"] / "stage2"


def stage2_paths(cfg: dict, tag: str = "qi4", variant: str | None = None) -> dict:
    """Files of a family of stage-2 runs: ``results/<tag>_runs.csv``, ``results/<tag>_cv.csv`` and the test scores
    ``<output>/stage2/<tag>_test_scores.npz`` (one npz per variant, ``<tag>_<variant>_test_scores.npz``, when the
    family covers several representations, each with its own test candidates: QI6.1)."""
    res = cfg["paths"]["results"]
    npz = f"{tag}_test_scores.npz" if variant is None else f"{tag}_{variant}_test_scores.npz"
    return {"runs": res / f"{tag}_runs.csv", "cv": res / f"{tag}_cv.csv", "scores": stage2_dir(cfg) / npz}


def qi4_paths(cfg: dict) -> dict:
    """Files of QI4.3 (F-X): ``qi4_runs.csv``, ``qi4_cv.csv``, ``stage2/qi4_test_scores.npz``."""
    return stage2_paths(cfg, "qi4")


def run_id(name: str, d: int, n: int, seed: int, c: float | None = None, variant: str | None = None) -> str:
    """Key of a run in the runs CSV and the scores npz; ``c`` only for the per-c runs of Fig. 3b; ``variant`` prefixed
    for the runs of several representations (QI6.1), not for the F-X runs of QI4.3."""
    return ("" if variant is None else f"{variant}_") + f"{name}_d{d}_n{n}_s{seed}" + ("" if c is None else f"_c{c:g}")


def done_runs(cfg: dict, paths: dict | None = None) -> set:
    import pandas as pd

    p = (paths or qi4_paths(cfg))["runs"]
    return set(pd.read_csv(p)["run_id"]) if p.exists() else set()


def save_unit(cfg: dict, unit: str, cv_rows: list[dict], runs: list[dict], scores: dict, paths: dict | None = None,
              cols: list[str] | None = None) -> None:
    """Write one unit (one variant, d, n, seed) of QI4.3 / QI6.1 / QJ1 to ``paths`` (default the QI4.3 files): its CV
    table (rows of the unit and of the same methods replaced, so one method can be redone alone), its test scores (npz rewritten through a temporary file) and, last, its runs
    with the columns ``cols`` (default ``RUN_COLS``; the completion marker used to resume)."""
    import os

    import pandas as pd

    from qc_data import written

    p = paths or qi4_paths(cfg)
    p["scores"].parent.mkdir(parents=True, exist_ok=True)
    cv = pd.DataFrame(cv_rows).assign(unit=unit)
    if p["cv"].exists():
        old = pd.read_csv(p["cv"])
        keep = (old["unit"] != unit) | ~old["map"].isin(set(cv["map"]))      # rows of the same unit and method replaced
        cv = pd.concat([old[keep], cv], ignore_index=True)
    cv.to_csv(p["cv"], index=False, float_format="%.5g")
    old = dict(np.load(p["scores"])) if p["scores"].exists() else {}
    tmp = p["scores"].with_name("tmp_" + p["scores"].name)
    np.savez(tmp, **{**old, **scores})
    os.replace(tmp, p["scores"])
    runs_df = pd.DataFrame(runs)[cols or RUN_COLS]
    if p["runs"].exists():
        old = pd.read_csv(p["runs"])
        runs_df = pd.concat([old[~old["run_id"].isin(runs_df["run_id"])], runs_df], ignore_index=True)
    runs_df.to_csv(p["runs"], index=False, float_format="%.5g")
    for k in p.values():
        written(k)


def refit_and_score(cfg: dict, a_tr: np.ndarray, y_tr: np.ndarray, a_te: np.ndarray, y_te: np.ndarray, picks: list[dict],
                    device: str) -> list[tuple[dict, dict, dict]]:
    """Every configuration of ``picks`` ({map, c, r, s, C, ...}) refitted on the whole training sample (``fit_svc``)
    and scored on the test candidates: [(pick, {proba, margin}, {test_ap, test_f1, support_vectors})].  The kernels
    of the picks with the same c come from one ``grid_kernels`` call (QK-Z states simulated once for QK-Z and QK-P)."""
    out = []
    for c in sorted({p["c"] for p in picks}):
        group = [p for p in picks if p["c"] == c]
        maps = {p["map"] for p in group}
        reps = sorted({p["r"] for p in group if p["r"] is not None and not np.isnan(p["r"])}) or [1]
        reps = [int(r) for r in reps]
        scales = sorted({p["s"] for p in group if p["s"] is not None and not np.isnan(p["s"])}) or [1.0]
        for m, r, s, k_tr, k_te in grid_kernels(a_tr, a_te, c, maps, reps, scales, device):
            for p in group:
                same = p["map"] == m and (m == "QK-A" or int(p["r"]) == r) and (m != "QK-P" or float(p["s"]) == s)
                if not same:
                    continue
                model = fit_svc(k_tr, y_tr, float(p["C"]))
                sc = svc_scores(model, k_te)
                met = {"test_ap": average_precision(sc["proba"], y_te), "test_f1": f1(sc["pred"], y_te),
                       "support_vectors": int(fitted_svc(model).n_support_.sum())}
                out.append((p, {"proba": sc["proba"].astype(np.float32), "margin": sc["margin"].astype(np.float32)}, met))
    return out


def run_unit(cfg: dict, x_tr, i_tr, x_te, i_te, d: int, n: int, seed: int, maps, per_c: bool, device: str,
             te_index: np.ndarray | None = None, variant: str | None = None, step: str = "QI4.3",
             paths: dict | None = None) -> None:
    """One unit of QI4.3 (or QI6.1 with ``variant``, ``step`` and ``paths``): the CV of ``cv_search`` (all ``maps``
    together) on the sample (n, seed) at d, the chosen configuration of every map refitted and scored on the test
    candidates, and with ``per_c`` also the best configuration at every c (Fig. 3b); saved by ``save_unit``.  With
    ``variant`` the run ids and the unit name carry the variant."""
    import datetime

    vid = variant or cfg["stage2"]["main"]["variant"]
    y_tr, y_te = i_tr["cls"].astype(int), i_te["cls"].astype(int)
    t0 = time.time()
    rows, _ = cv_search(x_tr, y_tr, i_tr["block"], d, cfg, maps=tuple(maps), seed=seed, device=device)
    table = cv_table(rows)
    picks = []
    for name in maps:
        picks.append({**select_best(table, name), "role": "chosen", "run_id": run_id(name, d, n, seed, None, variant)})
        if per_c:
            for c in sorted({t["c"] for t in table}):
                b = select_best([t for t in table if t["c"] == c], name)
                picks.append({**b, "role": "per_c", "run_id": run_id(name, d, n, seed, c, variant)})
    a_tr, a_te = reduce_features(x_tr, d, x_te)
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    runs = []
    scores = {f"test_{k}": np.asarray(v) for k, v in i_te.items()}          # identity of the test candidates
    if te_index is not None:
        scores["test_index"] = np.asarray(te_index)
    for p, sc, met in refit_and_score(cfg, a_tr, y_tr, a_te, y_te, picks, device):
        runs.append({"step": step, "script": "qc_experiments.py", "run_id": p["run_id"], "role": p["role"], "variant": vid,
                     "map": p["map"], "d": d, "n": n, "seed": seed, "c": p["c"], "r": p["r"], "s": p["s"], "C": p["C"],
                     "cv_ap": p["cv_ap"], "cv_ap_std": p["cv_ap_std"], "cv_f1": p["cv_f1"], **met,
                     "seconds": round(time.time() - t0, 1), "timestamp": stamp})
        scores.update({f"{p['run_id']}__proba": sc["proba"], f"{p['run_id']}__margin": sc["margin"]})
    cv_rows = [{"variant": vid, "d": d, "n": n, "seed": seed, **t} for t in table]
    unit = ("" if variant is None else f"{variant}_") + f"d{d}_n{n}_s{seed}"
    save_unit(cfg, unit, cv_rows, runs, scores, paths)
    for r_ in runs:
        if r_["role"] == "chosen":
            print(f"{step} {r_['run_id']}: c {r_['c']} r {r_['r']} s {r_['s']} C {r_['C']}, CV AP {r_['cv_ap']:.3f}, "
                  f"test AP {r_['test_ap']:.3f}", flush=True)
    print(f"{step} unit {unit}: {time.time() - t0:.0f} s", flush=True)


def best_dims(cfg: dict) -> dict:
    """Best d of every map: argmax over d of the mean CV AP over the seeds of the main grid (``role == chosen``,
    n = the grid size), ties (1e-9) -> the smaller d."""
    import pandas as pd

    main = cfg["stage2"]["main"]
    df = pd.read_csv(qi4_paths(cfg)["runs"])
    g = df[(df["role"] == "chosen") & (df["n"] == int(main["n"]))].groupby(["map", "d"])["cv_ap"].mean().reset_index()
    return {m: int(min(sub.itertuples(), key=lambda t: (-round(t.cv_ap, 9), t.d)).d) for m, sub in g.groupby("map")}


def qi4_summary(cfg: dict) -> dict:
    """QI4.3 report from ``qi4_runs.csv``: main grid (mean / std over the seeds of test and CV AP, chosen c / r / C per
    map and d), best d per map, test AP per c (Fig. 3b), learning curve (mean / std of the test AP per map and n)."""
    import pandas as pd

    main = cfg["stage2"]["main"]
    df = pd.read_csv(qi4_paths(cfg)["runs"])
    ch, n0 = df[df["role"] == "chosen"], int(main["n"])
    ms = lambda v: [round(float(v.mean()), 4), round(float(v.std(ddof=1)), 4) if len(v) > 1 else None, int(len(v))]
    grid = {f"{m}_d{d}": {"test_ap_mean_std_runs": ms(g["test_ap"]), "cv_ap_mean_std_runs": ms(g["cv_ap"]),
                          "test_f1_mean": round(float(g["test_f1"].mean()), 4),
                          "chosen_c": g["c"].value_counts().to_dict(), "chosen_C": g["C"].value_counts().to_dict()}
            for (m, d), g in ch[ch["n"] == n0].groupby(["map", "d"])}
    pc = df[df["role"] == "per_c"]
    per_c = {f"{m}_d{d}": {f"c{c:g}": ms(g2["test_ap"])[:2] for c, g2 in g.groupby("c")} for (m, d), g in pc.groupby(["map", "d"])}
    out = {"grid": grid, "per_c_test_ap": per_c, "variant": main["variant"], "grid_size": n0}
    try:
        bd = best_dims(cfg)
        out["best_d"] = bd
        out["learning_curve"] = {f"{m}_d{bd[m]}": {f"n{n}": ms(g["test_ap"]) for n, g in ch[(ch["map"] == m) & (ch["d"] == bd[m])].groupby("n")}
                                 for m in bd}
    except (ValueError, KeyError):
        pass
    out["note"] = ("mean / std over the seeds (qk_samples.seeds) of the chosen configuration of QI4.2 refitted on the whole "
                   "sample, Platt AP on the 2,000 test candidates (QH5); [mean, std, runs]; best d = argmax of the mean CV "
                   "AP over the seeds (ties -> smaller d); per_c = best configuration by CV at each c, refitted (Fig. 3b); "
                   "every run in results/qi4_runs.csv, CV tables in results/qi4_cv.csv, test scores in "
                   "<output>/stage2/qi4_test_scores.npz")
    save_report(cfg, "QI4.3", out)
    return out


def run_grid(cfg: dict, device: str = "cuda") -> None:
    """QI4.3 main grid: every d of ``quantum.dims`` and seed of ``qk_samples.seeds`` at the grid size, all maps; the
    per-c runs at ``stage2.main.per_c_dims``.  Units already in ``qi4_runs.csv`` are skipped (resume)."""
    main, seeds = cfg["stage2"]["main"], [int(s) for s in cfg["qk_samples"]["seeds"]]
    vid, n, maps = main["variant"], int(main["n"]), list(main["maps"])
    z = np.load(features_path(cfg))
    done = done_runs(cfg)
    for seed in seeds:
        x_tr, i_tr, x_te, i_te = load_qk(cfg, vid, n, seed, z)
        for d in [int(v) for v in cfg["quantum"]["dims"]]:
            per_c = d in [int(v) for v in main["per_c_dims"]]
            if all(run_id(m, d, n, seed) in done for m in maps) and \
                    (not per_c or all(run_id(m, d, n, seed, float(c)) in done for m in maps for c in cfg["quantum"]["bandwidths"])):
                continue
            run_unit(cfg, x_tr, i_tr, x_te, i_te, d, n, seed, maps, per_c, device, z[f"{vid}_test_index"])
    print(json.dumps({k: v["test_ap_mean_std_runs"] for k, v in qi4_summary(cfg)["grid"].items()}, indent=1))


def run_curve(cfg: dict, device: str = "cuda") -> None:
    """QI4.3 learning curve: every size of ``qk_samples.sizes`` and seed at the best d of each map (``best_dims``,
    from the complete main grid); maps with the same best d share a unit; done runs are skipped."""
    main, seeds = cfg["stage2"]["main"], [int(s) for s in cfg["qk_samples"]["seeds"]]
    vid = main["variant"]
    bd = best_dims(cfg)
    print(f"QI4.3 best d per map: {bd}", flush=True)
    z = np.load(features_path(cfg))
    done = done_runs(cfg)
    for n in [int(v) for v in cfg["qk_samples"]["sizes"]]:
        for seed in seeds:
            todo = [m for m in main["maps"] if run_id(m, bd[m], n, seed) not in done]
            if not todo:
                continue
            x_tr, i_tr, x_te, i_te = load_qk(cfg, vid, n, seed, z)
            for d in sorted({bd[m] for m in todo}):
                run_unit(cfg, x_tr, i_tr, x_te, i_te, d, n, seed, [m for m in todo if bd[m] == d], False, device,
                         z[f"{vid}_test_index"])
    print(json.dumps(qi4_summary(cfg)["learning_curve"], indent=1))


# ---------------------------------------------------------------------------
# QI4.4 - Metrics on the test candidates and end to end
# ---------------------------------------------------------------------------


def threshold_metrics(pred: np.ndarray, y: np.ndarray, band: np.ndarray) -> dict:
    """Precision, recall, F1 and share predicted positive of the 0 / 1 prediction ``pred`` on labels ``y`` (0 / 1);
    per band (``BANDS``) the recall of the positives of that band and the precision against all the negatives (the
    convention of ``band_masks``: positives of the band + every negative)."""
    pos, neg = y == 1, y == 0
    tp, fp = int(np.sum(pred[pos] == 1)), int(np.sum(pred[neg] == 1))
    out = {"precision": tp / max(tp + fp, 1), "recall": tp / max(int(pos.sum()), 1), "f1": f1(pred, y),
           "predicted_share": float(np.mean(pred == 1))}
    for b, name in enumerate(BANDS):
        pb = pos & (band == b)
        tpb = int(np.sum(pred[pb] == 1))
        out[f"recall_{name}"] = tpb / max(int(pb.sum()), 1)
        out[f"precision_{name}"] = tpb / max(tpb + fp, 1)
    return out


def precision_at_recall(score: np.ndarray, y: np.ndarray, recall: float) -> tuple[float, float]:
    """Stage 1 alone at a given recall: the rule score >= t with the highest t whose recall on ``y`` (0 / 1) reaches
    ``recall``; returns (precision, share of the samples flagged)."""
    o = np.argsort(-score, kind="stable")
    tp = np.cumsum(y[o] == 1)
    k = int(np.searchsorted(tp, max(np.ceil(recall * tp[-1] - 1e-9), 1)))   # first rank with tp >= target
    return float(tp[k] / (k + 1)), float((k + 1) / y.size)


def stage1_reference(cfg: dict, vid: str) -> dict:
    """Stage 1 alone on every labelled Test sub-tile of ``vid`` (QH3 logits, QH2 threshold = the candidate flag):
    {score, y (labelled), cand, cls, score_all (all Test sub-tiles), metrics of the candidate flag, counts}."""
    from qc_stage1 import load_candidates

    score, cand, idx = load_candidates(cfg, "test", vid)
    lab = idx["cls"] >= 0
    y, flag = idx["cls"][lab].astype(int), cand[lab].astype(int)
    return {"score": score[lab].astype(np.float64), "y": y, "cand": cand, "cls": idx["cls"], "score_all": score.astype(np.float64),
            "metrics": threshold_metrics(flag, y, idx["band"][lab]),
            "counts": {"subtiles": int(cand.size), "labelled_subtiles": int(lab.sum()), "positives": int(y.sum()),
                       "candidates": int(cand.sum()), "labelled_candidates": int(flag.sum())}}


RANK_BY_MARGIN = ("RFF", "MLP", "GBT")   # QJ1 methods ranked by their logit (the probability saturates, MLP)


def stage2_metrics(cfg: dict, vid: str, runs, z) -> dict:
    """QI4.4 metrics of the chosen runs ``runs`` (DataFrame rows of a runs CSV, one variant) from their test scores
    ``z`` (opened npz): per run the AP of the Platt score (of the logit for ``RANK_BY_MARGIN``) overall and per band, precision / recall / F1 at
    ``stage2.default_threshold`` overall and per band, the end-to-end estimate on all the labelled Test sub-tiles of
    ``vid`` (stage-1 recall x stage-2 recall, stage-2 precision of the proportional subset of QH5, flagged share) and
    stage 1 alone at its QH2 threshold and at the end-to-end recall of the run.  Returns {per_run (map, d, n ->
    [mean, std] over the seeds), stage1 (QH2 operating point, AP of the stage-1 logit on the same candidates),
    subset, checks}."""
    import pandas as pd

    thr = float(cfg["stage2"]["default_threshold"])
    s1 = stage1_reference(cfg, vid)
    m1 = s1["metrics"]
    y, band, ti = z["test_cls"].astype(int), z["test_band"], z["test_index"]
    masks = band_masks({"cls": y, "band": band})
    qh3 = load_report(cfg, "QH3")["sets"]["test"][vid]
    checks = {"subset_labels_match_stage1": bool(np.array_equal(s1["cls"][ti], y)),
              "subset_all_candidates": bool(s1["cand"][ti].all()), "subset_unique": bool(np.unique(ti).size == ti.size),
              "stage1_recall_precision_match_qh3": bool(abs(m1["recall"] - qh3["recall"]) < 1e-3
                                                        and abs(m1["precision"] - qh3["precision"]) < 1e-3)}
    rows, ap_diff, agree = [], 0.0, []
    for r in runs.itertuples():
        proba = z[f"{r.run_id}__proba"].astype(np.float64)
        rank = z[f"{r.run_id}__margin"].astype(np.float64) if r.map in RANK_BY_MARGIN else proba
        pred = (proba >= thr).astype(int)
        agree.append(float(np.mean(pred == (z[f"{r.run_id}__margin"] > 0))))
        m = threshold_metrics(pred, y, band)
        aps = {("ap" if k == "all" else f"ap_{k}"): average_precision(rank[mk], y[mk]) for k, mk in masks.items()}
        ap_diff = max(ap_diff, abs(aps["ap"] - r.test_ap))
        e2e = {"e2e_recall": m1["recall"] * m["recall"], "e2e_precision": m["precision"],
               "e2e_flagged_share": s1["counts"]["labelled_candidates"] / s1["counts"]["labelled_subtiles"] * m["predicted_share"],
               **{f"e2e_recall_{b}": m1[f"recall_{b}"] * m[f"recall_{b}"] for b in BANDS}}
        p_eq, share_eq = precision_at_recall(s1["score"], s1["y"], e2e["e2e_recall"])
        gain = {"stage1_precision_at_e2e_recall": p_eq, "stage1_flagged_share_at_e2e_recall": share_eq,
                "precision_gain_ratio_vs_qh2": m["precision"] / m1["precision"],
                "precision_gain_at_equal_recall": m["precision"] - p_eq}
        rows.append({"map": r.map, "d": r.d, "n": r.n, "seed": r.seed, **aps, **m, **e2e, **gain})
    df = pd.DataFrame(rows)
    checks.update({"runs": len(df), "max_abs_ap_diff_vs_runs_csv": round(ap_diff, 6),
                   "proba_vs_margin_agreement_min": round(min(agree), 4)})
    ms = lambda v: [round(float(v.mean()), 4), round(float(v.std(ddof=1)), 4) if len(v) > 1 else None]
    metrics = [c for c in df.columns if c not in ("map", "d", "n", "seed")]
    per_run = {f"{m}_d{d}": {f"n{n}": {k: ms(g2[k]) for k in metrics} for n, g2 in g.groupby("n")}
               for (m, d), g in df.groupby(["map", "d"])}
    stage1 = {**s1["counts"], **{k: round(v, 4) for k, v in m1.items() if k != "f1"},
              "candidate_share_of_labelled": round(s1["counts"]["labelled_candidates"] / s1["counts"]["labelled_subtiles"], 4)}
    s1_sub = s1["score_all"][ti]                                       # stage-1 logit of the same 2,000 candidates
    stage1["subset_ap"] = {("all" if k == "all" else k): round(average_precision(s1_sub[mk], y[mk]), 4) for k, mk in masks.items()}
    subset = {"candidates": int(y.size), "positives": int(y.sum()),
              "positives_per_band": [int(np.sum((y == 1) & (band == b))) for b in range(len(BANDS))]}
    return {"per_run": per_run, "stage1": stage1, "subset": subset, "checks": checks}


def run_metrics(cfg: dict) -> dict:
    """QI4.4: ``stage2_metrics`` of every chosen run of QI4.3 at the best d of its map (all sizes n, all seeds), on
    F-X.  Default threshold = Platt probability >= 0.5, end to end estimated from the QH5 subset, precision gain over
    stage 1 alone at its QH2 threshold and at the same end-to-end recall (decided with the user).  Report: mean / std
    over the seeds per map and n."""
    import pandas as pd

    main = cfg["stage2"]["main"]
    vid = main["variant"]
    p = qi4_paths(cfg)
    runs, z = pd.read_csv(p["runs"]), np.load(p["scores"])
    bd = best_dims(cfg)
    ch = runs[(runs["role"] == "chosen") & (runs["variant"] == vid)]
    ch = ch[[d == bd[m] for m, d in zip(ch["map"], ch["d"])]]
    r = stage2_metrics(cfg, vid, ch, z)
    out = {"variant": vid, "default_threshold": float(cfg["stage2"]["default_threshold"]), "best_d": bd,
           "subset": r["subset"], "stage1_alone_qh2": r["stage1"], "stage2": r["per_run"], "checks": r["checks"],
           "note": ("[mean, std] over the seeds of the chosen QI4.3 run at the best d of each map; stage-2 metrics on the "
                    "2,000 test candidates of QH5, predicted positive = Platt probability >= default_threshold; per band: "
                    "positives of the band + all the negatives; e2e_* = estimate on all the labelled Test sub-tiles "
                    "(stage-1 recall x stage-2 recall, stage-2 precision of the proportional subset); "
                    "stage1_*_at_e2e_recall = stage 1 alone with its threshold raised to the e2e recall of the run; "
                    "stage1_alone_qh2.subset_ap = AP of the stage-1 logit on the same 2,000 candidates; "
                    "4-8 km band on few positives (see subset)")}
    save_report(cfg, "QI4.4", out)
    return out


# ---------------------------------------------------------------------------
# QI6.1 - The quantum stage on every representation
# ---------------------------------------------------------------------------

QI6_KEYS = ("ap", "f1", "precision", "recall", "e2e_recall", "e2e_precision", "stage1_precision_at_e2e_recall",
            "precision_gain_at_equal_recall")


def run_representations(cfg: dict, device: str = "cuda") -> None:
    """QI6.1 runs (decided with the user): on every variant of ``stage2.representations.variants`` (each with its
    own stage 1, QK samples of QH4 and test candidates of QH5), every map of ``stage2.main.maps`` at its best d of
    QI4.3 (``best_dims``), at the grid size, every seed: the CV of QI4.2 on the sample of that representation and the
    refit of the chosen configuration (``run_unit``), saved in ``qi6_runs.csv``, ``qi6_cv.csv`` and
    ``stage2/qi6_<variant>_test_scores.npz``; done runs skipped (resume).  F-X is not rerun (its runs are QI4.3)."""
    main = cfg["stage2"]["main"]
    n, seeds, maps = int(main["n"]), [int(s) for s in cfg["qk_samples"]["seeds"]], list(main["maps"])
    bd = best_dims(cfg)
    z = np.load(features_path(cfg))
    for vid in cfg["stage2"]["representations"]["variants"]:
        paths = stage2_paths(cfg, "qi6", vid)
        done = done_runs(cfg, paths)
        for seed in seeds:
            todo = [m for m in maps if run_id(m, bd[m], n, seed, None, vid) not in done]
            if not todo:
                continue
            x_tr, i_tr, x_te, i_te = load_qk(cfg, vid, n, seed, z)
            for d in sorted({bd[m] for m in todo}):
                run_unit(cfg, x_tr, i_tr, x_te, i_te, d, n, seed, [m for m in todo if bd[m] == d], False, device,
                         z[f"{vid}_test_index"], variant=vid, step="QI6.1", paths=paths)


def qi6_summary(cfg: dict) -> dict:
    """QI6.1 report: per representation (F-X from QI4.3 at the same d and n) and map, the ``stage2_metrics`` of the
    chosen runs ([mean, std] over the seeds, the keys of ``QI6_KEYS``), the chosen c / C, and stage 1 alone (QH2
    operating point, AP of its logit on the same candidates); checks per representation."""
    import pandas as pd

    main = cfg["stage2"]["main"]
    n, bd = int(main["n"]), best_dims(cfg)
    sources = [(main["variant"], qi4_paths(cfg))] + [(v, stage2_paths(cfg, "qi6", v))
                                                     for v in cfg["stage2"]["representations"]["variants"]]
    out, checks = {}, {}
    for vid, p in sources:
        runs = pd.read_csv(p["runs"])
        ch = runs[(runs["role"] == "chosen") & (runs["variant"] == vid) & (runs["n"] == n)]
        ch = ch[[d == bd[m] for m, d in zip(ch["map"], ch["d"])]]
        r = stage2_metrics(cfg, vid, ch, np.load(p["scores"]))
        out[vid] = {"stage1_alone_qh2": {k: r["stage1"][k] for k in ("recall", "precision", "subset_ap")},
                    "subset_positives": r["subset"]["positives"],
                    **{k: {**{m: v[f"n{n}"][m] for m in QI6_KEYS},
                           "chosen_c": ch[ch["map"] == k.split("_d")[0]]["c"].value_counts().to_dict(),
                           "chosen_C": ch[ch["map"] == k.split("_d")[0]]["C"].value_counts().to_dict()}
                       for k, v in r["per_run"].items()}}
        checks[vid] = r["checks"]
    rep = {"n": n, "best_d": bd, "default_threshold": float(cfg["stage2"]["default_threshold"]), "variants": out,
           "checks": checks,
           "note": ("per representation (each with its own stage 1, QK samples and 2,000 test candidates) and map: the "
                    "chosen configuration of the spatial CV (QI4.2) on its own sample at the best d of QI4.3, refitted, "
                    "[mean, std] over the seeds; F-X = the QI4.3 runs; metrics as QI4.4 (Platt probability >= "
                    "default_threshold; e2e = stage-1 recall x stage-2 recall, stage-2 precision; stage 1 alone at the "
                    "e2e recall); runs in results/qi6_runs.csv, test scores in <output>/stage2/qi6_<variant>_test_scores.npz")}
    save_report(cfg, "QI6.1", rep)
    return rep


# ---------------------------------------------------------------------------
# QJ1 - Classical baselines on the same inputs
# ---------------------------------------------------------------------------

QJ_COLS = ["step", "script", "run_id", "role", "variant", "map", "d", "n", "seed", "s", "C", "params", "cv_ap", "cv_ap_std",
           "cv_f1", "test_ap", "test_f1", "support_vectors", "train_test_seconds", "seconds", "timestamp"]
SVM_METHODS = ("RBF", "RBF-wide", "LIN", "LIN-full")


def classical_grid(cfg: dict, method: str) -> list[dict]:
    """QJ1 hyper-parameter grid of ``method`` (section ``classical`` and ``stage2.svm_C``), simplest configuration
    first (the tie rule of ``select_classical``): RBF / RBF-wide C x s (gamma = s x the median gamma), LIN C, RFF
    C x s, MLP hidden units x alpha (strongest penalty first), GBT iterations x depth x learning rate."""
    import itertools

    cl, Cs = cfg["classical"], [float(v) for v in cfg["stage2"]["svm_C"]]
    if method in ("RBF", "RBF-wide"):
        sc = cl["rbf_gamma_scales" if method == "RBF" else "rbf_wide_gamma_scales"]
        return [{"C": C, "s": float(s)} for C, s in itertools.product(Cs, sc)]
    if method in ("LIN", "LIN-full"):
        return [{"C": float(C)} for C in cl["linear_C"]]
    if method == "RFF":
        return [{"C": float(C), "s": float(s)} for C, s in itertools.product(cl["rff"]["C"], cl["rbf_gamma_scales"])]
    if method == "MLP":
        return [{"hidden": int(h), "alpha": float(a)} for h, a in itertools.product(cl["mlp"]["hidden"], cl["mlp"]["alpha"])]
    if method == "GBT":
        g = cl["gbt"]
        return [{"max_iter": int(i), "max_depth": int(dp), "learning_rate": float(lr)}
                for i, dp, lr in itertools.product(g["max_iter"], g["max_depth"], g["learning_rate"])]
    raise ValueError(f"unknown classical method {method}")


def classical_scores(cfg: dict, method: str, p: dict, a_tr: np.ndarray, y_tr: np.ndarray, a_te: np.ndarray, seed: int,
                     calibrate: bool = True) -> dict:
    """Fit ``method`` with the configuration ``p`` on the standardised components ``a_tr`` (labels ``y_tr``) and score
    ``a_te``: {score (ranking score), proba, margin, pred, support_vectors}.  RBF / RBF-wide / LIN:
    ``SVC(kernel="precomputed", class_weight="balanced")`` on the RBF kernel (gamma = s x the median gamma of ``a_tr``)
    or the linear kernel, as the QSVC: with ``calibrate`` (refit) the Platt score of ``fit_svc``, otherwise (CV) the
    bare SVC, whose margin gives the same AP.  RFF: ``RBFSampler`` (same gamma, sample seed) + class-balanced
    ``LogisticRegression``; MLP: ``MLPClassifier`` (one hidden layer, L-BFGS, sample seed); GBT: class-balanced
    ``HistGradientBoostingClassifier`` without early stopping (sample seed); their ranking score and margin are the
    exact logit (``decision_function``, ``mlp_logit``: the MLP probability saturates), the probability is kept.  Predicted positive: SVM margin > 0, otherwise probability >= 0.5 (the rule of QI4)."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.svm import SVC

    if method in SVM_METHODS:
        if method in ("LIN", "LIN-full"):
            k_tr, k_te = a_tr @ a_tr.T, a_te @ a_tr.T
        else:
            g = p["s"] * median_gamma(a_tr)
            k_tr, k_te = kernel_projected(a_tr, a_tr, g), kernel_projected(a_te, a_tr, g)
        if calibrate:
            model = fit_svc(k_tr, y_tr, p["C"])
            sc = svc_scores(model, k_te)
            return {**sc, "score": sc["margin"], "support_vectors": int(fitted_svc(model).n_support_.sum())}
        svc = SVC(kernel="precomputed", C=p["C"], class_weight="balanced").fit(k_tr, y_tr)
        m = svc.decision_function(k_te)
        return {"score": m, "margin": m, "proba": None, "pred": svc.predict(k_te), "support_vectors": int(svc.n_support_.sum())}
    if method == "RFF":
        from sklearn.kernel_approximation import RBFSampler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        rff = RBFSampler(gamma=p["s"] * median_gamma(a_tr), n_components=int(cfg["classical"]["rff"]["components"]),
                         random_state=seed)
        model = make_pipeline(rff, LogisticRegression(C=p["C"], class_weight="balanced", max_iter=5000))
    elif method == "MLP":
        from sklearn.neural_network import MLPClassifier

        model = MLPClassifier(hidden_layer_sizes=(p["hidden"],), alpha=p["alpha"], solver="lbfgs",
                              max_iter=int(cfg["classical"]["mlp"]["max_iter"]), random_state=seed)
    elif method == "GBT":
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(max_iter=p["max_iter"], max_depth=p["max_depth"],
                                               learning_rate=p["learning_rate"], early_stopping=False,
                                               class_weight="balanced", random_state=seed)
    else:
        raise ValueError(f"unknown classical method {method}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(a_tr, y_tr)
    proba = model.predict_proba(a_te)[:, list(model.classes_).index(1)]
    logit = mlp_logit(model, a_te) if method == "MLP" else model.decision_function(a_te)
    return {"score": logit, "proba": proba, "margin": logit, "pred": (proba >= 0.5).astype(int), "support_vectors": None}


def mlp_logit(model, a: np.ndarray) -> np.ndarray:
    """Pre-sigmoid output of a binary ``MLPClassifier`` (ReLU hidden layers) on ``a``: the exact ranking score, which
    ``predict_proba`` loses where the sigmoid saturates (many test probabilities equal to 1 in float64 / float32)."""
    assert model.activation == "relu" and model.out_activation_ == "logistic" and list(model.classes_) == [0, 1]
    h = a
    for w, b in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        h = np.maximum(h @ w + b, 0.0)
    return (h @ model.coefs_[-1] + model.intercepts_[-1]).ravel()


def _cv_task(cfg: dict, m: str, k: int, p: dict, f: int, a_tr, y_tr, a_va, y_va, seed: int) -> dict:
    sc = classical_scores(cfg, m, p, a_tr, y_tr, a_va, seed, calibrate=False)
    return {"map": m, "config": k, "params": json.dumps(p), "fold": f,
            "ap": average_precision(sc["score"], y_va), "f1": f1(sc["pred"], y_va)}


def standardize(x_tr: np.ndarray, *others: np.ndarray, ref_dim: int | None = None) -> list[np.ndarray]:
    """QJ2: the full vectors standardised per feature with the mean / std (ddof 1) of ``x_tr`` (no PCA); with
    ``ref_dim`` also scaled by sqrt(ref_dim / D), so that their mean squared norm equals that of ``ref_dim``
    standardised PCA components (a linear SVM is equivalent under x -> a x, C -> C / a^2: the C grid of LIN keeps its
    meaning, and the large C stay fast)."""
    mu, sd = x_tr.mean(0), x_tr.std(0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    k = 1.0 if ref_dim is None else float(np.sqrt(ref_dim / x_tr.shape[1]))
    return [(k * (x - mu) / sd).astype(np.float64) for x in (x_tr, *others)]


def reduce_inputs(x_tr: np.ndarray, d: int | None, *others: np.ndarray, ref_dim: int | None = None) -> list[np.ndarray]:
    """Inputs of stage 2: the d standardised PCA components of QI1.1, or with ``d`` None the full standardised
    vectors scaled to ``ref_dim`` (QJ2, ``classical.full_reference_dim``)."""
    return standardize(x_tr, *others, ref_dim=ref_dim) if d is None else reduce_features(x_tr, d, *others)


def classical_cv(cfg: dict, x: np.ndarray, y: np.ndarray, blocks: np.ndarray, d: int | None, seed: int, methods,
                 n_jobs: int = 1) -> list[dict]:
    """QJ1 cross-validation with the protocol of QI4.2 (``spatial_folds``, PCA of QI1.1 refitted on every training
    fold, AP of the ranking score on the validation fold, F1 of the prediction) for every configuration of
    ``classical_grid``: one row per method, configuration and fold.  The (fold, configuration) fits run in ``n_jobs``
    processes (joblib) with **one OpenMP / BLAS thread each**, whatever ``OMP_NUM_THREADS`` of the job: joblib passes
    an inherited value to every worker, and n_jobs x OMP_NUM_THREADS threads on n_jobs cores slow GBT down by ~10x
    or more (job 50962).  Every fit is seeded, so the rows do not depend on ``n_jobs``."""
    from joblib import Parallel, delayed, parallel_config

    tasks = []
    for f, (tr, va) in enumerate(spatial_folds(y, blocks, int(cfg["stage2"]["cv_folds"]), seed)):
        a_tr, a_va = reduce_inputs(x[tr], d, x[va], ref_dim=cfg["classical"].get("full_reference_dim"))
        tasks += [(m, k, p, f, a_tr, y[tr], a_va, y[va]) for m in methods for k, p in enumerate(classical_grid(cfg, m))]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=n_jobs)(delayed(_cv_task)(cfg, *t, seed) for t in tasks)


def classical_table(rows: list[dict]) -> list[dict]:
    """Mean / std over the folds of every (method, configuration)."""
    groups: dict = {}
    for row in rows:
        groups.setdefault((row["map"], row["config"]), []).append(row)
    out = []
    for (m, k), g in groups.items():
        ap, f = np.array([v["ap"] for v in g]), np.array([v["f1"] for v in g])
        out.append({"map": m, "config": k, "params": g[0]["params"], "cv_ap": float(ap.mean()),
                    "cv_ap_std": float(ap.std(ddof=1)), "cv_f1": float(f.mean()), "folds": len(g)})
    return out


def select_classical(table: list[dict], method: str) -> dict:
    """QJ1 choice for ``method``: highest mean fold AP; ties (1e-9) -> the first configuration of the grid."""
    return min((t for t in table if t["map"] == method), key=lambda t: (-round(t["cv_ap"], 9), t["config"]))


def run_classical_unit(cfg: dict, x_tr, i_tr, x_te, i_te, d: int | None, n: int, seed: int, vid: str, te_index, methods,
                       paths: dict, n_jobs: int = 1, step: str = "QJ1") -> None:
    """One QJ1 unit (variant, d, n, seed): ``classical_cv`` for all ``methods``, the chosen configuration of every
    method refitted on the whole sample and scored on the test candidates (Platt for the SVMs), saved by
    ``save_unit`` (``QJ_COLS``; ``train_test_seconds`` = refit + scoring of the test candidates, for QJ4)."""
    import datetime

    y_tr, y_te = i_tr["cls"].astype(int), i_te["cls"].astype(int)
    t0 = time.time()
    table = classical_table(classical_cv(cfg, x_tr, y_tr, i_tr["block"], d, seed, methods, n_jobs))
    a_tr, a_te = reduce_inputs(x_tr, d, x_te, ref_dim=cfg["classical"].get("full_reference_dim"))
    d = a_tr.shape[1]                                                     # full dimension when d is None (QJ2)
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    runs, scores = [], {f"test_{k}": np.asarray(v) for k, v in i_te.items()}
    scores["test_index"] = np.asarray(te_index)
    for m in methods:
        b = select_classical(table, m)
        p = json.loads(b["params"])
        t1 = time.time()
        sc = classical_scores(cfg, m, p, a_tr, y_tr, a_te, seed)
        rid = run_id(m, d, n, seed, None, vid)
        runs.append({"step": step, "script": "qc_experiments.py", "run_id": rid, "role": "chosen", "variant": vid, "map": m,
                     "d": d, "n": n, "seed": seed, "s": p.get("s"), "C": p.get("C"), "params": b["params"],
                     "cv_ap": b["cv_ap"], "cv_ap_std": b["cv_ap_std"], "cv_f1": b["cv_f1"],
                     "test_ap": average_precision(sc["margin"] if m in RANK_BY_MARGIN else sc["proba"], y_te),
                     "test_f1": f1(sc["pred"], y_te),
                     "support_vectors": sc["support_vectors"], "train_test_seconds": round(time.time() - t1, 3),
                     "timestamp": stamp})
        scores.update({f"{rid}__proba": sc["proba"].astype(np.float32), f"{rid}__margin": sc["margin"].astype(np.float32)})
    for r_ in runs:
        r_["seconds"] = round(time.time() - t0, 1)
    unit = f"{vid}_d{d}_n{n}_s{seed}"
    save_unit(cfg, unit, [{"variant": vid, "d": d, "n": n, "seed": seed, **t} for t in table], runs, scores, paths, QJ_COLS)
    for r_ in runs:
        print(f"{step} {r_['run_id']}: {r_['params']}, CV AP {r_['cv_ap']:.3f}, test AP {r_['test_ap']:.3f}", flush=True)
    print(f"{step} unit {unit}: {time.time() - t0:.0f} s", flush=True)


def qj1_units(cfg: dict) -> list[tuple[str, int, int]]:
    """(variant, d, n) of QJ1, the mirror of the quantum runs (decided with the user): F-X at every d and the grid
    size (QI4.3 grid), every other representation at the best d of QI4.3 and the grid size (QI6.1), F-X at every size
    and the best d (QI4.3 learning curve, last: the slowest)."""
    main = cfg["stage2"]["main"]
    vid, n0, bds = main["variant"], int(main["n"]), sorted(set(best_dims(cfg).values()))
    units = [(vid, int(d), n0) for d in cfg["quantum"]["dims"]]
    units += [(v, d, n0) for v in cfg["stage2"]["representations"]["variants"] for d in bds]
    units += [(vid, d, int(n)) for d in bds for n in cfg["qk_samples"]["sizes"] if int(n) != n0]
    return units


def run_baselines(cfg: dict, methods=None, redo: bool = False, redo_before: str | None = None) -> None:
    """QJ1 runs: every unit of ``qj1_units`` x seed of ``qk_samples.seeds``, all ``methods`` (default
    ``classical.methods``); a unit is skipped when all its runs are in ``results/qj1_runs.csv`` (resume), otherwise
    all its methods are rerun; with ``redo`` every unit is rerun for ``methods`` (e.g. one method after a fix: its runs,
    CV rows and scores are replaced, those of the other methods kept); with ``redo_before`` (ISO time) only the runs
    written before that time are redone (to resume an interrupted redo).  CV fits in parallel on the CPUs of the
    SLURM job (``SLURM_CPUS_PER_TASK``, 1 outside a job)."""
    import os

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    methods = list(methods or cfg["classical"]["methods"])
    seeds = [int(s) for s in cfg["qk_samples"]["seeds"]]
    z = np.load(features_path(cfg))
    done = done_runs(cfg, stage2_paths(cfg, "qj1"))
    if redo_before is not None:
        import pandas as pd

        r = pd.read_csv(stage2_paths(cfg, "qj1")["runs"])
        done, redo = set(r.loc[r["timestamp"] >= redo_before, "run_id"]), False
    for vid, d, n in qj1_units(cfg):
        for seed in seeds:
            if not redo and all(run_id(m, d, n, seed, None, vid) in done for m in methods):
                continue
            x_tr, i_tr, x_te, i_te = load_qk(cfg, vid, n, seed, z)
            run_classical_unit(cfg, x_tr, i_tr, x_te, i_te, d, n, seed, vid, z[f"{vid}_test_index"], methods,
                               stage2_paths(cfg, "qj1", vid), n_jobs)


def qj1_summary(cfg: dict) -> dict:
    """QJ1 report from ``qj1_runs.csv``: F-X grid per method and d (test / CV AP [mean, std, runs], F1, most chosen
    configurations) with the QSVC of QI4.3 at the same d as reference; learning curve at the best d; per
    representation at the best d and the grid size the QI4.4 metrics (``stage2_metrics``) with the QSVC test AP of
    QI4.3 / QI6.1; the RBF-wide sensitivity check (choices at the lower edge of the RBF grid, test AP paired per
    sample); median refit + scoring time per method and n (QJ4); checks."""
    import pandas as pd

    main, cl = cfg["stage2"]["main"], cfg["classical"]
    vid0, n0 = main["variant"], int(main["n"])
    bds = sorted(set(best_dims(cfg).values()))
    runs = pd.read_csv(stage2_paths(cfg, "qj1")["runs"])
    ms = lambda v: [round(float(v.mean()), 4), round(float(v.std(ddof=1)), 4) if len(v) > 1 else None, int(len(v))]
    fx = runs[runs["variant"] == vid0]
    q4 = pd.read_csv(qi4_paths(cfg)["runs"])
    q4 = q4[q4["role"] == "chosen"]
    grid = {f"{m}_d{d}": {"test_ap": ms(g["test_ap"]), "cv_ap": ms(g["cv_ap"]), "test_f1_mean": round(float(g["test_f1"].mean()), 4),
                          "chosen": g["params"].value_counts().head(3).to_dict()}
            for (m, d), g in fx[fx["n"] == n0].groupby(["map", "d"])}
    quantum_grid = {f"{m}_d{d}": ms(g["test_ap"])[:2] for (m, d), g in q4[q4["n"] == n0].groupby(["map", "d"])}
    curve = {f"{m}_d{d}": {f"n{n}": ms(g2["test_ap"]) for n, g2 in g.groupby("n")}
             for (m, d), g in fx[fx["d"].isin(bds)].groupby(["map", "d"])}
    q6 = pd.read_csv(stage2_paths(cfg, "qi6")["runs"])
    qall = pd.concat([q4, q6[q6["role"] == "chosen"]], ignore_index=True)
    reps, checks = {}, {}
    for v in [vid0] + list(cfg["stage2"]["representations"]["variants"]):
        ch = runs[(runs["variant"] == v) & (runs["n"] == n0) & runs["d"].isin(bds)]
        r = stage2_metrics(cfg, v, ch, np.load(stage2_paths(cfg, "qj1", v)["scores"]))
        qv = qall[(qall["variant"] == v) & (qall["n"] == n0) & qall["d"].isin(bds)]
        reps[v] = {"stage1_subset_ap": r["stage1"]["subset_ap"]["all"],
                   "quantum_test_ap": {f"{m}_d{d}": ms(g["test_ap"])[:2] for (m, d), g in qv.groupby(["map", "d"])},
                   **{k: {m: e[f"n{n0}"][m] for m in QI6_KEYS} for k, e in r["per_run"].items()}}
        checks[v] = r["checks"]
    lo = min(float(s) for s in cl["rbf_gamma_scales"])
    sens = {}
    if "RBF-wide" in set(runs["map"]):
        a = runs[runs["map"] == "RBF"].set_index(["variant", "d", "n", "seed"])
        b = runs[runs["map"] == "RBF-wide"].set_index(["variant", "d", "n", "seed"])
        j = a.merge(b, left_index=True, right_index=True, suffixes=("", "_w"))
        j["d_"] = j.index.get_level_values("d")
        sens = {"rbf_lower_edge": lo, "pairs": int(len(j)),
                "rbf_share_at_lower_edge_per_d": {int(d): round(float((g["s"] == lo).mean()), 3) for d, g in j.groupby("d_")},
                "wide_share_below_edge_per_d": {int(d): round(float((g["s_w"] < lo).mean()), 3) for d, g in j.groupby("d_")},
                "test_ap_wide_minus_rbf_per_d": {int(d): ms(g["test_ap_w"] - g["test_ap"])[:2] for d, g in j.groupby("d_")},
                "cv_ap_wide_minus_rbf_per_d": {int(d): ms(g["cv_ap_w"] - g["cv_ap"])[:2] for d, g in j.groupby("d_")}}
    times = {m: {f"n{n}": round(float(g2["train_test_seconds"].median()), 3) for n, g2 in g.groupby("n")}
             for m, g in fx[fx["d"].isin(bds)].groupby("map")}
    expected = len(qj1_units(cfg)) * len(cfg["qk_samples"]["seeds"]) * len(cl["methods"])
    checks["all"] = {"runs": int(len(runs)), "expected": expected, "run_ids_unique": bool(runs["run_id"].is_unique)}
    rep = {"methods": cl["methods"], "grid_sizes": {m: len(classical_grid(cfg, m)) for m in cl["methods"]}, "grid_size_n": n0,
           "best_d": bds, "grid": grid, "quantum_grid_test_ap": quantum_grid, "learning_curve": curve,
           "representations": reps, "rbf_wide_sensitivity": sens, "median_train_test_seconds": times, "checks": checks,
           "note": ("chosen configuration of the spatial CV of QI4.2 (same samples, folds, per-fold PCA, criterion) refitted "
                    "on the whole sample; [mean, std, runs] over the seeds; test AP of the probability (Platt for the SVMs) "
                    "on the 2,000 test candidates of each representation; representations: QI4.4 metrics at the best d and "
                    "the grid size, [mean, std]; quantum_* = QSVC of QI4.3 / QI6.1 at the same d and n; RBF-wide = RBF with "
                    "the extended gamma grid, sensitivity check only; runs in results/qj1_runs.csv, CV tables in "
                    "results/qj1_cv.csv, test scores in <output>/stage2/qj1_<variant>_test_scores.npz")}
    save_report(cfg, "QJ1", rep)
    return rep


# ---------------------------------------------------------------------------
# QJ2 - Reference lines: stage 1 alone, the linear probe and the linear SVM without compression
# ---------------------------------------------------------------------------


def qg1_on_candidates(cfg: dict, vid: str) -> dict:
    """The linear probe of QG1 (class-balanced logistic regression on the full vectors of all S1-train, ``qc_probes``)
    on the 2,000 test candidates of ``vid`` (QH5), matched through the cell (ci, cj): {score = its logit, y, band}.
    Checks: every candidate found, same class as ``stage1_scores.npz``."""
    from qc_probes import probes_dir
    from qc_stage1 import stage1_dir

    s1 = np.load(stage1_dir(cfg) / "stage1_scores.npz")
    sub = np.load(stage1_dir(cfg) / "test_subset.npz")[vid]
    ci, cj = s1["test_ci"][sub].astype(np.int64), s1["test_cj"][sub].astype(np.int64)
    z = np.load(probes_dir(cfg) / "qg1_test_scores.npz")
    key = z["ci"].astype(np.int64) * 100_000 + z["cj"].astype(np.int64)
    order = np.argsort(key, kind="stable")
    idx = order[np.searchsorted(key[order], ci * 100_000 + cj)]
    y = s1["test_cls"][sub].astype(int)
    if not (np.array_equal(z["ci"][idx], ci) and np.array_equal(z["cj"][idx], cj) and np.array_equal(z["cls"][idx].astype(int), y)):
        raise ValueError(f"QG1 scores do not match the test candidates of {vid}")
    return {"score": z[f"score_{vid}"][idx].astype(np.float64), "y": y, "band": s1["test_band"][sub].astype(int)}


def qj2_units(cfg: dict) -> list[tuple[str, int]]:
    """(variant, n) of the LIN-full runs (decided with the user): F-X at every size of ``qk_samples.sizes`` (the
    learning curve of QI4.3 / QJ1), every other representation at the grid size (QI6.1 / QJ1)."""
    main = cfg["stage2"]["main"]
    vid, n0 = main["variant"], int(main["n"])
    return [(vid, int(n)) for n in cfg["qk_samples"]["sizes"]] + [(v, n0) for v in cfg["stage2"]["representations"]["variants"]]


def run_references(cfg: dict) -> None:
    """QJ2 runs: the linear SVM of QJ1 on the full standardised vectors (``LIN-full``, no PCA) of the same QK samples,
    with the same spatial CV, C grid, refit and Platt score, on every unit of ``qj2_units`` x seed; saved in
    ``results/qj2_runs.csv``, ``qj2_cv.csv``, ``<output>/stage2/qj2_<variant>_test_scores.npz``; done runs skipped."""
    import os

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    z = np.load(features_path(cfg))
    done = done_runs(cfg, stage2_paths(cfg, "qj2"))
    for vid, n in qj2_units(cfg):
        dim = int(z[f"{vid}_qk_x"].shape[1])
        for seed in [int(s) for s in cfg["qk_samples"]["seeds"]]:
            if run_id("LIN-full", dim, n, seed, None, vid) in done:
                continue
            x_tr, i_tr, x_te, i_te = load_qk(cfg, vid, n, seed, z)
            run_classical_unit(cfg, x_tr, i_tr, x_te, i_te, None, n, seed, vid, z[f"{vid}_test_index"], ["LIN-full"],
                               stage2_paths(cfg, "qj2", vid), n_jobs, step="QJ2")


def qj2_summary(cfg: dict) -> dict:
    """QJ2 report: per representation, on its 2,000 test candidates, the reference lines of stage 2: stage 1 alone (AP
    of its logit; its precision at the recall of every stage-2 run is in QI4.4 / QI6.1 / QJ1), the linear probe of QG1
    (full vectors, all S1-train: AP overall and per band, precision / recall / F1 at logit 0 = probability 0.5), the
    linear SVM on the full vectors of the QK samples (LIN-full, QI4.4 metrics, [mean, std] over the seeds) next to the
    same SVM on the PCA components (LIN of QJ1) and the QSVC (QI4.3 / QI6.1) at the best d; the decomposition of the gap
    stage 1 - stage 2 (data and model: stage 1 - probe; samples: probe - LIN-full; compression: LIN-full - LIN); F-X
    LIN-full vs LIN for every n."""
    import pandas as pd

    main = cfg["stage2"]["main"]
    vid0, n0 = main["variant"], int(main["n"])
    bd = best_dims(cfg)
    d0 = sorted(set(bd.values()))[0]
    runs = pd.read_csv(stage2_paths(cfg, "qj2")["runs"])
    qj1 = pd.read_csv(stage2_paths(cfg, "qj1")["runs"])
    q = pd.concat([pd.read_csv(qi4_paths(cfg)["runs"]), pd.read_csv(stage2_paths(cfg, "qi6")["runs"])], ignore_index=True)
    q = q[(q["role"] == "chosen") & (q["d"] == d0)]
    ms = lambda v: [round(float(v.mean()), 4), round(float(v.std(ddof=1)), 4) if len(v) > 1 else None, int(len(v))]
    out, checks = {}, {}
    for v in [vid0] + list(cfg["stage2"]["representations"]["variants"]):
        ch = runs[(runs["variant"] == v) & (runs["n"] == n0)]
        r = stage2_metrics(cfg, v, ch, np.load(stage2_paths(cfg, "qj2", v)["scores"]))
        pr = qg1_on_candidates(cfg, v)
        masks = band_masks({"cls": pr["y"], "band": pr["band"]})
        probe = {("ap" if k == "all" else f"ap_{k}"): round(average_precision(pr["score"][mk], pr["y"][mk]), 4) for k, mk in masks.items()}
        probe.update({k: round(val, 4) for k, val in threshold_metrics((pr["score"] > 0).astype(int), pr["y"], pr["band"]).items()
                      if k in ("precision", "recall", "f1", "predicted_share")})
        lin16 = qj1[(qj1["variant"] == v) & (qj1["map"] == "LIN") & (qj1["n"] == n0) & (qj1["d"] == d0)]["test_ap"]
        lin_full = r["per_run"][next(iter(r["per_run"]))][f"n{n0}"]
        qv = q[(q["variant"] == v) & (q["n"] == n0)].groupby("map")["test_ap"].mean()
        s1_ap = r["stage1"]["subset_ap"]["all"]
        out[v] = {"stage1_alone_ap": s1_ap, "qg1_probe_full_all_s1train": probe,
                  "lin_full_qk_sample": {k: lin_full[k] for k in QI6_KEYS},
                  f"lin_pca_d{d0}_test_ap": ms(lin16), f"qsvc_d{d0}_test_ap_mean": {m: round(float(a), 4) for m, a in qv.items()},
                  "gap_decomposition_ap": {"stage1_minus_probe": round(s1_ap - probe["ap"], 4),
                                           "probe_minus_lin_full": round(probe["ap"] - lin_full["ap"][0], 4),
                                           "lin_full_minus_lin_pca": round(lin_full["ap"][0] - float(lin16.mean()), 4)}}
        checks[v] = {**r["checks"], "qg1_matched_candidates": int(pr["y"].size)}
    fx_full = runs[runs["variant"] == vid0].groupby("n")["test_ap"]
    fx_lin = qj1[(qj1["variant"] == vid0) & (qj1["map"] == "LIN") & (qj1["d"] == d0)].groupby("n")["test_ap"]
    curve = {f"n{n}": {"lin_full": ms(g), f"lin_pca_d{d0}": ms(fx_lin.get_group(n))} for n, g in fx_full}
    checks["all"] = {"runs": int(len(runs)), "expected": len(qj2_units(cfg)) * len(cfg["qk_samples"]["seeds"]),
                     "run_ids_unique": bool(runs["run_id"].is_unique)}
    rep = {"best_d": d0, "grid_size_n": n0, "representations": out, "fx_learning_curve": curve, "checks": checks,
           "note": ("reference lines on the 2,000 test candidates of each representation: stage 1 alone = AP of the stage-1 "
                    "logit; qg1_probe = the QG1 logistic probe on the full vectors trained on all S1-train (threshold logit "
                    "0 = probability 0.5); lin_full = linear SVM on the full standardised vectors of the QK samples (same CV, "
                    "C grid, refit, Platt score as LIN of QJ1), QI4.4 metrics [mean, std] over the seeds; lin_pca = LIN of QJ1 "
                    "on the PCA components; gap decomposition: stage1 - probe = model + context + data, probe - lin_full = "
                    "training samples, lin_full - lin_pca = the PCA compression at equal data")}
    save_report(cfg, "QJ2", rep)
    return rep


# ---------------------------------------------------------------------------
# QJ3 - Statistics: block-bootstrap CI, McNemar QSVC vs classical
# ---------------------------------------------------------------------------

QJ3_REFERENCES = ("RBF", "LIN")      # classical side of the paired tests (RBF of the plan, LIN added with the user)


def mcnemar_p(err_a: np.ndarray, err_b: np.ndarray) -> tuple[float, int, int]:
    """Exact two-sided McNemar test of two classifiers on the same samples: (p, b = errors of a only, c = errors of
    b only); p = binomial test of min(b, c) in b + c trials at 0.5 (1 when the two never disagree)."""
    from scipy.stats import binomtest

    b, c = int(np.sum(err_a & ~err_b)), int(np.sum(~err_a & err_b))
    return (1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)), b, c


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error), same order as ``p``."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate(p[o] * (p.size - np.arange(p.size))))
    out = np.empty_like(adj)
    out[o] = adj
    return out


def qj3_runs(cfg: dict):
    """Chosen runs of the quantum maps (QI4.3, QI6.1) and of the classical methods (QJ1, RBF-wide excluded) with the
    npz of their test scores: DataFrame variant, map, d, n, seed, run_id, test_ap, test_f1, npz."""
    import pandas as pd

    cols = ["variant", "map", "d", "n", "seed", "run_id", "test_ap", "test_f1"]
    q4 = pd.read_csv(qi4_paths(cfg)["runs"]).assign(npz=str(qi4_paths(cfg)["scores"]))
    q6 = pd.read_csv(stage2_paths(cfg, "qi6")["runs"])
    q6["npz"] = [str(stage2_paths(cfg, "qi6", v)["scores"]) for v in q6["variant"]]
    c = pd.read_csv(stage2_paths(cfg, "qj1")["runs"])
    c = c[c["map"] != "RBF-wide"].copy()
    c["npz"] = [str(stage2_paths(cfg, "qj1", v)["scores"]) for v in c["variant"]]
    q = pd.concat([q4, q6], ignore_index=True)
    q = q[q["role"] == "chosen"]
    return pd.concat([q[cols + ["npz"]], c[cols + ["npz"]]], ignore_index=True)


def qj3_statistics(cfg: dict, n_boot: int | None = None, save: bool = True) -> dict:
    """QJ3 (choices decided with the user).  For every unit (variant, d, n) that has quantum runs and the mirror QJ1
    runs (F-X grid, F-X learning curve, the other representations): per method the test AP and F1 [mean, std] over the
    seeds and the block-bootstrap 95 % CI of the seed-averaged AP (1,000 resamples of the test blocks of the
    candidates, the same resamples for every method: paired); per quantum map and classical reference of
    ``QJ3_REFERENCES``: the AP difference quantum - classical with its paired CI, the exact McNemar test on the errors
    at probability >= ``stage2.default_threshold`` per seed (same sample, same candidates), the median p over the seeds
    (< 0.05 = in at least 3 of 5 seeds), the plan rule (median p < 0.05 and neither CI contains the other method's
    mean) and the same rule with the Holm-adjusted median p (family = all comparisons with that reference)."""
    from qc_probes import block_bootstrap, bootstrap_ap, ci95

    n_boot = int(cfg["probes"]["bootstrap"]) if n_boot is None else n_boot
    thr = float(cfg["stage2"]["default_threshold"])
    runs = qj3_runs(cfg)
    qmaps = list(cfg["stage2"]["main"]["maps"])
    ms = lambda v: [round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else None]
    units, comps, checks, cache = {}, [], {}, {}
    npz = lambda p: cache.setdefault(p, np.load(p))
    for (v, d, n), g in runs.groupby(["variant", "d", "n"]):
        if not (set(qmaps) & set(g["map"])) or not (set(QJ3_REFERENCES) <= set(g["map"])):
            continue
        key = f"{v}_d{d}_n{n}"
        z0 = npz(g["npz"].iloc[0])
        y, ti, blocks = z0["test_cls"].astype(int), z0["test_index"], z0["test_block"]
        same = all(np.array_equal(npz(p)["test_index"], ti) and np.array_equal(npz(p)["test_cls"], z0["test_cls"])
                   for p in g["npz"].unique())
        wts = block_bootstrap(blocks, n_boot, int(cfg["seed"]))
        per, boot, err, seeds = {}, {}, {}, {}
        for m, gm in g.groupby("map"):
            gm = gm.sort_values("seed")
            sc = [npz(r.npz)[f"{r.run_id}__margin" if m in RANK_BY_MARGIN else f"{r.run_id}__proba"].astype(np.float64)
                  for r in gm.itertuples()]
            err[m] = [(npz(r.npz)[f"{r.run_id}__proba"] >= thr) != (y == 1) for r in gm.itertuples()]
            seeds[m] = list(gm["seed"])
            boot[m] = np.mean([bootstrap_ap(s, y, wts) for s in sc], axis=0)
            ap_np = [average_precision(s, y) for s in sc]
            checks.setdefault("max_abs_ap_diff_vs_runs_csv", 0.0)
            checks["max_abs_ap_diff_vs_runs_csv"] = max(checks["max_abs_ap_diff_vs_runs_csv"],
                                                        float(np.max(np.abs(np.array(ap_np) - gm["test_ap"].to_numpy()))))
            per[m] = {"test_ap": ms(gm["test_ap"]), "test_ap_ci95": ci95(boot[m]), "test_f1": ms(gm["test_f1"]),
                      "seeds": len(gm)}
        units[key] = {"variant": v, "d": int(d), "n": int(n), "candidates": int(y.size), "positives": int(y.sum()),
                      "test_blocks": int(np.unique(blocks).size), "same_candidates_all_files": bool(same), "methods": per}
        for qm in [m for m in qmaps if m in per]:
            for cm in QJ3_REFERENCES:
                if seeds[qm] != seeds[cm]:
                    raise ValueError(f"{key}: seeds of {qm} and {cm} differ")
                tests = [mcnemar_p(a, b) for a, b in zip(err[qm], err[cm])]
                p = [t[0] for t in tests]
                mq, mc = per[qm]["test_ap"][0], per[cm]["test_ap"][0]
                (lq, hq), (lc, hc) = per[qm]["test_ap_ci95"], per[cm]["test_ap_ci95"]
                comps.append({"unit": key, "quantum": qm, "classical": cm,
                              "ap_difference": round(mq - mc, 4), "ap_difference_ci95": ci95(boot[qm] - boot[cm]),
                              "fraction_resamples_quantum_better": round(float(np.mean(boot[qm] > boot[cm])), 3),
                              "errors_quantum_only_per_seed": [t[1] for t in tests],
                              "errors_classical_only_per_seed": [t[2] for t in tests],
                              "mcnemar_p_per_seed": [float(f"{x:.3g}") for x in p], "mcnemar_p_median": float(np.median(p)),
                              "seeds_p_below_005": int(np.sum(np.array(p) < 0.05)),
                              "ci_rule": bool(not (lq <= mc <= hq) and not (lc <= mq <= hc))})
    out_c = {}
    for cm in QJ3_REFERENCES:
        cs = [c for c in comps if c["classical"] == cm]
        adj = holm([c["mcnemar_p_median"] for c in cs])
        for c, a in zip(cs, adj):
            sig = c["mcnemar_p_median"] < 0.05 and c["ci_rule"]
            sig_h = bool(a < 0.05) and c["ci_rule"]
            c.update({"mcnemar_p_median": float(f"{c['mcnemar_p_median']:.3g}"), "mcnemar_p_median_holm": float(f"{a:.3g}"),
                      "significant": sig, "significant_holm": sig_h,
                      "better": (("quantum" if c["ap_difference"] > 0 else "classical") if sig else None)})
        diff = np.array([c["ap_difference"] for c in cs])
        out_c[cm] = {"comparisons": len(cs), "mcnemar_median_p_below_005": int(sum(c["mcnemar_p_median"] < 0.05 for c in cs)),
                     "ci_rule": int(sum(c["ci_rule"] for c in cs)), "significant": int(sum(c["significant"] for c in cs)),
                     "significant_holm": int(sum(c["significant_holm"] for c in cs)),
                     "significant_quantum_better": int(sum(c["better"] == "quantum" for c in cs)),
                     "significant_classical_better": int(sum(c["better"] == "classical" for c in cs)),
                     "paired_ci_excludes_zero": int(sum(not (c["ap_difference_ci95"][0] <= 0 <= c["ap_difference_ci95"][1]) for c in cs)),
                     "ap_difference_median_min_max": [round(float(np.median(diff)), 4), round(float(diff.min()), 4), round(float(diff.max()), 4)],
                     "rows": {f"{c['unit']}|{c['quantum']}": {k: val for k, val in c.items() if k not in ("unit", "quantum", "classical")}
                              for c in cs}}
    checks.update({"units": len(units), "comparisons": len(comps),
                   "same_candidates_all_units": all(u["same_candidates_all_files"] for u in units.values()),
                   "max_abs_ap_diff_vs_runs_csv": round(checks.get("max_abs_ap_diff_vs_runs_csv", 0.0), 6),
                   "seeds_per_method": sorted({m["seeds"] for u in units.values() for m in u["methods"].values()})})
    rep = {"bootstrap_resamples": n_boot, "bootstrap_seed": int(cfg["seed"]), "threshold": thr,
           "references": list(QJ3_REFERENCES), "units": units, "comparisons": out_c, "checks": checks,
           "note": ("test_ap [mean, std] over the seeds; test_ap_ci95 = block-bootstrap 95 % CI of the seed-averaged AP "
                    "(test blocks of the 2,000 candidates, same resamples for every method of a unit, so the differences "
                    "are paired; Platt score for the SVMs, logit for RFF / MLP / GBT); McNemar = exact test on the errors at "
                    "probability >= threshold, one per seed on the same sample and candidates; median p < 0.05 = p < 0.05 in "
                    "at least 3 of 5 seeds; significant = median p < 0.05 and neither CI contains the other method's mean "
                    "(plan rule); significant_holm = the same with the Holm-adjusted median p, family = all comparisons with "
                    "that classical reference; McNemar assumes independent candidates (spatial correlation ignored), the "
                    "block-bootstrap CI rule accounts for it")}
    if save:
        save_report(cfg, "QJ3", rep)
    return rep


# ---------------------------------------------------------------------------
# QJ4 - Costs: wall-clock time on the node, circuit and shot counts for hardware
# ---------------------------------------------------------------------------

QJ4_ENGINES = {"cpu": ("cpu", "aer"), "gpu": ("cuda", "aer"), "gpu_torch": ("cuda", "torch")}   # (device, state engine)


def qj4_units(cfg: dict) -> list[tuple[int, int]]:
    """(d, n) of the QJ4 benchmark on F-X: every d at the grid size (QI4.3 grid) and every n at the best d (curve)."""
    n0, bd = int(cfg["stage2"]["main"]["n"]), sorted(set(best_dims(cfg).values()))[0]
    return [(int(d), n0) for d in cfg["quantum"]["dims"]] + [(bd, int(n)) for n in cfg["qk_samples"]["sizes"] if int(n) != n0]


def quantum_fit_score(m: str, p: dict, a_tr: np.ndarray, y_tr: np.ndarray, a_te: np.ndarray, device: str,
                      engine: str) -> dict:
    """One QSVC of the configuration ``p`` (c, r, s, C) of map ``m``: exact training and test kernels (states by
    ``engine`` on ``device``, as QI3.1), ``fit_svc``, ``svc_scores``; returns the seconds of each phase and the
    scores.  QK-A uses its closed form (no state simulation) on every engine."""
    th_tr, th_te = feature_angles(a_tr, p["c"]), feature_angles(a_te, p["c"])
    r = 1 if p.get("r") is None or np.isnan(p["r"]) else int(p["r"])
    t0 = time.perf_counter()
    if m == "QK-P":
        e_tr = projected_features(th_tr, r, device, engine=engine)
        e_te = projected_features(th_te, r, device, engine=engine)
        g = float(p["s"]) * median_gamma(e_tr)
        k_tr, k_te = kernel_projected(e_tr, e_tr, g), kernel_projected(e_te, e_tr, g)
    else:
        k_tr, k_te = fidelity_kernels(m, th_tr, th_te, r, device, engine=engine)
    t1 = time.perf_counter()
    model = fit_svc(k_tr, y_tr, float(p["C"]))
    t2 = time.perf_counter()
    sc = svc_scores(model, k_te)
    t3 = time.perf_counter()
    return {"kernel_s": t1 - t0, "fit_s": t2 - t1, "score_s": t3 - t2, "total_s": t3 - t0, "proba": sc["proba"],
            "support_vectors": int(fitted_svc(model).n_support_.sum())}


def timed(fn, repeats: int, long_s: float = 60.0) -> tuple[list[dict], dict]:
    """``fn()`` (returns a dict with ``total_s``) repeated up to ``repeats`` times, stopping after a run longer than
    ``long_s``: (timings, last result)."""
    out = []
    for _ in range(repeats):
        res = fn()
        out.append({k: v for k, v in res.items() if k.endswith("_s")})
        if res["total_s"] > long_s:
            break
    return out, res


def hardware_counts(cfg: dict) -> dict:
    """Circuits and shots that the chosen QSVC runs of QI4.3 (F-X) would need on a QPU (no device data: depth, gates
    and QPU time are QK3).  Fidelity kernels (QK-A, QK-Z), one compute-uncompute circuit per pair: training
    n(n-1)/2 (diagonal = 1), inference candidates x support vectors (only the support vectors enter the decision
    function); QK-P, the local expectations of each state: 3 circuits per sample (all qubits measured in X, in Y, in
    Z), training 3n, inference 3 x candidates, independent of the support vectors.  Shots = circuits x
    ``diagnostics.shots_reference``.  [mean over the seeds] per map, d, n."""
    import pandas as pd

    shots = int(cfg["diagnostics"]["shots_reference"])
    m_te = int(cfg["test_subset"]["size"])
    q = pd.read_csv(qi4_paths(cfg)["runs"])
    q = q[(q["role"] == "chosen") & (q["variant"] == cfg["stage2"]["main"]["variant"])]
    out = {}
    for (m, d, n), g in q.groupby(["map", "d", "n"]):
        sv = float(g["support_vectors"].mean())
        tr, te = (3 * n, 3 * m_te) if m == "QK-P" else (n * (n - 1) // 2, m_te * sv)
        out[f"{m}_d{d}_n{n}"] = {"qubits": int(d), "support_vectors_mean": round(sv, 1), "training_circuits": int(tr),
                                 "inference_circuits": int(round(te)), "total_shots": float(f"{(tr + te) * shots:.3g}")}
    return {"shots_per_circuit": shots, "test_candidates": m_te, "runs": out}


def run_costs(cfg: dict, device: str = "cuda", repeats: int = 3, units=None, save: bool = True) -> dict:
    """QJ4 (decided with the user): wall-clock time of training + inference of every stage-2 method on the same node,
    for the chosen configuration of the first seed of every unit of ``qj4_units`` (F-X): the PCA (common to all), the
    classical methods of QJ1 on the CPUs (``classical_scores``: kernel / fit / scoring of the 2,000 candidates, as
    ``train_test_seconds`` of QJ1) and the QSVC with every engine of ``QJ4_ENGINES`` (Aer states on the CPUs with the
    kernel products on the CPUs or on the GPU = the pipeline of QI4, and the PyTorch simulator all on the GPU); one
    untimed warm-up per configuration, then up to ``repeats`` timed runs (one if it takes > 60 s), median.  No
    hyper-parameter search: its cost (the CV, tens of fits per unit) is reported from the logged unit times.  Checks:
    test AP of every timed run against the runs CSV.  Plus ``hardware_counts``.  Report section QJ4 (``units`` /
    ``save``: a subset without saving, for a smoke test)."""
    import os
    import platform

    import pandas as pd
    import torch

    main = cfg["stage2"]["main"]
    vid, seed = main["variant"], int(cfg["qk_samples"]["seeds"][0])
    engines = {k: v for k, v in QJ4_ENGINES.items() if torch.cuda.is_available() or v[0] == "cpu"}
    q = pd.read_csv(qi4_paths(cfg)["runs"])
    q = q[(q["role"] == "chosen") & (q["variant"] == vid) & (q["seed"] == seed)]
    c = pd.read_csv(stage2_paths(cfg, "qj1")["runs"])
    c = c[(c["variant"] == vid) & (c["seed"] == seed) & (c["map"] != "RBF-wide")]
    z = np.load(features_path(cfg))
    med = lambda ts, k: round(float(np.median([t[k] for t in ts])), 4)
    todo = units or qj4_units(cfg)
    units, ap_diff = {}, 0.0
    for d, n in todo:
        x_tr, i_tr, x_te, i_te = load_qk(cfg, vid, n, seed, z)
        y_tr, y_te = i_tr["cls"].astype(int), i_te["cls"].astype(int)
        t0 = time.perf_counter()
        a_tr, a_te = reduce_features(x_tr, d, x_te)
        pca_s = time.perf_counter() - t0
        res = {}
        for r in c[(c["d"] == d) & (c["n"] == n)].itertuples():
            p = json.loads(r.params)

            def fn(m=r.map, p=p):
                t = time.perf_counter()
                sc = classical_scores(cfg, m, p, a_tr, y_tr, a_te, seed)
                return {"total_s": time.perf_counter() - t, **sc}
            fn()                                                           # warm-up
            ts, sc = timed(fn, repeats)
            ap = average_precision(sc["margin"] if r.map in RANK_BY_MARGIN else sc["proba"], y_te)
            ap_diff = max(ap_diff, abs(ap - r.test_ap))
            res[r.map] = {"cpu": {"total_s": med(ts, "total_s"), "repeats": len(ts)}}
        for r in q[(q["d"] == d) & (q["n"] == n)].itertuples():
            p = {"c": r.c, "r": r.r, "s": r.s, "C": r.C}
            res[r.map] = {}
            for e, (dev, eng) in engines.items():
                fn = lambda m=r.map, p=p, dev=dev, eng=eng: quantum_fit_score(m, p, a_tr, y_tr, a_te, dev, eng)
                fn()                                                       # warm-up (Aer transpilation, CUDA context)
                ts, sc = timed(fn, repeats)
                ap_diff = max(ap_diff, abs(average_precision(sc["proba"], y_te) - r.test_ap))
                res[r.map][e] = {k: med(ts, k) for k in ("kernel_s", "fit_s", "score_s", "total_s")} | {"repeats": len(ts)}
                print(f"QJ4 d{d} n{n} {r.map} {e}: {res[r.map][e]['total_s']:.2f} s", flush=True)
        units[f"d{d}_n{n}"] = {"d": d, "n": n, "pca_s": round(pca_s, 4), "methods": res}
        print(f"QJ4 d{d} n{n}: " + ", ".join(f"{m} {v['cpu']['total_s']:.2f} s" for m, v in res.items()), flush=True)
    logged = {}                                                           # whole units with the CV search (QI4.3, QJ1)
    qa = pd.read_csv(qi4_paths(cfg)["runs"])
    qa = qa[(qa["role"] == "chosen") & (qa["variant"] == vid)]
    ca = pd.read_csv(stage2_paths(cfg, "qj1")["runs"])
    ca = ca[ca["variant"] == vid]
    for d, n in todo:
        logged[f"d{d}_n{n}"] = {"quantum_3_maps_gpu_job_s": round(float(qa[(qa["d"] == d) & (qa["n"] == n)].groupby("seed")["seconds"].first().median()), 1),
                                "classical_6_methods_cpu_job_s": round(float(ca[(ca["d"] == d) & (ca["n"] == n)].groupby("seed")["seconds"].first().median()), 1)}
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    rep = {"variant": vid, "seed": seed, "repeats_max": repeats, "engines": {k: list(v) for k, v in engines.items()},
           "node": {"cpu": platform.processor() or platform.machine(), "cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())),
                    "omp_threads": os.environ.get("OMP_NUM_THREADS"), "gpu": gpu},
           "units": units, "unit_seconds_with_cv_logged": logged, "hardware_counts": hardware_counts(cfg),
           "checks": {"max_abs_test_ap_diff_vs_runs_csv": round(ap_diff, 6), "units": len(units)},
           "note": ("seconds of refit + scoring of the 2,000 test candidates of the chosen configuration (seed of the "
                    "first sample), median of the timed repeats after one warm-up; pca_s = the PCA of QI1.1 (common to all "
                    "methods); classical methods on the CPUs (kernel, fit and scoring in total_s); QSVC: kernel_s = states "
                    "+ training and test kernels (QK-A: closed form, no simulation), fit_s = SVC + Platt, score_s = test "
                    "scores; engines: cpu = Aer states and products on the CPUs, gpu = Aer states on the CPUs and products on "
                    "the GPU (the QI4 pipeline), gpu_torch = PyTorch simulator and products on the GPU; "
                    "unit_seconds_with_cv_logged = the logged time of a whole unit of QI4.3 (3 maps, CV + refit, GPU job) "
                    "and QJ1 (6 methods, CV + refit, CPU job, parallel CV), median over the seeds, different conditions; "
                    "hardware_counts = circuits / shots a QPU would need, without device data (QK3)")}
    if save:
        save_report(cfg, "QJ4", rep)
    return rep


def selftest() -> None:
    """``fit_svc`` / ``svc_scores`` on a precomputed RBF kernel against ``SVC(kernel="rbf")``, the balanced
    class weights against explicit ones, ``stage2_kernels`` against the functions of ``qc_kernels``."""
    metrics_selftest()
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.svm import SVC

    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 4))
    y = (x[:, 0] + 0.5 * x[:, 1] ** 2 + 0.3 * rng.normal(size=120) > 0.6).astype(int)  # unbalanced
    xt = rng.normal(size=(40, 4))
    g = 0.3
    from sklearn.calibration import CalibratedClassifierCV

    m = fit_svc(rbf_kernel(x, x, gamma=g), y, C=2.0)
    ref = SVC(kernel="rbf", gamma=g, C=2.0, class_weight="balanced").fit(x, y)
    ref_p = CalibratedClassifierCV(SVC(kernel="rbf", gamma=g, C=2.0, class_weight="balanced"), method="sigmoid",
                                   ensemble=False, cv=5).fit(x, y)
    sc = svc_scores(m, rbf_kernel(xt, x, gamma=g))
    assert np.allclose(sc["margin"], ref.decision_function(xt), atol=1e-8)
    assert np.allclose(sc["proba"], ref_p.predict_proba(xt)[:, 1], atol=1e-8)
    assert np.array_equal(sc["pred"], ref.predict(xt))
    assert np.array_equal(np.argsort(sc["proba"]), np.argsort(sc["margin"]))     # monotone sigmoid
    w = {k: len(y) / (2 * np.sum(y == k)) for k in (0, 1)}
    ref_w = SVC(kernel="rbf", gamma=g, C=2.0, class_weight=w).fit(x, y)
    assert np.allclose(sc["margin"], ref_w.decision_function(xt), atol=1e-8)
    assert f1(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])) == 0.5
    a_tr, a_te = rng.normal(size=(6, 3)), rng.normal(size=(4, 3))
    k_tr, k_te = stage2_kernels("QK-A", a_tr, a_te, 0.5, device="cpu")
    assert np.allclose(k_te, kernel_qk_a(0.5 * a_te, 0.5 * a_tr)) and np.allclose(k_tr, kernel_qk_a(0.5 * a_tr, 0.5 * a_tr))
    e_tr, e_te = projected_features(0.5 * a_tr, 1, "cpu"), projected_features(0.5 * a_te, 1, "cpu")
    kp_tr, kp_te = stage2_kernels("QK-P", a_tr, a_te, 0.5, gamma_scale=2.0, device="cpu")
    assert np.allclose(kp_te, kernel_projected(e_te, e_tr, 2 * median_gamma(e_tr)))
    kn_tr, kn_te = stage2_kernels("QK-Z", a_tr, a_te, 0.5, shots=1024, rng=np.random.default_rng(3), device="cpu")
    assert kn_te.shape == (4, 6) and np.linalg.eigvalsh(kn_tr).min() > -1e-10
    th_tr, th_va = 0.5 * a_tr, 0.5 * a_te
    for r in (1, 2):
        kz, ez = zz_kernel_and_expectations(th_tr, th_va, r, "cpu", chunk=4)
        ref_k = fidelity_kernels("QK-Z", th_tr, th_va, r, "cpu")
        assert np.allclose(kz[0], ref_k[0], atol=1e-12) and np.allclose(kz[1], ref_k[1], atol=1e-12)
        assert np.allclose(ez[0], projected_features(th_tr, r, "cpu"), atol=1e-12)
        assert np.allclose(ez[1], projected_features(th_va, r, "cpu"), atol=1e-12)
    blocks = np.repeat(np.arange(24), 5)
    yb = rng.permutation(np.r_[np.ones(60, int), np.zeros(60, int)])
    folds = spatial_folds(yb, blocks, 5, 7)
    fs = fold_summary(folds, yb, blocks)
    assert fs["blocks_disjoint"] and fs["each_sample_validated_once"] and len(folds) == 5
    tab = [{"map": "QK-Z", "r": 2, "s": None, "c": 0.1, "C": 1.0, "cv_ap": 0.5}, {"map": "QK-Z", "r": 1, "s": None, "c": 0.2, "C": 1.0, "cv_ap": 0.5},
           {"map": "QK-Z", "r": 1, "s": None, "c": 0.1, "C": 10.0, "cv_ap": 0.5}, {"map": "QK-Z", "r": 1, "s": None, "c": 0.5, "C": 0.1, "cv_ap": 0.4}]
    assert select_best(tab, "QK-Z") == tab[1]                                       # tie -> smaller C, then r
    cfg_t = {"quantum": {"bandwidths": [0.2, 1.0], "zz_reps": [1], "projected_gamma_scales": [1.0]},
             "stage2": {"svm_C": [1.0], "cv_folds": 3}}
    xr = rng.normal(size=(120, 8))
    rows, _ = cv_search(xr, yb, blocks, 3, cfg_t, seed=1, device="cpu")
    assert len(rows) == 3 * 2 * 3 and all(0 <= r_["ap"] <= 1 for r_ in rows)     # folds x c x (A, Z, P)
    import tempfile

    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:                                    # QI4.3 unit, saving and resume
        cfg_u = {**cfg_t, "paths": {"output": Path(tmp), "results": Path(tmp)}, "seed": 0,
                 "stage2": {**cfg_t["stage2"], "main": {"variant": "T", "n": 120, "maps": ["QK-A", "QK-Z", "QK-P"], "per_c_dims": [3]}}}
        xt_, it_ = rng.normal(size=(30, 8)), {"cls": np.r_[np.ones(10, int), np.zeros(20, int)], "band": np.zeros(30, int)}
        run_unit(cfg_u, xr, {"cls": yb, "block": blocks}, xt_, it_, 3, 120, 1, ["QK-A", "QK-Z", "QK-P"], True, "cpu", np.arange(30))
        run_unit(cfg_u, xr, {"cls": yb, "block": blocks}, xt_, it_, 3, 120, 1, ["QK-A", "QK-Z", "QK-P"], True, "cpu", np.arange(30))
        runs_ = pd.read_csv(qi4_paths(cfg_u)["runs"])
        assert len(runs_) == 3 * (1 + 2) and runs_["run_id"].is_unique                # chosen + one per c, no duplicate on rerun
        assert len(pd.read_csv(qi4_paths(cfg_u)["cv"])) == 2 * 1 + 2 * 1 + 2 * 1      # c x configurations per map
        sz = np.load(qi4_paths(cfg_u)["scores"])
        ch = runs_[runs_["role"] == "chosen"].iloc[0]
        assert abs(average_precision(sz[ch["run_id"] + "__proba"].astype(float), it_["cls"]) - ch["test_ap"]) < 1e-4
        assert np.array_equal(sz["test_index"], np.arange(30))
        p6 = stage2_paths(cfg_u, "qi6", "V2")                                       # QI6.1: another variant, own files
        run_unit(cfg_u, xr, {"cls": yb, "block": blocks}, xt_, it_, 3, 120, 1, ["QK-A"], False, "cpu", np.arange(30),
                 variant="V2", step="QI6.1", paths=p6)
        r6 = pd.read_csv(p6["runs"])
        assert list(r6["run_id"]) == ["V2_QK-A_d3_n120_s1"] and (r6["step"] == "QI6.1").all() and (r6["variant"] == "V2").all()
        assert len(pd.read_csv(qi4_paths(cfg_u)["runs"])) == len(runs_)               # QI4.3 files untouched
        assert "V2_QK-A_d3_n120_s1__proba" in np.load(p6["scores"]) and p6["scores"].name == "qi6_V2_test_scores.npz"
        best = select_best(cv_table(cv_search(xr, yb, blocks, 3, cfg_t, seed=1, device="cpu")[0]), "QK-Z")
        assert ch["map"] != "QK-Z" or abs(best["cv_ap"] - ch["cv_ap"]) < 1e-4
    classical_selftest()
    print("qc_experiments selftest: OK")


def classical_selftest() -> None:
    """QJ1: the SVM baselines on precomputed kernels against scikit-learn's RBF / linear SVC, the grid sizes of the
    configuration (budget), the tie rule, every method on toy data, a unit saved twice (resume, no duplicates) with
    the AP of the saved scores equal to the runs CSV."""
    import tempfile

    import pandas as pd
    from sklearn.svm import SVC

    cfg = load_config()
    sizes = {m: len(classical_grid(cfg, m)) for m in cfg["classical"]["methods"]}
    assert sizes == {"RBF": 28, "LIN": 6, "RFF": 28, "MLP": 20, "GBT": 24, "RBF-wide": 40}, sizes
    rng = np.random.default_rng(1)
    x = rng.normal(size=(120, 5))
    y = (x[:, 0] + 0.5 * x[:, 1] ** 2 + 0.3 * rng.normal(size=120) > 0.6).astype(int)
    xt = rng.normal(size=(40, 5))
    g = 0.5 * median_gamma(x)
    sc = classical_scores(cfg, "RBF", {"C": 2.0, "s": 0.5}, x, y, xt, 0, calibrate=False)
    assert np.allclose(sc["margin"], SVC(kernel="rbf", gamma=g, C=2.0, class_weight="balanced").fit(x, y).decision_function(xt), atol=1e-7)
    sc = classical_scores(cfg, "LIN", {"C": 0.1}, x, y, xt, 0, calibrate=False)
    assert np.allclose(sc["margin"], SVC(kernel="linear", C=0.1, class_weight="balanced").fit(x, y).decision_function(xt), atol=1e-6)
    for m in ("RBF", "LIN", "RFF", "MLP", "GBT"):
        p = classical_grid(cfg, m)[0]
        a = classical_scores(cfg, m, p, x, y, xt, 3)
        b = classical_scores(cfg, m, p, x, y, xt, 3)
        assert a["proba"].shape == (40,) and np.all((a["proba"] >= 0) & (a["proba"] <= 1)) and np.allclose(a["proba"], b["proba"])
    blocks = np.repeat(np.arange(24), 5)
    yb = rng.permutation(np.r_[np.ones(60, int), np.zeros(60, int)])
    xr = rng.normal(size=(120, 8))
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-4, solver="lbfgs", max_iter=500, random_state=0).fit(x, y)
    assert np.allclose(1 / (1 + np.exp(-mlp_logit(mlp, xt))), mlp.predict_proba(xt)[:, 1], atol=1e-10)
    tab = [{"map": "M", "config": 3, "cv_ap": 0.5}, {"map": "M", "config": 1, "cv_ap": 0.5}, {"map": "M", "config": 0, "cv_ap": 0.4}]
    assert select_classical(tab, "M")["config"] == 1                               # tie -> first configuration
    small = {**cfg, "classical": {**cfg["classical"], "rbf_gamma_scales": [0.5, 1], "linear_C": [1],
                                  "mlp": {"hidden": [4], "alpha": [0.1], "max_iter": 200},
                                  "gbt": {"max_iter": [20], "max_depth": [2], "learning_rate": [0.1]},
                                  "rff": {"components": 50, "C": [1]}},
             "stage2": {**cfg["stage2"], "svm_C": [1.0], "cv_folds": 3}}
    methods = ["RBF", "LIN", "RFF", "MLP", "GBT"]
    one = classical_cv(small, xr, yb, blocks, 3, 5, methods, n_jobs=1)
    two = classical_cv(small, xr, yb, blocks, 3, 5, methods, n_jobs=2)
    assert [r_["ap"] for r_ in one] == [r_["ap"] for r_ in two]                    # independent of n_jobs
    with tempfile.TemporaryDirectory() as tmp:
        small["paths"] = {**cfg["paths"], "output": Path(tmp), "results": Path(tmp)}
        xt_ = rng.normal(size=(30, 8))
        it_ = {"cls": np.r_[np.ones(10, int), np.zeros(20, int)], "band": np.zeros(30, int)}
        p = stage2_paths(small, "qj1", "T")
        for _ in range(2):
            run_classical_unit(small, xr, {"cls": yb, "block": blocks}, xt_, it_, 3, 120, 1, "T", np.arange(30), methods, p)
        run_classical_unit(small, xr, {"cls": yb, "block": blocks}, xt_, it_, 3, 120, 1, "T", np.arange(30), ["MLP"], p)
        runs_ = pd.read_csv(p["runs"])                                             # one method redone alone
        assert len(runs_) == 5 and runs_["run_id"].is_unique and list(runs_.columns) == QJ_COLS
        assert len(pd.read_csv(p["cv"])) == 2 + 1 + 2 + 1 + 1                      # configurations per method, all kept
        z = np.load(p["scores"])
        for r_ in runs_.itertuples():
            key = "margin" if r_.map in RANK_BY_MARGIN else "proba"
            assert abs(average_precision(z[f"{r_.run_id}__{key}"].astype(float), it_["cls"]) - r_.test_ap) < 1e-4
        pf = stage2_paths(small, "qj2", "T")                                     # QJ2: LIN-full, no PCA
        run_classical_unit(small, xr, {"cls": yb, "block": blocks}, xt_, it_, None, 120, 1, "T", np.arange(30), ["LIN-full"],
                           pf, step="QJ2")
        rf = pd.read_csv(pf["runs"])
        assert list(rf["run_id"]) == ["T_LIN-full_d8_n120_s1"] and (rf["step"] == "QJ2").all()
    a_s, b_s = standardize(xr, xt_)
    assert np.allclose(a_s.mean(0), 0) and np.allclose(a_s.std(0, ddof=1), 1)
    a_r = standardize(xr, ref_dim=2)[0]
    assert np.isclose(np.mean(np.sum(a_r ** 2, 1)), 2 * 119 / 120)                 # mean squared norm = ref_dim
    full = classical_scores(cfg, "LIN-full", {"C": 0.1}, a_s, yb, b_s, 0, calibrate=False)
    assert np.allclose(full["margin"], SVC(kernel="linear", C=0.1, class_weight="balanced").fit(a_s, yb).decision_function(b_s), atol=1e-6)
    print("qc_experiments classical selftest: OK")


def metrics_selftest() -> None:
    """QI4.4 helpers on hand-made cases (seconds; also run by ``selftest``)."""
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    band = np.array([0, 0, 1, 0, 0, 0, 0, 2])
    pred = np.array([1, 0, 1, 1, 0, 0, 0, 0])
    m = threshold_metrics(pred, y, band)
    assert m["precision"] == 2 / 3 and m["recall"] == 0.5 and m["predicted_share"] == 3 / 8
    assert m["recall_1_2_km"] == 0.5 and m["recall_2_4_km"] == 1.0 and m["recall_4_8_km"] == 0.0
    assert m["precision_2_4_km"] == 0.5 and m["precision_4_8_km"] == 0.0
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    assert precision_at_recall(score, np.array([1, 0, 1, 0, 1, 0, 0, 1]), 0.5) == (2 / 3, 3 / 8)
    assert precision_at_recall(score, np.array([1, 0, 1, 0, 1, 0, 0, 1]), 1.0) == (0.5, 1.0)
    assert precision_at_recall(score, np.array([1, 0, 1, 0, 1, 0, 0, 1]), 0.01) == (1.0, 1 / 8)
    ea = np.array([1] * 8 + [0] * 2 + [1] * 5 + [0] * 5, bool)          # QJ3: b = 8 errors of a only, c = 2 of b only
    eb = np.array([0] * 8 + [1] * 2 + [1] * 5 + [0] * 5, bool)
    assert mcnemar_p(ea, eb) == (2 * 56 / 1024, 8, 2) and mcnemar_p(ea, ea)[0] == 1.0
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    print("qc_experiments metrics selftest: OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["svm-check", "cv-check", "grid", "curve", "summary", "metrics", "representations",
                                                           "representations-summary", "baselines", "baselines-summary",
                                                           "references", "references-summary", "statistics", "costs"])
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--methods", nargs="+", default=None, help="baselines: subset of classical.methods")
    ap.add_argument("--redo", action="store_true", help="baselines: rerun the --methods also where already done")
    ap.add_argument("--redo-before", default=None, help="baselines: redo only the runs written before this ISO time")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if args.command is None:
        ap.error("a command or --selftest is required")
    cfg = load_config(args.config)
    if args.command == "svm-check":
        print(json.dumps(run_svm_check(cfg, args.device)["checks"], indent=1))
    elif args.command == "cv-check":
        print(json.dumps(run_cv_check(cfg, args.device)["folds"], indent=1))
    elif args.command == "grid":
        run_grid(cfg, args.device)
    elif args.command == "curve":
        run_curve(cfg, args.device)
    elif args.command == "summary":
        print(json.dumps(qi4_summary(cfg).get("best_d"), indent=1))
    elif args.command in ("representations", "representations-summary"):
        if args.command == "representations":
            run_representations(cfg, args.device)
        r = qi6_summary(cfg)
        print(json.dumps({v: {k: (e["ap"], e["f1"], e["precision_gain_at_equal_recall"]) for k, e in d.items() if "_d" in k}
                          for v, d in r["variants"].items()}, indent=1))
        print(json.dumps({v: d["stage1_alone_qh2"]["subset_ap"]["all"] for v, d in r["variants"].items()}, indent=1))
        print(json.dumps(r["checks"], indent=1))
    elif args.command in ("baselines", "baselines-summary"):
        if args.command == "baselines":
            run_baselines(cfg, args.methods, args.redo, args.redo_before)
        r = qj1_summary(cfg)
        print(json.dumps({k: v["test_ap"] for k, v in r["grid"].items()}, indent=1))
        print(json.dumps({v: {k: (e["ap"], e["f1"]) for k, e in d.items() if "_d" in k and k != "quantum_test_ap"}
                          for v, d in r["representations"].items()}, indent=1))
        print(json.dumps({"rbf_wide_sensitivity": r["rbf_wide_sensitivity"], "checks": r["checks"]}, indent=1))
    elif args.command in ("references", "references-summary"):
        if args.command == "references":
            run_references(cfg)
        r = qj2_summary(cfg)
        print(json.dumps({v: {"stage1": e["stage1_alone_ap"], "probe": e["qg1_probe_full_all_s1train"]["ap"],
                              "lin_full": e["lin_full_qk_sample"]["ap"], **{k: e[k] for k in e if k.startswith("lin_pca")},
                              "gap": e["gap_decomposition_ap"]} for v, e in r["representations"].items()}, indent=1))
        print(json.dumps({"fx_learning_curve": r["fx_learning_curve"], "checks": r["checks"]}, indent=1))
    elif args.command == "costs":
        r = run_costs(cfg, args.device)
        print(json.dumps({u: {m: {e: t["total_s"] for e, t in v.items()} for m, v in e_["methods"].items()}
                          for u, e_ in r["units"].items()}, indent=1))
        print(json.dumps(r["checks"], indent=1))
    elif args.command == "statistics":
        r = qj3_statistics(cfg)
        print(json.dumps({k: {m: e[m] for m in e if m != "rows"} for k, e in r["comparisons"].items()}, indent=1))
        print(json.dumps(r["checks"], indent=1))
    elif args.command == "metrics":
        r = run_metrics(cfg)
        main_n = f"n{cfg['stage2']['main']['n']}"
        keys = ["ap", "f1", "precision", "recall", "e2e_recall", "e2e_precision", "stage1_precision_at_e2e_recall",
                "precision_gain_at_equal_recall"]
        print(json.dumps({"stage1_subset_ap": r["stage1_alone_qh2"]["subset_ap"], "checks": r["checks"],
                          **{k: {m: v[m] for m in keys} for k, v in ((k, v[main_n]) for k, v in r["stage2"].items())}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
