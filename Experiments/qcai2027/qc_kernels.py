#!/usr/bin/env python3
"""Quantum-kernel stage of the QCAI 2027 experiment (plan phase QI): inputs, reduction and scaling.

The quantum stage classifies the stage-1 candidates (QH3) of each representation (QF1 variant) from its
sub-tile vectors.  It is trained on the QK training samples (QH4) and tested on the fixed 2,000 test
candidates (QH5).

QK vectors.  ``run_features``: per variant the vectors of the union of all its QK training samples (indices
into the QK-pool candidates) and of its 2,000 test candidates, read once from the token / fused-map caches
(14 GB) and kept in ``Dataset/Moon/qcai2027/qk/qk_features.npz`` (float16 as the caches, exact).
``load_qk`` returns the vectors, labels and strata of one training sample and of the test candidates.

QI1.1 - Reduction and scaling.  ``reduce_features``: PCA of the centred training features (covariance, QD6,
the same as QG3 via ``qc_probes.pca_fit``), fitted on the QK training sample only; the first d components
(d = qubits, ``quantum.dims``) standardised with the mean / std of the training sample; the same transform
applied to the other sets (test, or the held-out fold in the cross-validation of QI4.2, where it is refitted
on every training fold).  Computed on the fly (milliseconds), not stored.  ``run_pca`` reports per variant,
sample size and d the explained variance (mean / std over the 5 seeds, against the S1-train PCA of QG3), the
stability of the component subspace across seeds and the std of the standardised test components.

QI1.2 - Bandwidth.  ``feature_angles``: every standardised component x_i becomes the angle c * x_i, with c in
``quantum.bandwidths``; the feature maps of QI2 use it as written in the plan (QK-A: R_Y(c x_i); QK-Z: R_Z(2 c x_i)
and R_ZZ(2 c^2 x_i x_j)).  The single-qubit similarity of two samples is cos^2(c dx_i / 2) for R_Y and
cos^2(c dx_i) for the phase of QK-Z: it decreases with |dx_i| only while |c dx_i| <= pi (R_Y) or <= pi/2 (R_Z),
beyond that two far samples look similar again (aliasing).  ``run_bandwidth`` reports per variant and c, on the
QK samples of the main grid size, the quantiles of |c x| on the test candidates and the share of the
training-test component pairs beyond the two limits (``aliased_fraction``).

QI2 - Feature maps.  ``feature_map(name, d)``: Qiskit circuit of d qubits whose parameters are the angles
theta = c x of QI1.2 (so one circuit serves every c), used for the statevectors of QI3, the Aer check of QI3.2
and the noise model of QK1.  QI2.1 QK-A (``qk_a``): R_Y(theta_i) on each qubit, no entanglement; its fidelity
kernel has the closed form prod_i cos^2((theta_i - theta'_i) / 2) (``kernel_qk_a``), a classical sanity
baseline.  ``run_maps`` checks the closed form against the Qiskit statevectors on real angles.
QI2.2 QK-Z (``qk_z``, Havlicek et al. 2019): r repetitions of [H on every qubit, R_Z(2 theta_i), R_ZZ(2 theta_i
theta_{i+1}) on the linear chain] (theta_i theta_{i+1} = c^2 x_i x_{i+1}); ``qk_z_qiskit`` = the same with the Qiskit
default pair function (pi - theta_i)(pi - theta_{i+1}), for the comparison with the literature.  ``run_zz`` checks
both against ``qiskit.circuit.library.zz_feature_map`` (linear entanglement; P and CX-P-CX instead of R_Z and
R_ZZ: equal up to a global phase, so the fidelities are equal) and reports the kernel values on real angles.
QI2.3 QK-P (projected quantum kernel, Huang et al. 2021): from the QK-Z state (product pair function, r of
``quantum.zz_reps``) the expectations <X>, <Y>, <Z> of every qubit (``local_expectations``, 3d numbers, the
single-qubit reduced states), then the Gaussian kernel exp(-gamma ||m(x) - m(x')||^2) (``kernel_projected``) with
gamma = s / median of the squared distances of the training sample (``median_gamma``), s in
``quantum.projected_gamma_scales`` (chosen by cross-validation, QI4.2).  ``run_projected`` checks the expectations
against Qiskit and compares the projected and the fidelity kernel on the same real pairs.

QI3.1 - Exact kernels.  The states are computed **once** per sample with the standard Qiskit Aer simulator
(``AerSimulator(method="statevector")``, batches of bound, transpiled feature-map circuits with
``save_statevector``; ``state_chunks``), moved in chunks to the GPU, and all the pairs come from matrix products
(``fidelity_kernels``): K_train = |Psi_tr Psi_tr^+|^2, K_test = |Psi_te Psi_tr^+|^2 (element-wise), the training
states kept on the GPU (2,000 states of 20 qubits = 32 GB in complex128), the test states in chunks.  QK-A by its
closed form (``kernel_qk_a``, equal to the statevectors, QI2.1); QK-P: the 3d expectations of the same Aer states
(``projected_features``).  States and products in complex128 (complex64 products: errors up to 2e-4 at d = 20).
A dedicated PyTorch simulator of the QK-Z family (``zz_states``: r x [fast Walsh-Hadamard transform, diagonal
phases]) is kept only as an independent cross-check (``engine="torch"``, selftest and ``run_exact``); the
Qiskit ``Statevector`` path is too slow at d = 20 (``run_simtime``).  ``run_exact`` reports the agreement of the
Aer states with Qiskit ``Statevector`` and with the PyTorch states, the precision, the properties of the kernels
and the time at n = 2,000.

QI3.2 - Verification.  ``compute_uncompute``: the circuit U(x) followed by U(x')^+ (two parameter vectors, transpiled
once for Aer), whose probability of |0...0> is the fidelity |<psi(x')|psi(x)>|^2.  ``run_verify``: for
``quantum.verify_pairs`` random training-test pairs of real F-X angles, every fidelity feature map, d, r and c, the
value of ``fidelity_kernels`` (QI3.1) against that probability, exact (Aer statevector of the circuit) and estimated
with N shots (``quantum.shots``, measurement of all qubits): exact difference < 1e-6 required, shot estimates within
4 binomial standard deviations sqrt(k (1 - k) / N) (+ 1/N).

QI3.3 - Shot noise.  Hardware estimates every value from N measurements, simulated without circuits (QI3.2 shows
the binomial model holds): fidelity kernels (``shot_kernel``) K_hat = Binomial(N, K) / N for every training pair
(upper triangle, once per pair, mirrored; diagonal = 1, known on hardware) and every test-training pair; QK-P
(``shot_expectations``, extension of the plan decided with the user) <P>_hat = 2 Binomial(N, (1 + <P>) / 2) / N - 1
for the X, Y, Z expectations of every qubit, then the Gaussian kernel with the median gamma of the noisy training
expectations.  ``psd_project``: the training matrix projected on the nearest positive semi-definite matrix
(eigenvalues < 0 set to 0), needed by the SVM; the test matrix is used as estimated.  N in ``quantum.shots``
(and infinity = exact).  ``run_shots`` reports the error of the noisy kernels against the exact ones.

Commands
--------
features     QK vectors -> ``Dataset/Moon/qcai2027/qk/qk_features.npz``; prints the counts and checks
             (finite, labels equal to QH3, test candidates equal to QH5).  Reads the caches:
             a few minutes (CPU job: run_qcai_cpu.sh).

pca          QI1.1 -> report section QI1.1 of ``results/qi_kernels.json``.  Seconds.

bandwidth    QI1.2 -> report section QI1.2 of ``results/qi_kernels.json``.  Seconds.

maps         QI2.1 -> report section QI2.1: per d the QK-A circuit (gates, depth, two-qubit gates) and
             the largest difference between the closed-form kernel and the fidelity of the Qiskit
             statevectors on real F-X angles of every c.  ~1 min with many threads (d = 20
             statevectors): CPU job (run_qcai_cpu.sh).
             ``--map QK-Z`` QI2.2 -> report section QI2.2: per d and r the QK-Z circuits (ours and
             the Qiskit pair function), their fidelities against ``zz_feature_map``, kernel values
             on real F-X angles of every c, angles of the pair terms.  ~20 min: CPU job.
             ``--map QK-P`` QI2.3 -> report section QI2.3: per d and r the expectations against
             Qiskit, the Bloch-vector lengths, the median gamma and the projected kernel against the
             fidelity kernel of the same QK-Z states on real F-X pairs, every c.  ~15 min: CPU job.

exact        QI3.1 -> report section QI3.1: Aer states against Qiskit Statevector and PyTorch (every
             d, r, pair function), complex64 against complex128 products, kernels of a 2,000
             training sample x 2,000 test candidates (diagonal, symmetry, smallest eigenvalue,
             time) per d and r.  GPU job (run_qcai_gpu.sh, ``--device cuda``; Aer on the CPUs).

verify       QI3.2 -> report section QI3.2: per feature map, d and r, largest |K - P(0...0)| of the
             compute-uncompute circuit (exact) and shot estimates against the binomial error, for
             every c.  ~10 min: CPU job (run_qcai_cpu.sh).

shots        QI3.3 -> report section QI3.3: per map, d, r, c and N the relative Frobenius error of
             the noisy training (after the PSD projection) and test kernels, share of training
             values estimated as 0, eigenvalues clipped.  GPU job (run_qcai_gpu.sh, ~25 min).

simtime      QI3.1 -> report section QI3.1-qiskit: time per state of the standard Qiskit
             simulators (``Statevector``; ``AerSimulator`` statevector method, batches of bound
             circuits) on real F-X angles, per d and r, and the extrapolated time of the QI3.1
             check and of the QI4.3 main grid with them.  CPU job (run_qcai_cpu.sh).

Usage
-----
    python Experiments/qcai2027/qc_kernels.py features     (CPU job: run_qcai_cpu.sh)
    python Experiments/qcai2027/qc_kernels.py pca
    python Experiments/qcai2027/qc_kernels.py bandwidth
    python Experiments/qcai2027/qc_kernels.py maps
    python Experiments/qcai2027/qc_kernels.py maps --map QK-Z
    python Experiments/qcai2027/qc_kernels.py maps --map QK-P
    python Experiments/qcai2027/qc_kernels.py exact --device cuda     (GPU job: run_qcai_gpu.sh)
    python Experiments/qcai2027/qc_kernels.py verify                   (CPU job: run_qcai_cpu.sh)
    python Experiments/qcai2027/qc_kernels.py shots --device cuda      (GPU job: run_qcai_gpu.sh)
    python Experiments/qcai2027/qc_kernels.py simtime                  (CPU job: run_qcai_cpu.sh)
    python Experiments/qcai2027/qc_kernels.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_data import CONFIG_PATH, load_config, load_report, save_report, written  # noqa: E402
from qc_fusion import VARIANTS  # noqa: E402

STRATA = ("cls", "band", "block", "tercile")                       # per-sample info kept with the vectors


def qk_dir(cfg: dict) -> Path:
    """Inputs of the quantum stage: ``<output>/qk/``."""
    return cfg["paths"]["output"] / "qk"


def features_path(cfg: dict) -> Path:
    return qk_dir(cfg) / "qk_features.npz"


# ---------------------------------------------------------------------------
# QK vectors
# ---------------------------------------------------------------------------


def candidate_vectors(store, vid: str, idx: dict, sel: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Vectors (len(sel), D) float16 of variant ``vid`` for the owned sub-tiles ``sel`` (indices into the
    arrays ``idx`` of ``load_candidates``), read tile by tile from the caches."""
    from qc_fusion import variant_maps

    rows, inv = np.unique(idx["row"][sel], return_inverse=True)
    x = np.empty((sel.size, VARIANTS[vid].dim), dtype=np.float16)
    for s in range(0, rows.size, chunk):
        m = variant_maps(store, vid, rows[s:s + chunk])
        k = np.flatnonzero((inv >= s) & (inv < s + chunk))
        x[k] = m[inv[k] - s, idx["i"][sel[k]], idx["j"][sel[k]]]
    return x


def run_features(cfg: dict) -> dict:
    """Vectors of the union of the QK training samples (QH4) and of the test candidates (QH5) of every
    variant, with their QK-pool / Test indices and strata, in ``qk/qk_features.npz``
    (keys ``<variant>_<qk|test>_<index|x|cls|band|block|tercile>``)."""
    from qc_fusion import TokenStore
    from qc_stage1 import load_candidates, stage1_dir

    store = TokenStore(cfg)
    samples = np.load(stage1_dir(cfg) / "qk_samples.npz")
    test_sub = np.load(stage1_dir(cfg) / "test_subset.npz")
    arrays, out = {}, {}
    for vid in VARIANTS:
        out[vid] = {}
        for part, set_name, sel in (("qk", "qk_pool", np.unique(np.concatenate([samples[k] for k in samples.files
                                                                                 if k.startswith(f"{vid}_n")]))),
                                    ("test", "test", test_sub[vid])):
            _, cand, idx = load_candidates(cfg, set_name, vid)
            x = candidate_vectors(store, vid, idx, sel)
            arrays[f"{vid}_{part}_index"] = sel.astype(np.int32)
            arrays[f"{vid}_{part}_x"] = x
            for k in STRATA:
                arrays[f"{vid}_{part}_{k}"] = idx[k][sel]
            out[vid][part] = {"samples": int(sel.size), "dim": int(x.shape[1]),
                              "positives": int((idx["cls"][sel] == 1).sum()),
                              "checks": {"finite": bool(np.isfinite(x).all()), "candidates_only": bool(cand[sel].all()),
                                         "labelled": bool((idx["cls"][sel] >= 0).all())}}
        print(f"QK vectors {vid}: {out[vid]['qk']['samples']} QK-pool + {out[vid]['test']['samples']} test, "
              f"checks {all(c for p in out[vid].values() for c in p['checks'].values())}", flush=True)
    qk_dir(cfg).mkdir(parents=True, exist_ok=True)
    np.savez(features_path(cfg), **arrays)
    written(features_path(cfg))
    return out


def load_qk(cfg: dict, vid: str, n: int, seed: int, z=None) -> tuple[np.ndarray, dict, np.ndarray, dict]:
    """Training sample (``<vid>_n<n>_s<seed>`` of QH4) and test candidates (QH5) of variant ``vid``:
    (x_train float32, strata {cls, band, block, tercile}, x_test, strata).  ``z``: the opened
    ``qk_features.npz`` (to avoid re-opening it in a loop)."""
    from qc_stage1 import stage1_dir

    z = np.load(features_path(cfg)) if z is None else z
    sel = np.load(stage1_dir(cfg) / "qk_samples.npz")[f"{vid}_n{n}_s{seed}"]
    pos = np.searchsorted(z[f"{vid}_qk_index"], sel)
    if not np.array_equal(z[f"{vid}_qk_index"][pos], sel):
        raise ValueError(f"{vid}_n{n}_s{seed} not in qk_features.npz: run qc_kernels.py features")
    tr = {k: z[f"{vid}_qk_{k}"][pos] for k in STRATA}
    te = {k: z[f"{vid}_test_{k}"] for k in STRATA}
    return z[f"{vid}_qk_x"][pos].astype(np.float32), tr, z[f"{vid}_test_x"].astype(np.float32), te


# ---------------------------------------------------------------------------
# QI1.1 - PCA and standardisation
# ---------------------------------------------------------------------------


def fit_reduction(x_tr: np.ndarray, d: int) -> dict:
    """QI1.1 transform fitted on ``x_tr`` (n, D): PCA of the centred features (``qc_probes.pca_fit``,
    float64), first ``d`` components, standardised with their training mean / std (ddof 1).
    Returns {mean, components (D, d), comp_mean, comp_std, explained (d,)}."""
    import torch

    from qc_probes import pca_fit

    xt = torch.from_numpy(np.asarray(x_tr, dtype=np.float64))
    mu, vec, frac = pca_fit(xt, d)
    p = (xt - mu) @ vec
    return {"mean": mu.numpy(), "components": vec.numpy(), "comp_mean": p.mean(0).numpy(),
            "comp_std": p.std(0).clamp(min=1e-12).numpy(), "explained": frac}


def apply_reduction(red: dict, x: np.ndarray) -> np.ndarray:
    """Standardised components (n, d) float64 of ``x`` with a transform of ``fit_reduction``."""
    return ((np.asarray(x, dtype=np.float64) - red["mean"]) @ red["components"] - red["comp_mean"]) / red["comp_std"]


def reduce_features(x_tr: np.ndarray, d: int, *others: np.ndarray) -> list[np.ndarray]:
    """QI1.1: the d standardised principal components of ``x_tr`` and of every array of ``others``, with the
    transform fitted on ``x_tr`` only (the input of the feature maps of QI2, one component per qubit)."""
    red = fit_reduction(x_tr, d)
    return [apply_reduction(red, x) for x in (x_tr, *others)]


def subspace_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap of the spans of two orthonormal bases (D, d): ||A^T B||_F^2 / d, 1 = same subspace, ~d/D at
    random."""
    return float(np.sum((a.T @ b) ** 2) / a.shape[1])


def run_pca(cfg: dict) -> dict:
    """QI1.1 report: per variant, sample size n and d, the explained variance of the first d components
    (mean / std over the seeds; S1-train reference of QG3), the mean overlap of the component subspaces of
    two seeds, the std of the standardised test components (1 = the training scale holds on the test
    candidates) and the checks of the standardisation."""
    q = cfg["qk_samples"]
    sizes, seeds = [int(v) for v in q["sizes"]], [int(v) for v in q["seeds"]]
    dims = [int(d) for d in cfg["quantum"]["dims"]]
    z = np.load(features_path(cfg))
    qg3 = load_report(cfg, "QG3")["variants"]
    out, ok = {}, {"train_mean_0": True, "train_std_1": True, "finite": True, "rank_ge_d": True}
    for vid in VARIANTS:
        r = {}
        for n in sizes:
            fits, test_sd = [], []
            for seed in seeds:
                x_tr, _, x_te, _ = load_qk(cfg, vid, n, seed, z)
                red = fit_reduction(x_tr, max(dims))
                a_tr, a_te = apply_reduction(red, x_tr), apply_reduction(red, x_te)
                ok["train_mean_0"] &= bool(np.abs(a_tr.mean(0)).max() < 1e-8)
                ok["train_std_1"] &= bool(np.abs(a_tr.std(0, ddof=1) - 1).max() < 1e-8)
                ok["finite"] &= bool(np.isfinite(a_te).all())
                ok["rank_ge_d"] &= bool(np.linalg.matrix_rank(x_tr - x_tr.mean(0)) >= max(dims))
                fits.append(red)
                test_sd.append(a_te.std(0, ddof=1))
            ev = np.array([np.cumsum(f["explained"]) for f in fits])          # (seeds, d_max)
            sd_te = np.array(test_sd)
            r[f"n{n}"] = {f"d{d}": {"explained_variance": [round(float(ev[:, d - 1].mean()), 4), round(float(ev[:, d - 1].std()), 4)],
                                    "subspace_overlap_between_seeds": round(float(np.mean(
                                        [subspace_overlap(fits[a]["components"][:, :d], fits[b]["components"][:, :d])
                                         for a in range(len(seeds)) for b in range(a + 1, len(seeds))])), 3),
                                    "test_component_std_range": [round(float(sd_te[:, :d].min()), 3), round(float(sd_te[:, :d].max()), 3)]}
                          for d in dims}
        r["explained_variance_s1_train_qg3"] = {f"d{d}": qg3[vid][f"d{d}"]["explained_variance_s1_train"] for d in dims
                                                if f"d{d}" in qg3.get(vid, {})}
        out[vid] = r
        e = r[f"n{sizes[-1]}"][f"d{dims[-1]}"]
        print(f"QI1.1 {vid}: n {sizes[-1]} d {dims[-1]} explained {e['explained_variance'][0]:.3f}, "
              f"seed overlap {e['subspace_overlap_between_seeds']:.3f}", flush=True)
    report = {"variants": out, "dims": dims, "sizes": sizes, "seeds": seeds, "checks": ok,
              "note": "PCA of the centred features (QD6) fitted on each QK training sample (QH4), first d components "
                      "standardised on the sample and applied to the 2,000 test candidates (QH5); explained_variance = "
                      "[mean, std] over the seeds of the cumulative fraction of the first d; subspace overlap = "
                      "||U_a^T U_b||_F^2 / d averaged over the seed pairs; test_component_std_range = [min, max] over seeds "
                      "and components of the std of the standardised test components; transform computed on the fly"}
    save_report(cfg, "QI1.1", report)
    return report


# ---------------------------------------------------------------------------
# QI1.2 - Bandwidth
# ---------------------------------------------------------------------------


def feature_angles(x: np.ndarray, c: float) -> np.ndarray:
    """QI1.2: the angles c * x of the standardised components ``x`` (n, d), float64 (input of the feature maps)."""
    return float(c) * np.asarray(x, dtype=np.float64)


def aliased_fraction(a: np.ndarray, b: np.ndarray, limit: float) -> float:
    """Share of the pairs (row of ``a``, row of ``b``) and components i with |a_i - b_i| > ``limit`` (angles),
    counted per component by sorting (no n x m matrix)."""
    tot = 0
    for i in range(a.shape[1]):
        s = np.sort(b[:, i])
        tot += np.searchsorted(s, a[:, i] - limit, "left").sum() + (s.size - np.searchsorted(s, a[:, i] + limit, "right")).sum()
    return float(tot / (a.shape[0] * b.shape[0] * a.shape[1]))


def run_bandwidth(cfg: dict) -> dict:
    """QI1.2 report: per variant and c, on the samples of size ``quantum.bandwidth_check_size`` (5 seeds, the
    largest d), the 50 / 99 % quantiles and the maximum of |c x| on the test candidates and the share of the
    training-test component pairs with |c dx| > pi (non-monotone single-qubit similarity of R_Y, QK-A) and
    > pi / 2 (of the phase R_Z(2 c x) of QK-Z), mean over the seeds."""
    cs = [float(c) for c in cfg["quantum"]["bandwidths"]]
    n, d = int(cfg["quantum"]["bandwidth_check_size"]), max(int(v) for v in cfg["quantum"]["dims"])
    seeds = [int(v) for v in cfg["qk_samples"]["seeds"]]
    z = np.load(features_path(cfg))
    out = {}
    for vid in VARIANTS:
        pairs = []
        for s in seeds:
            x_tr, _, x_te, _ = load_qk(cfg, vid, n, s, z)
            pairs.append(reduce_features(x_tr, d, x_te))
        r = {}
        for c in cs:
            a_te = np.abs(np.concatenate([feature_angles(p[1], c) for p in pairs]))
            r[f"c{c:g}"] = {"abs_angle_test_q50_q99_max": [round(float(v), 3) for v in (*np.quantile(a_te, [0.5, 0.99]), a_te.max())],
                            "aliased_fraction_ry_pi": float(f"{np.mean([aliased_fraction(feature_angles(p[0], c), feature_angles(p[1], c), np.pi) for p in pairs]):.2e}"),
                            "aliased_fraction_rz_half_pi": float(f"{np.mean([aliased_fraction(feature_angles(p[0], c), feature_angles(p[1], c), np.pi / 2) for p in pairs]):.2e}")}
        out[vid] = r
        print(f"QI1.2 {vid}: " + ", ".join(f"c {c:g} aliased RY {r[f'c{c:g}']['aliased_fraction_ry_pi']:.1e} "
                                         f"RZ {r[f'c{c:g}']['aliased_fraction_rz_half_pi']:.1e}" for c in cs), flush=True)
    report = {"variants": out, "bandwidths": cs, "sample_size": n, "dim": d, "seeds": seeds,
              "note": "angle = c * x of the standardised PCA components (QI1.1, fitted on each QK sample); "
                      "abs_angle_test = |c x| on the 2,000 test candidates (all seeds and components); aliased_fraction = "
                      "share of the training-test pairs x components with |c dx| > pi (R_Y of QK-A: cos^2(c dx / 2) no "
                      "longer decreasing) or > pi / 2 (phase R_Z(2 c x) of QK-Z: cos^2(c dx)), mean over the seeds"}
    save_report(cfg, "QI1.2", report)
    return report


# ---------------------------------------------------------------------------
# QI2 - Feature maps
# ---------------------------------------------------------------------------


def qk_a(d: int):
    """QI2.1 QK-A, angle encoding: R_Y(theta_i) on qubit i, no entanglement (theta = c x, QI1.2)."""
    from qiskit.circuit import ParameterVector, QuantumCircuit

    theta = ParameterVector("theta", d)
    qc = QuantumCircuit(d, name="QK-A")
    for i in range(d):
        qc.ry(theta[i], i)
    return qc


def qk_z(d: int, reps: int = 1, pair: str = "product"):
    """QI2.2 QK-Z, ZZ feature map on a linear chain: ``reps`` times [H on all qubits, R_Z(2 theta_i), R_ZZ(2 phi_i)
    on (i, i + 1)], phi_i = theta_i theta_{i+1} (``pair="product"``, = c^2 x_i x_{i+1}) or the Qiskit default
    (pi - theta_i)(pi - theta_{i+1}) (``pair="qiskit"``)."""
    from qiskit.circuit import ParameterVector, QuantumCircuit

    theta = ParameterVector("theta", d)
    qc = QuantumCircuit(d, name="QK-Z" if pair == "product" else "QK-Z-qiskit")
    for _ in range(reps):
        qc.h(range(d))
        for i in range(d):
            qc.rz(2 * theta[i], i)
        for i in range(d - 1):
            phi = theta[i] * theta[i + 1] if pair == "product" else (np.pi - theta[i]) * (np.pi - theta[i + 1])
            qc.rzz(2 * phi, i, i + 1)
    return qc


def qk_z_qiskit(d: int, reps: int = 1):
    """QK-Z with the Qiskit default pair function (literature comparison of QI2.2)."""
    return qk_z(d, reps, pair="qiskit")


FEATURE_MAPS = {"QK-A": qk_a, "QK-Z": qk_z, "QK-Z-qiskit": qk_z_qiskit}   # QI2.3 (QK-P) to add


def map_reps(name: str, reps) -> list[int]:
    """Repetitions to run for the feature map ``name``: QK-A has none (one layer), the others ``reps``."""
    return [1] if name == "QK-A" else [int(r) for r in reps]


def rep_suffix(name: str, r) -> str:
    """Part of a result key that names the repetitions: empty for QK-A, ``_r<r>`` otherwise."""
    return "" if name == "QK-A" else f"_r{int(r)}"


def feature_map(name: str, d: int, reps: int = 1):
    """Parametrised circuit of the feature map ``name`` on d qubits (``reps`` repetitions for QK-Z);
    parameters = the d angles theta = c x."""
    return FEATURE_MAPS[name](d) if name == "QK-A" else FEATURE_MAPS[name](d, reps)


def qiskit_zz_reference(d: int, reps: int, pair: str = "product"):
    """``qiskit.circuit.library.zz_feature_map`` with linear entanglement, with our pair function
    (``product``) or its default (``qiskit``): the reference of the QK-Z check."""
    from qiskit.circuit.library import zz_feature_map

    f = (lambda x: x[0] if len(x) == 1 else x[0] * x[1]) if pair == "product" else None
    return zz_feature_map(d, reps=reps, entanglement="linear", data_map_func=f)


def bind_angles(qc, theta: np.ndarray):
    """``qc`` with its parameters set to the angles ``theta`` (d,) of one sample."""
    return qc.assign_parameters(np.asarray(theta, dtype=np.float64))


def kernel_qk_a(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """QI2.1 closed form of the QK-A fidelity kernel |<psi(a)|psi(b)>|^2 = prod_i cos^2((a_i - b_i) / 2)
    between the angle rows of ``a`` (n, d) and ``b`` (m, d): (n, m), float64."""
    k = np.ones((a.shape[0], b.shape[0]))
    for i in range(a.shape[1]):
        k *= np.cos((a[:, i, None] - b[None, :, i]) / 2) ** 2
    return k


def statevector_fidelity(qc, a: np.ndarray, b: np.ndarray) -> float:
    """|<psi(a)|psi(b)>|^2 from the Qiskit statevectors of ``qc`` bound to the angles ``a`` and ``b``."""
    from qiskit.quantum_info import Statevector

    return float(abs(Statevector(bind_angles(qc, a)).inner(Statevector(bind_angles(qc, b)))) ** 2)


def circuit_summary(qc) -> dict:
    """Qubits, gate counts, depth and two-qubit gates of a circuit (before transpilation)."""
    ops = qc.count_ops()
    return {"qubits": qc.num_qubits, "gates": {k: int(v) for k, v in ops.items()}, "depth": int(qc.depth()),
            "two_qubit_gates": int(sum(1 for ins in qc.data if ins.operation.num_qubits == 2))}


def run_maps(cfg: dict, n_pairs: int = 20) -> dict:
    """QI2.1 report: per d the QK-A circuit and, for every c, the largest |closed form - statevector fidelity|
    on ``n_pairs`` random training-test pairs of real F-X angles (QK sample of the main grid size, first seed),
    with the range of the kernel values checked."""
    q = cfg["quantum"]
    dims, cs = [int(v) for v in q["dims"]], [float(c) for c in q["bandwidths"]]
    n, seed = int(q["bandwidth_check_size"]), int(cfg["qk_samples"]["seeds"][0])
    x_tr, _, x_te, _ = load_qk(cfg, "F-X", n, seed)
    rng = np.random.default_rng(int(cfg["seed"]))
    i, j = rng.integers(0, x_tr.shape[0], n_pairs), rng.integers(0, x_te.shape[0], n_pairs)
    out = {}
    for d in dims:
        a_tr, a_te = reduce_features(x_tr, d, x_te)
        qc = feature_map("QK-A", d)
        r = {"circuit": circuit_summary(qc)}
        for c in cs:
            ta, tb = feature_angles(a_tr[i], c), feature_angles(a_te[j], c)
            closed = np.array([kernel_qk_a(ta[k:k + 1], tb[k:k + 1])[0, 0] for k in range(n_pairs)])
            sv = np.array([statevector_fidelity(qc, ta[k], tb[k]) for k in range(n_pairs)])
            r[f"c{c:g}"] = {"max_abs_diff_closed_form_statevector": float(f"{np.abs(closed - sv).max():.1e}"),
                            "kernel_min_median_max": [float(f"{v:.3g}") for v in (closed.min(), np.median(closed), closed.max())]}
        out[f"d{d}"] = r
        print(f"QI2.1 d {d}: max |closed - statevector| "
              f"{max(r[f'c{c:g}']['max_abs_diff_closed_form_statevector'] for c in cs):.1e}", flush=True)
    report = {"feature_map": "QK-A", "dims": out, "bandwidths": cs, "pairs": n_pairs, "variant": "F-X",
              "sample": f"n{n}_s{seed}",
              "note": "circuit parameters = angles theta = c x (QI1.2), one circuit for every c; R_Y(theta_i) per qubit, "
                      "no entanglement; closed form prod cos^2(dtheta_i / 2) against the fidelity of the Qiskit "
                      "Statevector of the bound circuits on random training-test pairs; kernel values of the same pairs"}
    save_report(cfg, "QI2.1", report)
    return report


def run_zz(cfg: dict, n_pairs: int = 20, n_ref: int = 3) -> dict:
    """QI2.2 report: per d and r (``quantum.zz_reps``) the circuits of QK-Z and of its Qiskit-pair variant;
    for every c the largest |fidelity of ours - fidelity of ``zz_feature_map``| on ``n_ref`` training-test pairs,
    and the kernel values (min / median / max) on ``n_pairs`` random training-test pairs of real F-X angles (QK
    sample of the main grid size, first seed); per c the quantiles of |2 theta_i theta_{i+1}| (angle of the pair
    terms of QK-Z) on the test candidates."""
    q = cfg["quantum"]
    dims, cs, reps = [int(v) for v in q["dims"]], [float(c) for c in q["bandwidths"]], [int(v) for v in q["zz_reps"]]
    n, seed = int(q["bandwidth_check_size"]), int(cfg["qk_samples"]["seeds"][0])
    x_tr, _, x_te, _ = load_qk(cfg, "F-X", n, seed)
    rng = np.random.default_rng(int(cfg["seed"]))
    i, j = rng.integers(0, x_tr.shape[0], n_pairs), rng.integers(0, x_te.shape[0], n_pairs)
    a20 = reduce_features(x_tr, max(dims), x_te)[1]
    pair_angles = {f"c{c:g}": [round(float(v), 3) for v in (*np.quantile(np.abs(2 * c * c * a20[:, :-1] * a20[:, 1:]), [0.5, 0.99]),
                                                            np.abs(2 * c * c * a20[:, :-1] * a20[:, 1:]).max())] for c in cs}
    out = {}
    for d in dims:
        a_tr, a_te = reduce_features(x_tr, d, x_te)
        for r in reps:
            res = {}
            for pair, name in (("product", "QK-Z"), ("qiskit", "QK-Z-qiskit")):
                qc, ref = feature_map(name, d, r), qiskit_zz_reference(d, r, pair)
                e = {"circuit": circuit_summary(qc)}
                for c in cs:
                    ta, tb = feature_angles(a_tr[i], c), feature_angles(a_te[j], c)
                    k = np.array([statevector_fidelity(qc, ta[p], tb[p]) for p in range(n_pairs)])
                    k_ref = np.array([statevector_fidelity(ref, ta[p], tb[p]) for p in range(n_ref)])
                    e[f"c{c:g}"] = {"max_abs_diff_qiskit_zz_feature_map": float(f"{np.abs(k[:n_ref] - k_ref).max():.1e}"),
                                    "kernel_min_median_max": min_median_max(k)}
                res[name] = e
            out[f"d{d}_r{r}"] = res
            print(f"QI2.2 d {d} r {r}: max |ours - zz_feature_map| "
                  f"{max(res[m][f'c{c:g}']['max_abs_diff_qiskit_zz_feature_map'] for m in res for c in cs):.1e}, "
                  f"median kernel c 0.2 {res['QK-Z']['c0.2']['kernel_min_median_max'][1]}", flush=True)
    report = {"feature_map": "QK-Z", "configs": out, "pair_term_abs_angle_test_q50_q99_max": pair_angles,
              "bandwidths": cs, "reps": reps, "pairs": n_pairs, "reference_pairs": n_ref, "variant": "F-X",
              "sample": f"n{n}_s{seed}",
              "note": "circuit parameters = angles theta = c x (QI1.2); QK-Z = reps x [H, R_Z(2 theta_i), R_ZZ(2 theta_i "
                      "theta_{i+1}) on the linear chain]; QK-Z-qiskit = pair (pi - theta_i)(pi - theta_{i+1}); fidelities "
                      "of the Qiskit statevectors against zz_feature_map (linear entanglement, same pair function) on the "
                      "first reference pairs; kernel values on random training-test pairs; pair-term angle = |2 theta_i "
                      "theta_{i+1}| on the test candidates (d = 20)"}
    save_report(cfg, "QI2.2", report)
    return report


# QI2.3 - Projected quantum kernel (QK-P)


def statevectors(qc, theta: np.ndarray) -> np.ndarray:
    """Qiskit statevectors (m, 2^d) complex128 of ``qc`` bound to every row of the angles ``theta`` (m, d)."""
    from qiskit.quantum_info import Statevector

    return np.stack([Statevector(bind_angles(qc, t)).data for t in np.atleast_2d(theta)])


def local_expectations(psi: np.ndarray) -> np.ndarray:
    """QI2.3: <X>, <Y>, <Z> of every qubit of the states ``psi`` (m, 2^d) (Qiskit order: qubit k = bit k of
    the index), from the single-qubit reduced density matrices rho_k: <X> = 2 Re rho_01, <Y> = -2 Im rho_01,
    <Z> = rho_00 - rho_11.  Returns (m, 3d) float64, qubit-major [X_0, Y_0, Z_0, X_1, ...]."""
    psi = np.atleast_2d(psi)
    m, d = psi.shape[0], int(np.log2(psi.shape[1]))
    t = psi.reshape((m,) + (2,) * d)
    out = np.empty((m, d, 3))
    for k in range(d):
        a = np.moveaxis(t, 1 + d - 1 - k, 1).reshape(m, 2, -1)        # qubit k = axis d - 1 - k (C order)
        rho = np.einsum("mak,mbk->mab", a, a.conj())
        out[:, k] = np.stack([2 * rho[:, 0, 1].real, -2 * rho[:, 0, 1].imag, (rho[:, 0, 0] - rho[:, 1, 1]).real], 1)
    return out.reshape(m, 3 * d)


def sq_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances (n, m) between the rows of ``a`` and ``b``."""
    return np.maximum((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None] - 2 * a @ b.T, 0)


def median_gamma(e_train: np.ndarray) -> float:
    """1 / median of the squared distances between distinct training rows (base of the gamma grid of QK-P)."""
    dist = sq_distances(e_train, e_train)
    return float(1 / np.median(dist[np.triu_indices(dist.shape[0], 1)]))


def kernel_projected(ea: np.ndarray, eb: np.ndarray, gamma: float) -> np.ndarray:
    """QI2.3 projected kernel exp(-gamma ||m(a) - m(b)||^2) between the expectation rows of ``ea`` and ``eb``."""
    return np.exp(-gamma * sq_distances(ea, eb))


def min_median_max(v: np.ndarray) -> list[float]:
    """[min, median, max] of ``v`` with 3 significant digits (report summaries of kernel values)."""
    return [float(f"{t:.3g}") for t in (v.min(), np.median(v), v.max())]


def projected_kernels(e_tr: np.ndarray, e_te: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """(K_train, K_test) of ``kernel_projected`` with width ``gamma``: training x training, test x training."""
    return kernel_projected(e_tr, e_tr, gamma), kernel_projected(e_te, e_tr, gamma)


def run_projected(cfg: dict, n_pairs: int = 20, n_ref: int = 3) -> dict:
    """QI2.3 report: per d, r (``quantum.zz_reps``) and c, on ``n_pairs`` random training-test pairs of real F-X
    angles (the same pairs as QI2.1-QI2.2): the largest |expectation - Qiskit ``expectation_value``| on ``n_ref``
    states, the largest Bloch-vector length, the median gamma of the ``n_pairs`` training states, and the
    min / median / max of the projected kernel (s = 1) and of the fidelity kernel of the same QK-Z states."""
    from qiskit.quantum_info import Pauli, Statevector

    q = cfg["quantum"]
    dims, cs, reps = [int(v) for v in q["dims"]], [float(c) for c in q["bandwidths"]], [int(v) for v in q["zz_reps"]]
    n, seed = int(q["bandwidth_check_size"]), int(cfg["qk_samples"]["seeds"][0])
    x_tr, _, x_te, _ = load_qk(cfg, "F-X", n, seed)
    rng = np.random.default_rng(int(cfg["seed"]))
    i, j = rng.integers(0, x_tr.shape[0], n_pairs), rng.integers(0, x_te.shape[0], n_pairs)
    out = {}
    for d in dims:
        a_tr, a_te = reduce_features(x_tr, d, x_te)
        for r in reps:
            qc, res = feature_map("QK-Z", d, r), {}
            for c in cs:
                s_tr, s_te = statevectors(qc, feature_angles(a_tr[i], c)), statevectors(qc, feature_angles(a_te[j], c))
                e_tr, e_te = local_expectations(s_tr), local_expectations(s_te)
                ref = np.array([[Statevector(s_tr[p]).expectation_value(Pauli("I" * (d - 1 - k) + P + "I" * k)).real
                                 for k in range(d) for P in "XYZ"] for p in range(n_ref)])
                g0 = median_gamma(e_tr)
                kp = np.exp(-g0 * ((e_tr - e_te) ** 2).sum(1))
                kf = np.abs(np.einsum("mk,mk->m", s_tr.conj(), s_te)) ** 2
                res[f"c{c:g}"] = {"max_abs_diff_expectations_qiskit": float(f"{np.abs(e_tr[:n_ref] - ref).max():.1e}"),
                                  "max_bloch_length": round(float(np.linalg.norm(e_tr.reshape(n_pairs, d, 3), axis=2).max()), 6),
                                  "median_gamma": float(f"{g0:.3g}"),
                                  "projected_kernel_s1_min_median_max": min_median_max(kp), "fidelity_kernel_min_median_max": min_median_max(kf)}
            out[f"d{d}_r{r}"] = res
            print(f"QI2.3 d {d} r {r}: max |exp - qiskit| "
                  f"{max(v['max_abs_diff_expectations_qiskit'] for v in res.values()):.1e}, median projected / fidelity c 1: "
                  f"{res['c1']['projected_kernel_s1_min_median_max'][1]} / {res['c1']['fidelity_kernel_min_median_max'][1]}", flush=True)
    report = {"feature_map": "QK-P", "configs": out, "bandwidths": cs, "reps": reps, "gamma_scales": q["projected_gamma_scales"],
              "pairs": n_pairs, "reference_states": n_ref, "variant": "F-X", "sample": f"n{n}_s{seed}",
              "note": "QK-Z state (product pair function); expectations <X>, <Y>, <Z> per qubit from the reduced density "
                      "matrices against Qiskit Statevector.expectation_value; median_gamma = 1 / median squared distance "
                      "of the expectation vectors of the training states of the pairs (indicative: in QI4 it is computed "
                      "on the whole training sample); kernels of the same random training-test pairs, projected at s = 1"}
    save_report(cfg, "QI2.3", report)
    return report


# ---------------------------------------------------------------------------
# QI3.1 - Exact kernels (fast statevectors on the GPU)
# ---------------------------------------------------------------------------


def z_signs(d: int, device="cpu"):
    """(2^d, d) float tensor of z_i = +1 / -1 for bit i = 0 / 1 of every basis index (qubit i = bit i, Qiskit)."""
    import torch

    idx = torch.arange(2 ** d, device=device)
    return 1 - 2 * ((idx[:, None] >> torch.arange(d, device=device)[None]) & 1).to(torch.float64)


def walsh_hadamard(x):
    """H on every qubit of the batched states ``x`` (B, 2^d): fast Walsh-Hadamard transform, normalised."""
    import torch

    b, n = x.shape
    h = 1
    while h < n:
        y = x.view(b, n // (2 * h), 2, h)
        x = torch.stack((y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]), 2).reshape(b, n) / np.sqrt(2)
        h *= 2
    return x


def zz_states(theta, reps: int, pair: str = "product", z=None, dtype=None):
    """QK-Z states (B, 2^d) of the angles ``theta`` (B, d) (torch tensor, its device is used): ``reps`` times
    psi = D(theta) WHT(psi) from |0...0>; D = exp(-i sum_i theta_i z_i - i sum_i phi_i z_i z_{i+1}), phi_i =
    theta_i theta_{i+1} (``product``) or (pi - theta_i)(pi - theta_{i+1}) (``qiskit``).  Equal to the Qiskit
    statevector of ``qk_z`` up to a global phase.  ``z``: ``z_signs`` (reused across calls)."""
    import torch

    dtype = dtype or torch.complex128
    b, d = theta.shape
    z = z_signs(d, theta.device) if z is None else z
    th = theta.to(torch.float64)
    phi = th[:, :-1] * th[:, 1:] if pair == "product" else (np.pi - th[:, :-1]) * (np.pi - th[:, 1:])
    phase = -(th @ z.T) - phi @ (z[:, :-1] * z[:, 1:]).T                    # (B, 2^d), float64
    diag = torch.polar(torch.ones_like(phase), phase).to(dtype)
    psi = torch.full((b, 2 ** d), 2 ** (-d / 2), dtype=dtype, device=theta.device)   # WHT|0...0>
    psi = psi * diag
    for _ in range(reps - 1):
        psi = walsh_hadamard(psi) * diag
    return psi


def expectations_torch(psi):
    """``local_expectations`` on torch states (B, 2^d): (B, 3d) float64, [X_0, Y_0, Z_0, X_1, ...]."""
    import torch

    b, n = psi.shape
    d = int(np.log2(n))
    out = []
    for k in range(d):
        y = psi.view(b, n // (2 ** (k + 1)), 2, 2 ** k)                      # axis 2 = bit k
        a0, a1 = y[:, :, 0].reshape(b, -1), y[:, :, 1].reshape(b, -1)
        r01 = (a0 * a1.conj()).sum(1)
        out.append(torch.stack((2 * r01.real, -2 * r01.imag, (a0.abs() ** 2).sum(1) - (a1.abs() ** 2).sum(1)), 1))
    return torch.stack(out, 1).reshape(b, 3 * d).double()


def map_states(name: str, theta, reps: int, z=None, dtype=None):
    """States of the feature map ``name`` (``QK-Z`` / ``QK-Z-qiskit``) for the angles ``theta``."""
    return zz_states(theta, reps, "product" if name == "QK-Z" else "qiskit", z, dtype)


_AER: dict = {}                                                     # simulator and transpiled circuits, built once


def aer_circuit(name: str, d: int, reps: int):
    """(AerSimulator statevector method, transpiled feature-map circuit with ``save_statevector``), cached."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    if "sim" not in _AER:
        _AER["sim"] = AerSimulator(method="statevector")
    key = (name, d, reps)
    if key not in _AER:
        qc = feature_map(name, d, reps).copy()
        qc.save_statevector()
        _AER[key] = transpile(qc, _AER["sim"])
    return _AER["sim"], _AER[key]


def aer_states(name: str, theta: np.ndarray, reps: int = 1) -> np.ndarray:
    """States (m, 2^d) complex128 of the feature map ``name`` for the angle rows ``theta``, one Aer run."""
    sim, ta = aer_circuit(name, theta.shape[1], reps)
    res = sim.run([ta.assign_parameters(np.asarray(t, dtype=np.float64)) for t in theta]).result()
    return np.stack([np.asarray(res.get_statevector(k)) for k in range(len(theta))])


def state_chunks(name: str, theta: np.ndarray, reps: int, device: str, chunk: int = 256, dtype=None,
                 engine: str = "aer"):
    """Torch states on ``device`` of the angle rows ``theta``, ``chunk`` rows at a time: from Qiskit Aer
    (``engine="aer"``, the reference) or from the PyTorch QK-Z simulator (``"torch"``, cross-check only)."""
    import torch

    dtype = dtype or torch.complex128
    for s in range(0, len(theta), chunk):
        t = theta[s:s + chunk]
        if engine == "aer":
            yield torch.as_tensor(aer_states(name, t, reps), device=device).to(dtype)
        else:
            yield map_states(name, torch.as_tensor(np.asarray(t, dtype=np.float64), device=device), reps, dtype=dtype)


def fidelity_kernels(name: str, th_tr: np.ndarray, th_te: np.ndarray | None = None, reps: int = 1,
                     device: str = "cuda", chunk: int = 256, dtype=None, engine: str = "aer") -> tuple:
    """QI3.1 exact fidelity kernels of the feature map ``name`` from the angles (QI1.2): (K_train (n, n),
    K_test (m, n) or None), float64 numpy.  QK-A by its closed form; QK-Z (-qiskit) from the training states
    (``state_chunks``) computed once and kept on ``device``, the test states in chunks of ``chunk``."""
    import torch

    if name == "QK-A":
        return kernel_qk_a(th_tr, th_tr), (None if th_te is None else kernel_qk_a(th_te, th_tr))
    p_tr = torch.cat(list(state_chunks(name, th_tr, reps, device, chunk, dtype, engine)))
    gram = lambda a: ((a @ p_tr.conj().T).abs() ** 2).double().cpu().numpy()
    k_tr = np.concatenate([gram(p_tr[s:s + chunk]) for s in range(0, len(th_tr), chunk)])
    k_te = None if th_te is None else np.concatenate([gram(a) for a in state_chunks(name, th_te, reps, device, chunk, dtype, engine)])
    return k_tr, k_te


def projected_features(th: np.ndarray, reps: int, device: str = "cuda", chunk: int = 256, dtype=None,
                       engine: str = "aer") -> np.ndarray:
    """QI3.1 for QK-P: the expectations (n, 3d) of the QK-Z states (product pair) of the angles ``th``; the
    kernel is ``kernel_projected`` with gamma from ``median_gamma`` of the training rows."""
    return np.concatenate([expectations_torch(a).cpu().numpy() for a in state_chunks("QK-Z", th, reps, device, chunk, dtype, engine)])


def phase_free_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """|<a|b>| of two normalised states: 1 when they are equal up to a global phase."""
    return float(abs(np.vdot(a, b)))


def run_exact(cfg: dict, device: str = "cuda", n_ref: int = 3, n_prec: int = 200) -> dict:
    """QI3.1 report.  (1) per d, r, pair function: |<Aer|Qiskit Statevector>| and |<Aer|PyTorch>| on ``n_ref`` real
    F-X training states at the largest c (1 = equal up to a global phase), and the QK-P expectations against
    ``local_expectations`` of the Qiskit states;
    (2) per d, r: on the largest QK sample (n = 2,000, first seed) and the 2,000 test candidates, c = 0.2: the
    largest |K complex64 - K complex128| on ``n_prec`` x ``n_prec`` blocks, diagonal and symmetry of K_train, its
    smallest eigenvalue, the time of the Aer states + kernels (and of the QK-P features), products on ``device``."""
    import os
    import time

    import torch

    q = cfg["quantum"]
    dims, reps = [int(v) for v in q["dims"]], [int(v) for v in q["zz_reps"]]
    n_big, seed, c_chk = int(cfg["qk_samples"]["sizes"][-1]), int(cfg["qk_samples"]["seeds"][0]), 0.2
    c_max = max(float(c) for c in q["bandwidths"])
    x_tr, _, x_te, _ = load_qk(cfg, "F-X", n_big, seed)
    sync = (lambda: torch.cuda.synchronize()) if str(device).startswith("cuda") else (lambda: None)
    out = {}
    for d in dims:
        a_tr, a_te = reduce_features(x_tr, d, x_te)
        for r in reps:
            res = {"agreement": {}}
            th_ref = feature_angles(a_tr[:n_ref], c_max)
            for name in ("QK-Z", "QK-Z-qiskit"):
                aer = aer_states(name, th_ref, r)
                qk = statevectors(feature_map(name, d, r), th_ref)
                fast = map_states(name, torch.as_tensor(th_ref, device=device), r, dtype=torch.complex128).cpu().numpy()
                res["agreement"][name] = {
                    "min_abs_overlap_aer_vs_qiskit_statevector": round(min(phase_free_overlap(aer[p], qk[p]) for p in range(n_ref)), 12),
                    "min_abs_overlap_aer_vs_pytorch": round(min(phase_free_overlap(aer[p], fast[p]) for p in range(n_ref)), 12)}
                if name == "QK-Z":
                    res["agreement"]["QK-P_max_abs_diff_expectations_vs_qiskit"] = float(
                        f"{np.abs(projected_features(th_ref, r, device) - local_expectations(qk)).max():.1e}")
            th_tr, th_te = feature_angles(a_tr, c_chk), feature_angles(a_te, c_chk)
            for name in ("QK-Z", "QK-Z-qiskit"):
                sync(); t0 = time.time()
                k_tr, k_te = fidelity_kernels(name, th_tr, th_te, r, device)
                sync(); sec = time.time() - t0
                p32 = fidelity_kernels(name, th_tr[:n_prec], th_te[:n_prec], r, device, dtype=torch.complex64)
                off = k_tr[~np.eye(len(k_tr), dtype=bool)]
                res[name] = {"seconds_train_2000x2000_test_2000x2000": round(sec, 2),
                             "max_abs_diff_complex64_vs_complex128": float(f"{max(np.abs(k_tr[:n_prec, :n_prec] - p32[0]).max(), np.abs(k_te[:n_prec, :n_prec] - p32[1]).max()):.1e}"),
                             "max_abs_diag_minus_1": float(f"{np.abs(np.diag(k_tr) - 1).max():.1e}"),
                             "max_abs_asymmetry": float(f"{np.abs(k_tr - k_tr.T).max():.1e}"),
                             "min_eigenvalue_k_train": float(f"{np.linalg.eigvalsh((k_tr + k_tr.T) / 2).min():.1e}"),
                             "offdiag_mean_c0.2": float(f"{off.mean():.3g}")}
            sync(); t0 = time.time()
            projected_features(th_tr, r, device), projected_features(th_te, r, device)          # timed only
            sync()
            res["QK-P"] = {"seconds_features_4000_states": round(time.time() - t0, 2)}
            out[f"d{d}_r{r}"] = res
            print(f"QI3.1 d {d} r {r}: Aer overlap with Statevector / PyTorch "
                  f"{min(res['agreement'][m]['min_abs_overlap_aer_vs_qiskit_statevector'] for m in ('QK-Z', 'QK-Z-qiskit')):.12f} / "
                  f"{min(res['agreement'][m]['min_abs_overlap_aer_vs_pytorch'] for m in ('QK-Z', 'QK-Z-qiskit')):.12f}, "
                  f"QK-Z {res['QK-Z']['seconds_train_2000x2000_test_2000x2000']} s, c64 err {res['QK-Z']['max_abs_diff_complex64_vs_complex128']}, "
                  f"min eig {res['QK-Z']['min_eigenvalue_k_train']}", flush=True)
    sync(); t0 = time.time()
    a_tr, a_te = reduce_features(x_tr, max(dims), x_te)
    fidelity_kernels("QK-A", feature_angles(a_tr, c_chk), feature_angles(a_te, c_chk))
    report = {"configs": out, "qk_a_closed_form_seconds_d20": round(time.time() - t0, 2), "device": str(device),
              "gpu": torch.cuda.get_device_name() if str(device).startswith("cuda") else None,
              "sample": f"F-X n{n_big}_s{seed} + 2,000 test candidates", "c_kernel_checks": c_chk, "c_qiskit_check": c_max,
              "cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())),
              "dtype": "Aer states in complex128, products in complex128 (complex64 only for the precision check), kernels in float64",
              "note": "states from Qiskit AerSimulator (statevector method, batches of 256 bound circuits, on the CPUs), "
                      "products on the GPU; overlaps |<Aer|Qiskit Statevector>| and |<Aer|PyTorch QK-Z simulator>| (1 = equal "
                      "up to a global phase); complex64 vs complex128 products on the first 200 x 200 entries of K_train and "
                      "K_test (same Aer states); time = training states once + K_train + K_test in chunks"}
    save_report(cfg, "QI3.1", report)
    return report


# ---------------------------------------------------------------------------
# QI3.2 - Verification against the compute-uncompute circuit
# ---------------------------------------------------------------------------


def compute_uncompute(name: str, d: int, reps: int = 1, measure: bool = False):
    """(Aer simulator, transpiled circuit U(theta) U(phi)^+, theta vector, phi vector), cached: P(|0...0>) of the
    circuit = |<psi(phi)|psi(theta)>|^2.  ``measure``: all qubits measured (shots), else ``save_statevector``."""
    from qiskit import transpile
    from qiskit.circuit import ParameterVector

    sim, _ = aer_circuit(name, d, reps)
    key = ("cu", name, d, reps, measure)
    if key not in _AER:
        u = feature_map(name, d, reps)
        phi = ParameterVector("phi", d)
        v = u.assign_parameters(dict(zip(sorted(u.parameters, key=lambda p: p.index), phi)))
        qc = u.compose(v.inverse())
        if measure:
            qc.measure_all()
        else:
            qc.save_statevector()
        _AER[key] = (transpile(qc, sim), sorted(u.parameters, key=lambda p: p.index), list(phi))
    ta, th_vec, ph_vec = _AER[key]
    return sim, ta, th_vec, ph_vec


def zero_probability(name: str, a: np.ndarray, b: np.ndarray, reps: int = 1, shots: int | None = None,
                     seed: int = 0) -> np.ndarray:
    """P(|0...0>) of the compute-uncompute circuits of the angle rows ``a`` (x) and ``b`` (x'), one per row
    pair: exact from the Aer statevector (``shots`` None) or the fraction of all-zero outcomes in ``shots``."""
    d = a.shape[1]
    sim, ta, th_vec, ph_vec = compute_uncompute(name, d, reps, measure=shots is not None)
    circs = [ta.assign_parameters(dict(zip(th_vec + ph_vec, np.concatenate([x, y]).astype(float)))) for x, y in zip(a, b)]
    if shots is None:
        res = sim.run(circs).result()
        return np.array([abs(np.asarray(res.get_statevector(k))[0]) ** 2 for k in range(len(circs))])
    res = sim.run(circs, shots=shots, seed_simulator=seed).result()
    return np.array([res.get_counts(k).get("0" * d, 0) / shots for k in range(len(circs))])


def run_verify(cfg: dict) -> dict:
    """QI3.2 report: per fidelity feature map (QK-A, QK-Z, QK-Z-qiskit), d, r and c, on ``quantum.verify_pairs``
    random training-test pairs of real F-X angles (n = 500 sample, first seed): the largest |K (QI3.1) - P(0...0)|
    of the compute-uncompute circuit simulated exactly by Aer, and for every N of ``quantum.shots`` the largest
    |K - estimate| in binomial standard deviations and the share of pairs within 4 sigma + 1/N."""
    q = cfg["quantum"]
    dims, cs, reps = [int(v) for v in q["dims"]], [float(c) for c in q["bandwidths"]], [int(v) for v in q["zz_reps"]]
    shots, n_pairs = [int(v) for v in q["shots"]], int(q["verify_pairs"])
    n, seed = int(q["bandwidth_check_size"]), int(cfg["qk_samples"]["seeds"][0])
    x_tr, _, x_te, _ = load_qk(cfg, "F-X", n, seed)
    rng = np.random.default_rng(int(cfg["seed"]))
    i, j = rng.integers(0, x_tr.shape[0], n_pairs), rng.integers(0, x_te.shape[0], n_pairs)
    out, worst = {}, {"exact": 0.0, "within_4sigma": 1.0}
    for d in dims:
        a_tr, a_te = reduce_features(x_tr, d, x_te)
        for name in ("QK-A", "QK-Z", "QK-Z-qiskit"):
            for r in map_reps(name, reps):
                res = {}
                for c in cs:
                    ta, tb = feature_angles(a_tr[i], c), feature_angles(a_te[j], c)
                    k = np.diag(fidelity_kernels(name, ta, tb, r, "cpu")[1])          # K(x'_p, x_p) of QI3.1
                    e = {"max_abs_diff_exact": float(f"{np.abs(k - zero_probability(name, ta, tb, r)).max():.1e}"),
                         "kernel_min_max": [float(f"{k.min():.3g}"), float(f"{k.max():.3g}")]}
                    for N in shots:
                        est = zero_probability(name, ta, tb, r, N, seed=int(cfg["seed"]) + N)
                        sig = np.sqrt(k * (1 - k) / N)
                        dev = np.abs(est - k)
                        e[f"N{N}"] = {"max_abs_diff": round(float(dev.max()), 4),
                                      "max_sigmas": round(float((dev / np.maximum(sig, 1e-12)).max()), 2) if (sig > 0).any() else 0.0,
                                      "share_within_4sigma": round(float(np.mean(dev <= 4 * sig + 1 / N)), 3)}
                        worst["within_4sigma"] = min(worst["within_4sigma"], e[f"N{N}"]["share_within_4sigma"])
                    worst["exact"] = max(worst["exact"], e["max_abs_diff_exact"])
                    res[f"c{c:g}"] = e
                out[f"{name}_d{d}{rep_suffix(name, r)}"] = res
                print(f"QI3.2 {name} d {d} r {r}: max |K - P0| exact "
                      f"{max(v['max_abs_diff_exact'] for v in res.values()):.1e}, min share within 4 sigma "
                      f"{min(v[f'N{N}']['share_within_4sigma'] for v in res.values() for N in shots)}", flush=True)
    report = {"configs": out, "pairs": n_pairs, "shots": shots, "bandwidths": cs, "variant": "F-X", "sample": f"n{n}_s{seed}",
              "summary": {"max_abs_diff_exact": worst["exact"], "exact_below_1e-6": bool(worst["exact"] < 1e-6),
                          "min_share_within_4sigma": worst["within_4sigma"]},
              "note": "K = fidelity_kernels of QI3.1 (Aer states + matrix products; QK-A closed form) for random training-test "
                      "pairs; P(0...0) of the compute-uncompute circuit U(x) U(x')^+ simulated by AerSimulator, exact "
                      "(statevector) and with N shots (all qubits measured, seed fixed); sigma = sqrt(K (1 - K) / N); "
                      "within = |estimate - K| <= 4 sigma + 1/N; QK-P is not a fidelity kernel and is checked in QI2.3/QI3.1"}
    save_report(cfg, "QI3.2", report)
    return report


# ---------------------------------------------------------------------------
# QI3.3 - Shot noise and PSD projection
# ---------------------------------------------------------------------------


def shot_kernel(k: np.ndarray, shots: int, rng, symmetric: bool = False) -> np.ndarray:
    """QI3.3 fidelity kernel estimated with ``shots`` measurements per value: Binomial(N, K) / N.  ``symmetric``
    (training matrix): one estimate per pair (upper triangle, mirrored) and diagonal = 1."""
    p = np.clip(k, 0, 1)
    if not symmetric:
        return rng.binomial(shots, p) / shots
    iu = np.triu_indices(k.shape[0], 1)
    out = np.eye(k.shape[0])
    out[iu] = rng.binomial(shots, p[iu]) / shots
    out[(iu[1], iu[0])] = out[iu]
    return out


def shot_expectations(e: np.ndarray, shots: int, rng) -> np.ndarray:
    """QI3.3 for QK-P: every expectation <P> in [-1, 1] estimated from ``shots`` measurements in its basis,
    2 Binomial(N, (1 + <P>) / 2) / N - 1."""
    return 2 * rng.binomial(shots, np.clip((1 + e) / 2, 0, 1)) / shots - 1


def psd_project(k: np.ndarray) -> tuple[np.ndarray, dict]:
    """QI3.3 nearest positive semi-definite matrix of the symmetric ``k`` (Frobenius norm): eigenvalues < 0 set
    to 0.  Returns (projected matrix, {smallest eigenvalue before, eigenvalues clipped, share of the absolute
    eigenvalue mass removed})."""
    w, v = np.linalg.eigh((k + k.T) / 2)
    neg = w < 0
    info = {"min_eigenvalue": float(w.min()), "eigenvalues_clipped": int(neg.sum()),
            "clipped_mass_share": float(np.abs(w[neg]).sum() / np.abs(w).sum())}
    return (v * np.clip(w, 0, None)) @ v.T, info


def noisy_kernels(name: str, exact: tuple, shots: int, rng, gamma_scale: float = 1.0) -> tuple:
    """QI3.3 noisy (K_train projected on the PSD cone, K_test, PSD info) from the exact quantities of
    ``exact``: (K_train, K_test) for the fidelity maps, (training, test expectations) for ``QK-P`` (kernel with
    gamma = ``gamma_scale`` x the median gamma of the noisy training expectations)."""
    if name == "QK-P":
        e_tr, e_te = (shot_expectations(e, shots, rng) for e in exact)
        g = gamma_scale * median_gamma(e_tr)
        k_tr, k_te = projected_kernels(e_tr, e_te, g)
    else:
        k_tr, k_te = shot_kernel(exact[0], shots, rng, symmetric=True), shot_kernel(exact[1], shots, rng)
    k_tr, info = psd_project(k_tr)
    return k_tr, k_te, info


def run_shots(cfg: dict, device: str = "cuda") -> dict:
    """QI3.3 report on the F-X sample of the main grid size (n = 500, first seed) and its 2,000 test candidates:
    per map (QK-A, QK-Z r, QK-P r with s = 1), d, c and N of ``quantum.shots``: relative Frobenius error
    ||K_hat - K|| / ||K|| of the training matrix after the PSD projection and of the test matrix, share of the
    off-diagonal training values estimated as exactly 0, smallest eigenvalue before the projection and share of
    the eigenvalue mass clipped; the exact mean off-diagonal training value for context."""
    q = cfg["quantum"]
    dims, cs, reps = [int(v) for v in q["dims"]], [float(c) for c in q["bandwidths"]], [int(v) for v in q["zz_reps"]]
    shots = [int(v) for v in q["shots"]]
    n, seed = int(q["bandwidth_check_size"]), int(cfg["qk_samples"]["seeds"][0])
    x_tr, _, x_te, _ = load_qk(cfg, "F-X", n, seed)
    rel = lambda a, b: float(f"{np.linalg.norm(a - b) / np.linalg.norm(b):.3g}")
    off = ~np.eye(n, dtype=bool)
    out = {}
    for d in dims:
        a_tr, a_te = reduce_features(x_tr, d, x_te)
        for m_id, (name, r) in enumerate([("QK-A", 1)] + [(m, r) for m in ("QK-Z", "QK-P") for r in reps]):
            res = {}
            for c_id, c in enumerate(cs):
                th_tr, th_te = feature_angles(a_tr, c), feature_angles(a_te, c)
                if name == "QK-P":
                    exact = (projected_features(th_tr, r, device), projected_features(th_te, r, device))
                    g0 = median_gamma(exact[0])
                    k_ex = projected_kernels(exact[0], exact[1], g0)
                else:
                    exact = k_ex = fidelity_kernels(name, th_tr, th_te, r, device)
                e = {"exact_offdiag_mean": float(f"{k_ex[0][off].mean():.3g}")}
                for N in shots:
                    rng = np.random.default_rng([int(cfg["seed"]), N, d, r, c_id, m_id])
                    k_tr, k_te, info = noisy_kernels(name, exact, N, rng)
                    raw0 = shot_kernel(k_ex[0], N, np.random.default_rng([int(cfg["seed"]), N, d, r, c_id, m_id]), True) \
                        if name != "QK-P" else None
                    e[f"N{N}"] = {"rel_frobenius_error_train_psd": rel(k_tr, k_ex[0]), "rel_frobenius_error_test": rel(k_te, k_ex[1]),
                                  "share_train_values_zero": None if raw0 is None else round(float((raw0[off] == 0).mean()), 4),
                                  "min_eigenvalue_before_psd": float(f"{info['min_eigenvalue']:.3g}"),
                                  "clipped_mass_share": float(f"{info['clipped_mass_share']:.3g}")}
                res[f"c{c:g}"] = e
            key = f"{name}_d{d}{rep_suffix(name, r)}"
            out[key] = res
            print(f"QI3.3 {key}: rel. error train N {shots[0]} / {shots[-1]} at c 0.2: "
                  f"{res['c0.2'][f'N{shots[0]}']['rel_frobenius_error_train_psd']} / {res['c0.2'][f'N{shots[-1]}']['rel_frobenius_error_train_psd']}", flush=True)
    report = {"configs": out, "shots": shots, "bandwidths": cs, "variant": "F-X", "sample": f"n{n}_s{seed} + 2,000 test candidates",
              "note": "exact kernels of QI3.1 (Aer states); fidelity maps: Binomial(N, K) / N per training pair (once, mirrored, "
                      "diagonal 1) and per test-training pair; QK-P: every <X>, <Y>, <Z> from N shots, Gaussian kernel with "
                      "s = 1 x median gamma of the noisy training expectations; training matrix projected on the PSD cone "
                      "(negative eigenvalues set to 0); rel. Frobenius error = ||K_hat - K|| / ||K||; share_train_values_zero = "
                      "off-diagonal training values estimated as 0 (before the projection); fixed seeds per configuration"}
    save_report(cfg, "QI3.3", report)
    return report


def run_simtime(cfg: dict, budget_s: float = 60.0) -> dict:
    """QI3.1 timing of the standard Qiskit simulators, to compare with the PyTorch path of ``run_exact``.  Per d
    and r, on real F-X angles (n = 500 sample, first seed, c = 0.2): seconds per state of ``Statevector`` and of
    ``AerSimulator(method="statevector")`` (one run on a batch of bound, transpiled circuits with
    ``save_statevector``), each measured on as many states as fit in ``budget_s`` (at least 2); agreement of
    the Aer states with the PyTorch states.  Extrapolations (states only, the kernel products and SVM fits are
    excluded): the QI3.1 check (QK-Z and QK-Z-qiskit, 2,000 + 2,000 states per d and r) and the QI4.3 main grid
    (QK-Z states, shared by QK-P: 5 c x 5 seeds x (500 training + 2,000 test) per d and r; and with the PCA refitted
    in each of the 5 folds of QI4.2, + 5 x 500 states per configuration)."""
    import os
    import time

    import torch
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator

    q = cfg["quantum"]
    dims, reps = [int(v) for v in q["dims"]], [int(v) for v in q["zz_reps"]]
    n, seed, c = int(q["bandwidth_check_size"]), int(cfg["qk_samples"]["seeds"][0]), 0.2
    n_c, n_seeds, n_test = len(q["bandwidths"]), len(cfg["qk_samples"]["seeds"]), int(cfg["test_subset"]["size"])
    x_tr, _, _, _ = load_qk(cfg, "F-X", n, seed)
    sim = AerSimulator(method="statevector")

    def timed(fn, cap):
        """Seconds per state of ``fn(k)`` (computes k states), doubling k from 2 until ``budget_s`` or ``cap``."""
        k = 2
        while True:
            t0 = time.time()
            fn(k)
            sec = time.time() - t0
            per = sec / k
            if sec > budget_s / 3 or 2 * k > cap:
                return per, k
            k *= 2

    out, tot = {}, {"qi31_check": [0.0, 0.0], "qi43_grid": [0.0, 0.0], "qi43_grid_cv_refit": [0.0, 0.0]}
    for d in dims:
        th = feature_angles(reduce_features(x_tr, d)[0], c)
        for r in reps:
            qc = feature_map("QK-Z", d, r)
            sv_per, sv_k = timed(lambda k: [Statevector(bind_angles(qc, th[p])) for p in range(k)], 64)
            qa = qc.copy()
            qa.save_statevector()
            t0 = time.time()
            ta = transpile(qa, sim)
            t_transpile = time.time() - t0
            aer_states = {}

            def aer(k):
                res = sim.run([ta.assign_parameters(th[p]) for p in range(k)]).result()
                aer_states["psi"] = np.asarray(res.get_statevector(0))

            aer_per, aer_k = timed(aer, 256)
            fast = zz_states(torch.as_tensor(th[:1]), r).numpy()[0]
            states = {"qi31_check": 2 * 2 * n_test, "qi43_grid": n_c * n_seeds * (n + n_test),
                      "qi43_grid_cv_refit": n_c * n_seeds * (n + n_test + 5 * n)}
            for key, m in states.items():
                tot[key][0] += m * sv_per
                tot[key][1] += m * aer_per
            out[f"d{d}_r{r}"] = {"seconds_per_state_statevector": float(f"{sv_per:.3g}"), "states_timed_statevector": sv_k,
                                 "seconds_per_state_aer": float(f"{aer_per:.3g}"), "states_timed_aer": aer_k,
                                 "seconds_transpile_aer": round(t_transpile, 3),
                                 "aer_vs_pytorch_abs_overlap": round(phase_free_overlap(aer_states["psi"], fast), 12)}
            print(f"simtime d {d} r {r}: Statevector {sv_per:.3g} s/state, Aer {aer_per:.3g} s/state "
                  f"(overlap with PyTorch {out[f'd{d}_r{r}']['aer_vs_pytorch_abs_overlap']})", flush=True)
    report = {"configs": out, "threads": os.cpu_count() if "SLURM_CPUS_PER_TASK" not in os.environ else int(os.environ["SLURM_CPUS_PER_TASK"]),
              "extrapolated_hours_states_only": {k: {"statevector": round(v[0] / 3600, 2), "aer": round(v[1] / 3600, 2)} for k, v in tot.items()},
              "assumptions": {"qi31_check": "QK-Z and QK-Z-qiskit, 2,000 training + 2,000 test states, every d and r",
                              "qi43_grid": f"QK-Z states (QK-P reuses them), {n_c} c x {n_seeds} seeds x ({n} + {n_test}) states, every d and r",
                              "qi43_grid_cv_refit": "as qi43_grid + 5 folds x 500 states (PCA refitted in each fold, QI4.2)"},
              "sample": f"F-X n{n}_s{seed}", "c": c,
              "note": "time per state = wall clock / states, one process; Aer: one run on a batch of transpiled circuits bound "
                      "to the angles (save_statevector), the transpilation (once per d, r) excluded; kernel products, SVM fits "
                      "and the learning curve excluded; the PyTorch path (QI3.1) computes a full 2,000 x 2,000 configuration "
                      "in <= 5.3 s on one H100"}
    save_report(cfg, "QI3.1-qiskit", report)
    return report


def selftest() -> None:
    """``reduce_features`` against scikit-learn PCA + StandardScaler (up to the sign of each component)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 40)) @ rng.normal(size=(40, 40)) + 3
    xt = rng.normal(size=(50, 40)) @ rng.normal(size=(40, 40))
    a_tr, a_te = reduce_features(x, 6, xt)
    pca = PCA(6).fit(x)
    sc = StandardScaler().fit(pca.transform(x))
    b_tr, b_te = sc.transform(pca.transform(x)), sc.transform(pca.transform(xt))
    sign = np.sign(np.sum(a_tr * b_tr, 0))
    corr = np.sqrt(300 / 299)                                          # StandardScaler: ddof 0
    assert np.allclose(a_tr * sign * corr, b_tr, atol=1e-8) and np.allclose(a_te * sign * corr, b_te, atol=1e-8)
    assert np.allclose(np.cumsum(fit_reduction(x, 6)["explained"]), np.cumsum(pca.explained_variance_ratio_), atol=1e-10)
    q, _ = np.linalg.qr(rng.normal(size=(40, 6)))
    assert abs(subspace_overlap(q, q) - 1) < 1e-12 and abs(subspace_overlap(q[:, :3], q[:, 3:])) < 1e-12
    a, b = rng.normal(size=(30, 3)), rng.normal(size=(40, 3))
    brute = float(np.mean(np.abs(a[:, None, :] - b[None, :, :]) > 1.1))
    assert abs(aliased_fraction(a, b, 1.1) - brute) < 1e-12
    th = rng.uniform(-np.pi, np.pi, size=(4, 5))
    qc = feature_map("QK-A", 5)
    k = kernel_qk_a(th, th)
    assert np.allclose(np.diag(k), 1) and np.allclose(k, k.T)
    assert all(abs(k[p, r] - statevector_fidelity(qc, th[p], th[r])) < 1e-12 for p in range(4) for r in range(4))
    for pair, name in (("product", "QK-Z"), ("qiskit", "QK-Z-qiskit")):
        for r in (1, 2):
            qz, ref = feature_map(name, 5, r), qiskit_zz_reference(5, r, pair)
            assert all(abs(statevector_fidelity(qz, th[p], th[t]) - statevector_fidelity(ref, th[p], th[t])) < 1e-12
                       for p in range(4) for t in range(4))
    from qiskit.quantum_info import Pauli, Statevector

    qz = feature_map("QK-Z", 5, 2)
    sv = statevectors(qz, th)
    e = local_expectations(sv)
    ref = np.array([[Statevector(v).expectation_value(Pauli("I" * (4 - k) + P + "I" * k)).real for k in range(5) for P in "XYZ"]
                    for v in sv])
    assert np.allclose(e, ref, atol=1e-12)
    ea = local_expectations(statevectors(feature_map("QK-A", 5), th)).reshape(4, 5, 3)   # R_Y: (sin, 0, cos)
    assert np.allclose(ea[..., 0], np.sin(th)) and np.allclose(ea[..., 1], 0) and np.allclose(ea[..., 2], np.cos(th))
    kp = kernel_projected(e, e, median_gamma(e))
    assert np.allclose(np.diag(kp), 1) and np.allclose(kp, kp.T) and np.all(np.linalg.eigvalsh(kp) > -1e-10)
    import torch

    for name in ("QK-Z", "QK-Z-qiskit"):
        for r in (1, 2):
            fast = map_states(name, torch.as_tensor(th), r, dtype=torch.complex128).numpy()
            qk = statevectors(feature_map(name, 5, r), th)
            assert all(abs(phase_free_overlap(fast[p], qk[p]) - 1) < 1e-12 for p in range(4))
            assert all(abs(phase_free_overlap(aer_states(name, th, r)[p], qk[p]) - 1) < 1e-12 for p in range(4))
            for engine in ("aer", "torch"):
                k_tr, k_te = fidelity_kernels(name, th[:3], th[3:], r, "cpu", chunk=2, engine=engine)
                assert np.allclose(k_tr, np.abs(qk[:3].conj() @ qk[:3].T) ** 2) and np.allclose(k_te, np.abs(qk[3:].conj() @ qk[:3].T) ** 2)
    for engine in ("aer", "torch"):
        assert np.allclose(projected_features(th, 2, "cpu", chunk=3, engine=engine), e, atol=1e-12)
    for name, r in (("QK-A", 1), ("QK-Z", 2), ("QK-Z-qiskit", 1)):
        k = np.diag(fidelity_kernels(name, th[:2], th[2:], r, "cpu")[1])
        assert np.allclose(zero_probability(name, th[2:], th[:2], r), k, atol=1e-10)
    rng2 = np.random.default_rng(1)
    kk = kernel_qk_a(th, th)
    ks = shot_kernel(kk, 100000, rng2, symmetric=True)
    assert np.allclose(np.diag(ks), 1) and np.allclose(ks, ks.T) and np.abs(ks - kk).max() < 0.01
    m = rng2.normal(size=(6, 6))
    kp, info = psd_project(m + m.T)
    assert np.linalg.eigvalsh(kp).min() > -1e-10 and info["eigenvalues_clipped"] > 0
    w, v = np.linalg.eigh(m + m.T)
    assert np.allclose(kp, (v * np.clip(w, 0, None)) @ v.T)
    es = shot_expectations(e, 200000, rng2)
    assert np.abs(es - e).max() < 0.02 and np.abs(es).max() <= 1
    print("qc_kernels selftest: OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["features", "pca", "bandwidth", "maps", "exact", "verify", "shots", "simtime"])
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--map", default="QK-A", choices=["QK-A", "QK-Z", "QK-P"], help="maps: feature map to check (QI2.1 / QI2.2 / QI2.3)")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if args.command is None:
        ap.error("a command or --selftest is required")
    cfg = load_config(args.config)
    if args.command == "features":
        print(json.dumps(run_features(cfg), indent=1))
    elif args.command == "pca":
        r = run_pca(cfg)
        print(json.dumps({k: r[k] for k in ("dims", "sizes", "checks")}, indent=1))
    elif args.command == "bandwidth":
        run_bandwidth(cfg)
    elif args.command == "maps":
        {"QK-A": run_maps, "QK-Z": run_zz, "QK-P": run_projected}[args.map](cfg)
    elif args.command == "exact":
        run_exact(cfg, args.device)
    elif args.command == "verify":
        print(json.dumps(run_verify(cfg)["summary"], indent=1))
    elif args.command == "shots":
        run_shots(cfg, args.device)
    elif args.command == "simtime":
        print(json.dumps(run_simtime(cfg)["extrapolated_hours_states_only"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
