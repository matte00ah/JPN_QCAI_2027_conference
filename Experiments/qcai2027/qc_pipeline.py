"""Runner of the three macro blocks of the experiment, in the order of the plan.

Each block is an ordered list of steps; a step is one command of one ``qc_*.py`` script, run as a separate
Python process from the project root (so every step frees its memory / GPU and a failure stops the block where
it happens).  The three blocks, each reading the outputs of the previous one:

  preprocessing    QA-QD   checks of the raw data, crater labels, geographic split, tiles, the cleaned rasters
                           of region R (WAC mosaic, DEM on the WAC grid and its derivatives, validity mask,
                           normalisation) and the data figures of the paper; CPU only
  representations  QE-QG   frozen DINOv2 tokens, self-supervised fusion (F-X and its control F-CM), probes of the
                           six representations; GPU recommended
  detection        QH-QK   stage 1 (candidates), quantum-kernel stage 2 with its diagnostics, classical baselines
                           and statistics, costs, hardware-noise study, crater-parameter regression and the
                           quantum-vs-classical figure; GPU recommended, plus long CPU steps

The steps are idempotent or resumable: the heavy ones keep their progress and the rasters are not rewritten
(see each script), so a block stopped by a failure is restarted with ``--from <step>``.

Entry points: ``run_preprocessing.py``, ``run_representations.py``, ``run_detection.py`` (or
``python qc_pipeline.py <block>``).  Options: ``--device cuda|cpu`` (default: cuda if available) for the steps
that use torch, ``--list`` (print the steps and exit), ``--from N`` / ``--to N`` (run a range of steps, 1-based),
``--dry-run`` (print the commands only).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]


@dataclass(frozen=True)
class Step:
    plan: str                     # step(s) of the plan
    script: str                   # qc_*.py in this folder
    args: tuple[str, ...]         # command and fixed options
    device: bool = False          # takes ``--device`` (torch steps)
    note: str = ""                # what it does


S = Step
BLOCKS: dict[str, list[Step]] = {
    "preprocessing": [
        S("QA2-QA3", "qc_checks.py", ("dinov2",), note="DINOv2 weights"),
        S("QA2", "qc_checks.py", ("data",), note="copies of the raw products against the originals"),
        S("QA3", "qc_checks.py", ("catalog",), note="crater catalogue"),
        S("QB1.1", "qc_data.py", ("region",), note="geometry of region R"),
        S("QB2.1", "qc_data.py", ("craters",), note="craters of R"),
        S("QB2.2-QB2.3", "qc_data.py", ("cells",), note="labels per sub-tile"),
        S("QB2.4", "qc_data.py", ("label-report",), note="report on the labels"),
        S("QB3.1", "qc_data.py", ("split",), note="geographic split"),
        S("QB3.2", "qc_tiles.py", ("buffer",), note="tiles and buffer between sets"),
        S("QC1.1", "qc_preprocess.py", ("wac",), note="WAC mosaic of R"),
        S("QC1.2", "qc_preprocess.py", ("dem",), note="checks of the DEM reading"),
        S("QC2.1", "qc_preprocess.py", ("dem-grid",), note="DEM on the WAC grid"),
        S("QC2.2", "qc_preprocess.py", ("coreg",), note="WAC-DEM co-registration check"),
        S("QC3.1", "qc_preprocess.py", ("mask",), note="validity mask"),
        S("QC4.1", "qc_preprocess.py", ("relief",), note="local relief"),
        S("QC4.2", "qc_preprocess.py", ("slope",), note="slope"),
        S("QC4.3", "qc_preprocess.py", ("hillshade",), note="hillshade"),
        S("QC5.1", "qc_preprocess.py", ("norm",), note="normalisation parameters"),
        S("QC7.1", "qc_tiles.py", ("grid",), note="tile grid check"),
        S("QC7.2", "qc_tiles.py", ("ownership",), note="sub-tile ownership check"),
        S("QC7.3", "qc_tiles.py", ("read-check",), note="tile reading check"),
        S("QC8.2", "qc_tiles.py", ("counts",), note="final counts per set"),
        S("QB2.1, QB3.2", "qc_figures.py", ("qb-checks",), note="check figure of labels and split"),
        S("QC8.1", "qc_figures.py", ("tile-check",), note="check figure of the input tiles"),
        S("QD-A1", "qc_figures.py", ("region-map",), note="Fig. D1 region map"),
        S("QD-A2", "qc_figures.py", ("sfd",), note="Fig. D2 size-frequency distribution"),
        S("QD-A3", "qc_figures.py", ("example-tiles",), note="Fig. D3 example tiles"),
        S("QD-A4", "qc_figures.py", ("dataset-table",), note="Table D1 dataset"),
        S("QD-A5", "qc_figures.py", ("complementarity", "--pdf"), note="Fig. D4 complementarity of WAC and DEM"),
    ],
    "representations": [
        S("QE1.1", "qc_features.py", ("backbone",), True, "frozen backbone check"),
        S("QE1.2", "qc_features.py", ("alignment",), True, "token-label alignment check"),
        S("QE2.1-QE2.2", "qc_features.py", ("extract",), True, "DINOv2 token cache"),
        S("QF1", "qc_fusion.py", ("variants",), note="registry of the six representations"),
        S("QF2", "qc_fusion.py", ("model",), note="F-X model check"),
        S("QF3.1-QF3.2", "qc_fusion.py", ("masking",), note="masking check"),
        S("QF3.3", "qc_fusion.py", ("decoder",), note="decoder check"),
        S("QF3.4", "qc_fusion.py", ("train",), True, "self-supervised training of F-X and F-CM"),
        S("QF4.1-QF4.2", "qc_fusion.py", ("checks",), note="checks of the trained fusions"),
        S("QF4.3", "qc_fusion.py", ("cache",), note="cache of the fused maps"),
        S("QG1", "qc_probes.py", ("linear",), True, "linear probe"),
        S("QG2", "qc_probes.py", ("mlp",), True, "MLP probe"),
        S("QG3", "qc_probes.py", ("pca",), True, "probe on principal components"),
        S("QG4", "qc_probes.py", ("missing",), True, "missing-modality robustness"),
        S("QG6", "qc_probes.py", ("terrain",), note="results per terrain"),
    ],
    "detection": [
        S("QH1", "qc_stage1.py", ("train",), True, "stage-1 heads"),
        S("QH2", "qc_stage1.py", ("threshold", "--device", "cpu"), note="stage-1 thresholds"),
        S("QH3", "qc_stage1.py", ("candidates", "--device", "cpu"), note="stage-1 candidates"),
        S("QH4", "qc_stage1.py", ("qk-samples",), note="quantum-stage training samples"),
        S("QH5", "qc_stage1.py", ("test-subset",), note="2,000 test candidates"),
        S("QI1.1", "qc_kernels.py", ("features",), note="inputs of the quantum stage"),
        S("QI1.1", "qc_kernels.py", ("pca",), note="PCA reduction"),
        S("QI1.2", "qc_kernels.py", ("bandwidth",), note="bandwidth check"),
        S("QI2.1", "qc_kernels.py", ("maps",), note="QK-A against Qiskit"),
        S("QI2.2", "qc_kernels.py", ("maps", "--map", "QK-Z"), note="QK-Z against Qiskit"),
        S("QI2.3", "qc_kernels.py", ("maps", "--map", "QK-P"), note="QK-P against Qiskit"),
        S("QI3.1", "qc_kernels.py", ("exact",), True, "exact kernels"),
        S("QI3.2", "qc_kernels.py", ("verify",), note="compute-uncompute verification"),
        S("QI3.3", "qc_kernels.py", ("shots",), True, "shot-noise kernels"),
        S("QI4.1", "qc_experiments.py", ("svm-check",), True, "SVM on precomputed kernels"),
        S("QI4.2", "qc_experiments.py", ("cv-check",), True, "spatial cross-validation check"),
        S("QI4.3", "qc_experiments.py", ("grid",), True, "main grid on F-X"),
        S("QI4.3", "qc_experiments.py", ("curve",), True, "learning curve"),
        S("QI4.4", "qc_experiments.py", ("metrics",), note="test metrics and end-to-end comparison"),
        S("QI5.1", "qc_diagnostics.py", ("concentration",), True, "kernel concentration"),
        S("QI5.2", "qc_diagnostics.py", ("alignment",), True, "kernel-target alignment"),
        S("QI5.3", "qc_diagnostics.py", ("geometric",), True, "geometric difference"),
        S("QI5.4", "qc_diagnostics.py", ("spectrum",), True, "kernel spectra"),
        S("QI6.1", "qc_experiments.py", ("representations",), True, "stage 2 on the other representations"),
        S("QJ1", "qc_experiments.py", ("baselines",), note="classical baselines"),
        S("QJ2", "qc_experiments.py", ("references",), note="reference lines"),
        S("QJ1-QJ2", "qc_figures.py", ("quantum-vs-classical", "--pdf"), note="Fig. J1 quantum vs classical"),
        S("QJ3", "qc_experiments.py", ("statistics",), note="statistics of the comparison"),
        S("QJ4", "qc_experiments.py", ("costs",), True, "costs"),
        S("QK1-QK2", "qc_noise.py", ("run",), note="kernels under IBM device noise, M3 mitigation"),
        S("QK3", "qc_noise.py", ("resources",), note="hardware resource table"),
        S("QI7.1", "qc_regression.py", ("run",), True, "regression of the crater parameters"),
    ],
}


def default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def command(step: Step, device: str) -> list[str]:
    return [sys.executable, "-u", str(HERE / step.script), *step.args, *(["--device", device] if step.device else [])]


def run_block(block: str, argv: list[str] | None = None) -> int:
    steps = BLOCKS[block]
    ap = argparse.ArgumentParser(description=f"Run the '{block}' block of the experiment ({len(steps)} steps).")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None, help="torch device (default: cuda if available)")
    ap.add_argument("--from", dest="first", type=int, default=1, help="first step to run (1-based, see --list)")
    ap.add_argument("--to", dest="last", type=int, default=len(steps), help="last step to run")
    ap.add_argument("--list", action="store_true", help="print the steps and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args(argv)
    if args.list:
        for k, s in enumerate(steps, 1):
            print(f"{k:3d}  {s.plan:13s} {s.script} {' '.join(s.args)}{' [--device]' if s.device else ''}  - {s.note}")
        return 0
    if not 1 <= args.first <= args.last <= len(steps):
        ap.error(f"need 1 <= --from <= --to <= {len(steps)}")
    todo = range(args.first, args.last + 1)
    needs_device = any(steps[k - 1].device for k in todo)
    device = args.device or (default_device() if needs_device else "cpu")       # torch imported only if needed
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    t_block = time.time()
    for k in todo:
        s = steps[k - 1]
        cmd = command(s, device)
        shown = " ".join(["python", str(Path(cmd[2]).relative_to(PROJECT_ROOT)), *cmd[3:]])
        print(f"\n=== [{block} {k}/{len(steps)}] {s.plan}: {s.note}\n    {shown}  ({time.strftime('%H:%M:%S')})", flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        code = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env).returncode
        if code != 0:
            print(f"\nFAILED: step {k} ({s.plan}, exit code {code}). Fix it, then resume with --from {k}.", flush=True)
            return code
        print(f"    done in {time.time() - t0:.0f} s", flush=True)
    print(f"\n{block}: steps {args.first}-{args.last} " + ("listed (dry run)" if args.dry_run else f"done in {(time.time() - t_block) / 60:.1f} min"), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in BLOCKS:
        print(f"usage: python qc_pipeline.py {{{','.join(BLOCKS)}}} [options]  (options: --help after the block)")
        return 2
    return run_block(argv[0], argv[1:])


if __name__ == "__main__":
    sys.exit(main())
