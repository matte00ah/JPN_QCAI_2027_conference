# QCAI 2027 - hybrid quantum-kernel crater detection (proxy experiment)

Code of the preliminary experiment for the paper to the 2nd QCAI Workshop (IEEE QCNC 2027).

The experiment tests, on one lunar region, a hybrid crater detector:

1. frozen foundation-model features (DINOv2 ViT-S/14) of LROC WAC optical images and of SLDEM2015 elevation;
2. a multimodal fusion layer trained with a self-supervised objective;
3. a classical first stage that proposes candidate sub-tiles (1.6 km);
4. a quantum-kernel SVM (simulated) that re-classifies the candidates;

compared with classical baselines on exactly the same inputs, with diagnostics of the quantum kernels and a
study under realistic hardware noise.
The full plan, with steps, decisions, status and results, is in `JPN_QCAI_conference_plan.md`.

**Region R**: 30°S-30°N, 0-90°E (maria and highlands). On the WAC grid, the reference grid of the experiment:
18,194 x 27,291 px at 100 m (~1,820 x 2,730 km); on the SLDEM grid 30,720 x 46,080 px at 59 m.

## Requirements

- Python ≥ 3.12, Linux or macOS.
- An NVIDIA GPU with CUDA is strongly recommended for blocks 2 and 3 (feature extraction, fusion training,
  probes, stage 1, quantum-kernel products). They also run on CPU with `--device cpu`, much more slowly.
- ~16 GB of RAM; ~35 GB of free disk (raw products ~9 GB; produced by the pipeline: cleaned rasters ~7 GB, token caches, models and scores ~14 GB).

### Installation

```bash
git clone <this-repository-url> JPN_QCAI_conference
cd JPN_QCAI_conference
python -m venv .venv && source .venv/bin/activate      # or any other environment manager (conda, uv, ...)
# optional: install the PyTorch build that matches your CUDA version first (see requirements.txt)
pip install -r requirements.txt
```

The DINOv2 weights (`timm` model `vit_small_patch14_dinov2.lvd142m`) are downloaded from the Hugging Face Hub
the first time a backbone step runs and are then read from the local cache (`~/.cache/huggingface/hub/`); after
that, the steps can run offline with `HF_HUB_OFFLINE=1`.

### Data

The raw data are not part of the repository. Each data folder has a `downloads.txt` with the URLs of its files;
`Dataset/scripts/A_sources/download_subset.py` downloads them (standard library only, resumable, files already
present are skipped). From the project root:

```bash
python Dataset/scripts/A_sources/download_subset.py --list Moon/LROC --out Moon/LROC/WAC_global
python Dataset/scripts/A_sources/download_subset.py --list Moon/LOLA --out Moon/LOLA/data/sldem2015/tiles/float_img
python Dataset/scripts/A_sources/download_subset.py --list Moon/catalogs/robbins2019
cd Dataset/Moon/catalogs/robbins2019 && unzip lunar_crater_database_robbins_2018 && cd -   # the catalogue comes as a ZIP
```

(`--dry-run` shows what would be downloaded; `--out` is relative to `Dataset/`.) The files end up in:

| Path | Product | Source |
|---|---|---|
| `Dataset/Moon/LROC/WAC_global/WAC_GLOBAL_E300{N,S}0450_100M.IMG` (~2 GB each) | LROC WAC global mosaic, 643 nm, 100 m/px, PDS3 with attached label (0-60°N and 60°S-0, 0-90°E) | PDS LROC archive (`LROLRC_2001/DATA/BDR/WAC_GLOBAL/`) |
| `Dataset/Moon/LOLA/data/sldem2015/tiles/float_img/sldem2015_512_{00n_30n,30s_00s}_{000_045,045_090}_float.{img,lbl,xml}` (~1.4 GB each) | SLDEM2015 (LOLA + Kaguya TC) elevation, 512 px/deg, float32 radius in km | PDS Geosciences Node, LRO LOLA RDR (`data/sldem2015/tiles/float_img/`) |
| `Dataset/Moon/catalogs/robbins2019/lunar_crater_database_robbins_2018.csv` (+ `.xml` label) | Robbins (2019) lunar crater database, 1.3 M craters ≥ 1 km | USGS Astrogeology (see `SOURCE.md` in the same folder; MD5 checked by block 1) |

**Dataset tooling.** The code reuses the dataset modules of the larger project this experiment comes from,
copied unchanged in `Dataset/scripts/`: PDS label parsing, raster readers with no-data handling and
georeferencing corrections, DEM and co-registration helpers (`data_inventory`, `data_quality`, `diag_dem_sources`,
`diag_lamp_snr`, `diag_coregistration`) and the downloader. It also reads the inventory and diagnostics of that
project, reduced to the rows of the six products used here: `Dataset/Moon/preprocessing/sources/`
(`data_inventory.csv`, `data_valid_coverage.csv`, `data_quality_metrics.csv`) and
`Dataset/Moon/preprocessing/diagnostics/B4/coregistration.csv`.

### Configuration

All the parameters (paths, region, products, tiling, labels, split, pre-processing, models, quantum stage,
baselines, seed) are in `Experiments/qcai2027/qc_config.yaml`. Relative paths are relative to the project root.

## How to run

The experiment is three blocks, run in this order from the **project root**. Each block reads the outputs of
the previous one:

```bash
python Experiments/qcai2027/run_preprocessing.py      # 1. data checks, labels, split, cleaned rasters of R, data figures
python Experiments/qcai2027/run_representations.py    # 2. DINOv2 tokens, self-supervised fusion, probes
python Experiments/qcai2027/run_detection.py          # 3. stage 1, quantum-kernel stage 2, baselines, noise study
```

| Block | Plan phases | What it produces | Hardware | Indicative time |
|---|---|---|---|---|
| 1. `run_preprocessing.py` | QA-QD | crater labels per 1.6 km sub-tile, geographic split, tiles; the cleaned rasters of R (WAC mosaic, DEM on the WAC grid, relief, slope, hillshade, normalisation); data figures and table of the paper | CPU | ~1-2 h |
| 2. `run_representations.py` | QE-QG | DINOv2 token cache, trained fusions F-X and F-CM with their fused-map cache, probes of the six representations (Table I) | GPU | ~1-2 h |
| 3. `run_detection.py` | QH-QK, QI7 | stage-1 candidates; quantum kernels (checks, exact, shot noise), stage-2 grids and learning curve, kernel diagnostics, other representations; classical baselines, statistics, costs; hardware-noise study; crater-parameter regression; Fig. J1 | GPU + 16 CPU cores | ~12 h (noise study ~4 h, baselines ~2 h) |

Each block runs its steps one after the other, each as a separate Python process. It stops at the first failed
step and prints how to resume from that step. Options, the same for the three blocks:

- `--device cuda|cpu`: device of the torch steps (default: `cuda` if available). Without a GPU, blocks 2-3 also
  run on CPU, but much more slowly.
- `--list`: print the steps of the block (number, plan step, script and command) and exit.
- `--from N` / `--to N`: run only steps N to N (e.g. `--from 12` resumes after a failure at step 12).
- `--dry-run`: print the commands without running them.

The blocks can be re-run safely. A raster that already exists is not rewritten, and the long steps keep their
progress and resume: feature extraction, fusion training, the stage-2 grids, the classical baselines and the
noise study. Every step prints the files it wrote. It also writes its numbers in the section of its plan step
(e.g. `QC4.1`) of the JSON report of its phase in `Experiments/qcai2027/results/` (see `results/INDEX.md`).

On a cluster, run each block inside one job of your scheduler. The long CPU steps use the threads given by
`OMP_NUM_THREADS`.

A single step can still be run alone, e.g. to redo it with different options. The command is the one shown by
`--list`, as `python Experiments/qcai2027/<script>.py <command> [options]`. Every script has `--help`, most have
`--selftest` (unit checks on synthetic data, ~1-2 min) and many have `--smoke` (the step on a small subset, to
test the code before a long run).

## Folders

| Folder | Content |
|---|---|
| `Experiments/qcai2027/` | the three block runners (`run_preprocessing.py`, `run_representations.py`, `run_detection.py`; the ordered steps of each block are in `qc_pipeline.py`), the code of the steps (`qc_*.py`) and the configuration (`qc_config.yaml`) |
| `Experiments/qcai2027/results/` | one JSON report per phase of the plan (one section per step, with the command, time and configuration MD5 that produced it), the CSV of the runs (`qi4_runs.csv`, `qi6_runs.csv`, `qj1_runs.csv`, `qj2_runs.csv`, `qi7_runs.csv` and their CV tables), `INDEX.md` (one line per result file and its producing command) |
| `Experiments/qcai2027/results/figures/` | check figures (labels and split, input tiles, token alignment) |
| `Experiments/qcai2027/results/paper/` | figures and tables of the paper (region map, size-frequency distribution, example tiles, dataset table, modality complementarity, quantum vs classical), plus hand-written text drafts (`qj5_related_work.tex`, `qj5_references.bib`) |
| `Dataset/Moon/LROC/`, `Dataset/Moon/LOLA/`, `Dataset/Moon/catalogs/robbins2019/` | raw data (not versioned): only their `downloads.txt` (and the catalogue's `SOURCE.md`) are in the repository, see Data |
| `Dataset/scripts/` | dataset tooling of the larger project (readers, DEM and co-registration helpers, downloader), see Data |
| `Dataset/Moon/preprocessing/sources/`, `Dataset/Moon/preprocessing/diagnostics/B4/` | inventory and diagnostics tables of the six products used (input of block 1) |
| `Dataset/Moon/preprocessing/clean_dataset/qcai2027_R/` | pre-processed rasters of R on the WAC grid, memory-mapped `.npy` with a `.json` sidecar each: `LROC/WAC_GLOBAL_100M/wac.npy`, `LOLA/SLDEM2015_512/dem_height_m.npy`, `dem_relief_m.npy`, `dem_slope_deg.npy`, `dem_hillshade.npy`, and `normalisation.json` (written by steps 8-14; originals untouched) |
| `Dataset/Moon/qcai2027/` | data produced by the experiment: crater labels, split, tile list and validity, token and fused-map caches (`tokens/`), trained fusion models (`fusion/`), probe scores (`probes/`), stage-1 heads and candidates (`stage1/`), quantum-stage inputs (`qk/`), stage-2 test scores (`stage2/`), noisy kernels (`noise/`) |
