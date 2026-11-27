#!/usr/bin/env python3
"""Checks of the backbone weights and of the input data of the QCAI 2027 experiment (plan phase QA).

Every check writes its section of ``results/qa_checks.json`` (``qc_data.save_report``), so that the statements of
the plan about the backbone weights and the input data can be reproduced.
(Package versions and hardware are not recorded here.)

Commands
--------
dinov2   QA3.3: the DINOv2 ViT-S/14 weights of ``timm`` (load, size, token grid
         for a 224 px input, normalisation of the pre-trained config), which fix
         the 16 x 16 px cells of QB2.2 and the normalisation of QC5.1.
         -> section QA3.3
data     QA2.1 + QA2.2: valid coverage of the products of R (from the inventory
         diagnostics ``data_valid_coverage.csv``) and integrity of the copies in
         ``Dataset/Moon`` against the full-project dataset (size, and MD5 unless
         ``--no-md5``); without the full-project dataset (``paths.source_root``
         missing from the config or not on disk) the files are only listed with
         their size and MD5.  -> section QA2
catalog  QA3.1: Robbins catalogue: MD5, columns, ids, missing values, value
         ranges (units).  -> section QA3.1

Usage
-----
    python Experiments/qcai2027/qc_checks.py dinov2 | data [--no-md5] | catalog
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_data import load_config, read_inventory, save_report, CONFIG_PATH  # noqa: E402

DINO_MODEL = "vit_small_patch14_dinov2.lvd142m"


def md5sum(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# dinov2 (QA3.3)
# ---------------------------------------------------------------------------


def check_dinov2(cfg: dict) -> dict:
    import timm
    import torch
    from huggingface_hub import constants

    m = timm.create_model(DINO_MODEL, pretrained=True, img_size=224).eval()
    with torch.no_grad():
        tokens = m.forward_features(torch.zeros(1, 3, 224, 224))
    out = {
        "model": DINO_MODEL, "hf_hub_id": m.pretrained_cfg.get("hf_hub_id"), "cache": constants.HF_HUB_CACHE,
        "params_millions": round(sum(p.numel() for p in m.parameters()) / 1e6, 3),
        "patch_size": list(m.patch_embed.patch_size), "embed_dim": m.embed_dim,
        "tokens_for_224px": list(tokens.shape), "prefix_tokens": m.num_prefix_tokens,
        "patch_token_grid": int((tokens.shape[1] - m.num_prefix_tokens) ** 0.5),
        "mean": list(m.pretrained_cfg.get("mean")), "std": list(m.pretrained_cfg.get("std")),
    }
    save_report(cfg, "QA3.3", out)
    return out


# ---------------------------------------------------------------------------
# data (QA2.1, QA2.2)
# ---------------------------------------------------------------------------


def check_data(cfg: dict, with_md5: bool = True) -> dict:
    inventory = read_inventory(cfg)
    reg = cfg["region"]
    products = [pid for mod in cfg["products"].values() for pid in mod]

    # QA2.1: valid fraction per 5 deg latitude band inside R, from the dataset diagnostics
    coverage = {pid: [] for pid in products}
    with open(cfg["paths"]["valid_coverage"], newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["product_id"] in coverage:
                lo, hi = float(row["band_min_lat"]), float(row["band_max_lat"])
                if lo >= reg["lat_min"] and hi <= reg["lat_max"]:
                    coverage[row["product_id"]].append({"band": [lo, hi], "valid_fraction": float(row["valid_fraction"])})
    cov_summary = {pid: {"bands": len(v), "min_valid_fraction": min((b["valid_fraction"] for b in v), default=None)}
                   for pid, v in coverage.items()}

    # QA2.2: every file of the products (data, label and the PDS4 .xml next to them) against the original
    source = cfg["paths"].get("source_root")
    source = source if source is not None and source.is_dir() else None     # full-project dataset available
    files = set()
    for pid in products:
        row = inventory[pid]
        for key in ("data_file", "label_file"):
            files.add(row[key])
        stem = Path(row["data_file"])
        xml = stem.with_suffix(".xml")
        if ((source or cfg["paths"]["dataset_root"]) / xml).exists():
            files.add(str(xml))
    copies = []
    for rel in sorted(files):
        a = cfg["paths"]["dataset_root"] / rel
        rec = {"file": rel, "size": a.stat().st_size}
        if source is not None:
            rec["size_original"] = (source / rel).stat().st_size
            rec["size_ok"] = rec["size"] == rec["size_original"]
        if with_md5:
            rec["md5"] = md5sum(a)
            if source is not None:
                rec["md5_original"] = md5sum(source / rel)
                rec["md5_ok"] = rec["md5"] == rec["md5_original"]
        copies.append(rec)
    out = {
        "region": reg, "coverage_source": str(cfg["paths"]["valid_coverage"]),
        "coverage_summary": cov_summary, "coverage": coverage,
        "all_bands_fully_valid": all(s["bands"] > 0 and s["min_valid_fraction"] == 1.0 for s in cov_summary.values()),
        "copies": copies, "n_files": len(copies),
        "all_copies_identical": (all(c["size_ok"] and c.get("md5_ok", True) for c in copies)
                                 if source is not None else None),           # None: no originals to compare with
        "md5_checked": with_md5,
    }
    save_report(cfg, "QA2", out)
    return out


# ---------------------------------------------------------------------------
# catalog (QA3.1)
# ---------------------------------------------------------------------------


def check_catalog(cfg: dict) -> dict:
    import pandas as pd

    path = cfg["paths"]["catalog"]
    cat = pd.read_csv(path)
    expected = ["LAT_CIRC_IMG", "LON_CIRC_IMG", "DIAM_CIRC_IMG", "CRATER_ID"]
    num = cat[["LAT_CIRC_IMG", "LON_CIRC_IMG", "DIAM_CIRC_IMG"]]
    out = {
        "file": str(path),
        "md5": md5sum(path), "md5_expected": cfg["catalog_md5"],
        "rows": int(len(cat)), "columns": list(cat.columns),
        "expected_columns_present": all(c in cat.columns for c in expected),
        "crater_id_unique": bool(cat["CRATER_ID"].is_unique),
        "missing_values": int(cat[expected].isna().sum().sum()),
        "ranges": {c: [float(num[c].min()), float(num[c].max())] for c in num.columns},
    }
    out["md5_ok"] = out["md5"] == out["md5_expected"]
    r = out["ranges"]
    # units: lat in degrees, lon east 0-360, diameter in km (catalogue limit 1 km; largest basin ~2,500 km)
    out["units_consistent"] = (-90 <= r["LAT_CIRC_IMG"][0] and r["LAT_CIRC_IMG"][1] <= 90
                               and 0 <= r["LON_CIRC_IMG"][0] and r["LON_CIRC_IMG"][1] <= 360
                               and abs(r["DIAM_CIRC_IMG"][0] - 1.0) < 1e-6 and 2000 < r["DIAM_CIRC_IMG"][1] < 3000)
    save_report(cfg, "QA3.1", out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["dinov2", "data", "catalog"])
    ap.add_argument("--no-md5", action="store_true", help="data: compare sizes only")
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "dinov2":
        res = check_dinov2(cfg)
    elif args.command == "data":
        res = check_data(cfg, with_md5=not args.no_md5)
        res = {k: v for k, v in res.items() if k not in ("coverage", "copies")} | {
            "copies": [{k: c[k] for k in c if not k.startswith("md5")} | {"md5_ok": c.get("md5_ok")} for c in res["copies"]]}
    else:
        res = check_catalog(cfg)
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
