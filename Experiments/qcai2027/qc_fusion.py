#!/usr/bin/env python3
"""Fusion variants, training, probes and stage 1 of the QCAI 2027 experiment (plan phases QF-QH).

QF1 - Variants.  Every representation compared in Table I gives one vector per
sub-tile, i.e. a 16 x 16 x D map per tile (token (i, j) = sub-tile (i, j) of the
tile, QE1.2).  ``VARIANTS`` lists them; the untrained ones are read from the token
cache of QE2 (``qc_features.load_tokens``), the trained ones from the fused maps of QF4.3:

    M-W   WAC tokens                      384   (cache ``wac``)
    M-D   DEM tokens                      384   (cache ``dem``)
    F-E   early fusion, one backbone pass 384   (cache ``early``)
    F-C   concatenation [WAC, DEM]        768
    F-CM  concatenation + MLP projector   512   trained (QF3.4), cache ``fcm`` (QF4.3)
    F-X   cross-attention fusion          512   trained (QF2-QF3.4), cache ``fx`` (QF4.3)

``variant_maps`` gives the maps of some tiles; ``owned_samples`` the vectors of the
sub-tiles owned by the tiles of one set (QC7.2: the central 11 x 11 of each tile),
with their labels (QB2.3), i.e. the samples of the probes (QG) and of stage 1 (QH).

Commands
--------
variants     QF1 check -> report section QF1 of ``results/qf_fusion.json``: per
             set, the owned sub-tiles and labels gathered from the token cache
             equal the dataset counts of QC8.2; per untrained variant, dimension
             and, on random tiles, values equal to the cached tokens (F-C = the two
             caches side by side).

model        QF2 check -> report section QF2: parameters of F-X (``build_fusion_x``,
             hyper-parameters in ``fusion`` of ``qc_config.yaml``) and its forward pass
             on real tokens (untrained, CPU): output 16 x 16 x 512, finite; the DEM
             tokens change the WAC stream (cross-attention); a fully masked modality
             ignores its values (mask token); the 2-D position matters; every
             parameter gets a gradient.

masking      QF3.1 check -> report section QF3.1: statistics of the random masks of
             ``sample_masks`` (ratio in ``fusion.mask_ratio``, per-position frequency,
             independence of the two modalities, reproducibility) and, on real tokens,
             the masked values do not reach the output of F-X, the visible ones do.
             QF3.2 check -> report section QF3.2: modality drop-out (one whole
             modality masked with probability ``fusion.modality_dropout``, the other
             fully visible), its frequency and balance, the tiles without drop-out
             unchanged, F-X ignores the dropped modality.

decoder      QF3.3 check -> report section QF3.3: decoder (``build_decoder``) and loss
             (``reconstruction_loss``) on real tokens: shapes, layer-normalised targets,
             loss on the masked positions only and equal to a direct computation,
             gradients of F-X and decoder, a short fit on 8 tiles lowers the loss.

train        QF3.4 (GPU) -> ``Dataset/Moon/qcai2027/fusion/<model>_lr<lr>_s<seed>.pt``
             and report section QF3.4: self-supervised training of F-X and F-CM
             (``train_fusion``, recipe in ``fusion.train``) on the S1-train tiles,
             early stopping on S1-val; per model the lr is chosen with the first seed,
             then the other seeds at that lr.  Resumable.  ``--smoke``: quick test.
             ``--sweep``: QF3.5 [opt], exploratory recipe of ``fusion.train_sweep``
             (higher lr, more epochs) -> ``fusion/sweep/`` and report section QF3.5
             with the comparison to QF3.4 and the switch rule (fixed in advance).

checks       QF4.1 -> report section QF4.1: S1-val reconstruction loss of F-X and F-CM
             (3 seeds) with the masks of QF3.4, with one modality fully masked (rebuilt
             from the other one), with both masked.  QF4.2 -> section QF4.2 and
             ``figures/qf_attention.png``: spread of the fused maps (not constant), the
             cross-attention of F-X on the crater cells (trained vs untrained).

cache        QF4.3 -> ``Dataset/Moon/qcai2027/tokens/fx.npy``, ``fcm.npy`` and report
             section QF4.3: fused maps of the downstream F-X and F-CM for every used
             tile (resumable), read by ``variant_maps`` like the other variants.

QF2 - F-X.  ``build_fusion_x``: per modality a linear projection 384 -> 256, a
learned mask token (QF3) placed after the projection, a learned modality embedding
and a fixed 2-D sine-cosine positional embedding; 2 fusion blocks (per stream:
cross-attention to the other stream, self-attention, FFN with ratio 4; pre-norm,
residual; 4 heads); final norm per stream, concatenation, linear layer to 512.

Usage
-----
    python Experiments/qcai2027/qc_fusion.py variants
    python Experiments/qcai2027/qc_fusion.py model
    python Experiments/qcai2027/qc_fusion.py masking
    python Experiments/qcai2027/qc_fusion.py decoder
    python Experiments/qcai2027/qc_fusion.py train --device cuda      (GPU job: run_qcai_gpu.sh)
    python Experiments/qcai2027/qc_fusion.py checks                   (CPU job: run_qcai_cpu.sh)
    python Experiments/qcai2027/qc_fusion.py cache                    (CPU job: run_qcai_cpu.sh)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_data import CONFIG_PATH, SETS, load_config, load_report, save_report, written  # noqa: E402


# ---------------------------------------------------------------------------
# QF1 - Variants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    vid: str
    description: str
    dim: int
    caches: tuple[str, ...] = ()       # token caches of QE2 concatenated along the channels (untrained)
    trained: bool = False              # produced by a fusion model trained in QF3.4 (cache written by QF4.3)


VARIANTS = {v.vid: v for v in (
    Variant("M-W", "WAC tokens only", 384, ("wac",)),
    Variant("M-D", "DEM tokens only", 384, ("dem",)),
    Variant("F-E", "early fusion: one backbone pass on [WAC, relief, slope]", 384, ("early",)),
    Variant("F-C", "concatenation of the WAC and DEM tokens", 768, ("wac", "dem")),
    Variant("F-CM", "concatenation + per-token MLP projector, self-supervised (QF3)", 512, ("fcm",), trained=True),
    Variant("F-X", "bidirectional cross-attention fusion, self-supervised (QF2-QF3)", 512, ("fx",), trained=True),
)}


class TokenStore:
    """The token caches of QE2 (memory-mapped) and their index, opened once."""

    def __init__(self, cfg: dict):
        from qc_features import load_tokens

        self.caches, idx = {}, None
        for name in ("wac", "dem", "early", "fx", "fcm"):
            try:
                self.caches[name], idx_n = load_tokens(cfg, name)
            except (ValueError, FileNotFoundError):
                if name in ("fx", "fcm"):                                 # fused maps of QF4.3, optional
                    continue
                raise
            if idx is not None and not np.array_equal(idx["tile"], idx_n["tile"]):
                raise ValueError("the token caches have different indices")
            idx = idx_n
        self.index = idx
        self.row_of_tile = {int(t): k for k, t in enumerate(idx["tile"])}


def variant_maps(store: TokenStore, vid: str, rows) -> np.ndarray:
    """(len(rows), 16, 16, D) float32 maps of variant ``vid`` for the rows ``rows`` of the token cache."""
    v = VARIANTS[vid]
    if any(c not in store.caches for c in v.caches):
        raise ValueError(f"the fused maps of {vid} are not cached: run qc_fusion.py cache (QF4.3)")
    rows = np.asarray(rows)
    order = np.argsort(rows)                                         # memmap reads in increasing order
    out = np.concatenate([np.asarray(store.caches[c][rows[order]], dtype=np.float32) for c in v.caches], axis=-1)
    out[order] = out.copy()
    return out


def owned_index(cfg: dict, store: TokenStore, set_name: str, max_tiles: int | None = None) -> dict:
    """Owned sub-tiles of the tiles of ``set_name`` in the token cache: per sample the cache row,
    the local token position (i, j), the cell of R (ci, cj) and its labels ``cls`` / ``band`` (QB2.3)."""
    from qc_tiles import owned_cells

    tile_px, stride_px, cell_px = (int(cfg["tiling"][k]) for k in ("tile_px", "stride_px", "cell_px"))
    lab = np.load(cfg["paths"]["output"] / "cell_labels.npz")
    rows = np.flatnonzero(store.index["set"] == set_name)[:max_tiles]
    (lr, lc), _ = owned_cells(0, 0, tile_px, stride_px, cell_px)
    li, lj = (a.ravel() for a in np.mgrid[lr, lc])                    # same local window for every tile
    k = stride_px // cell_px
    ci = (store.index["ti"][rows, None] * k + li[None]).ravel()       # cell of R = tile origin + local
    cj = (store.index["tj"][rows, None] * k + lj[None]).ravel()
    return {"row": np.repeat(rows, li.size), "i": np.tile(li, rows.size), "j": np.tile(lj, rows.size),
            "ci": ci, "cj": cj, "cls": lab["cls"][ci, cj], "band": lab["band"][ci, cj]}


def owned_samples(cfg: dict, store: TokenStore, vid: str, set_name: str, chunk: int = 512,
                  max_tiles: int | None = None) -> tuple[np.ndarray, dict]:
    """Vectors (n, D) float32 of variant ``vid`` for the owned sub-tiles of ``set_name`` and their
    index (``owned_index``); read in chunks of ``chunk`` tiles (``max_tiles``: only the first tiles)."""
    idx = owned_index(cfg, store, set_name, max_tiles)
    rows = np.unique(idx["row"])
    per_tile = idx["row"].size // rows.size
    x = np.empty((idx["row"].size, VARIANTS[vid].dim), dtype=np.float32)
    for s in range(0, rows.size, chunk):
        m = variant_maps(store, vid, rows[s:s + chunk])
        sl = slice(s * per_tile, (s + rows[s:s + chunk].size) * per_tile)
        x[sl] = m[np.repeat(np.arange(m.shape[0]), per_tile), idx["i"][sl], idx["j"][sl]]
    return x, idx


def run_variants_check(cfg: dict, n_tiles: int = 32) -> dict:
    """QF1: the samples of every set equal the dataset of QC8.2; untrained variants equal the caches."""
    store = TokenStore(cfg)
    qc82 = {r["set"]: r for r in load_report(cfg, "QC8.2")["per_set"]}
    split = np.load(cfg["paths"]["output"] / "split_cells.npz")
    sets = {}
    for s in SETS:
        idx = owned_index(cfg, store, s)
        ref = qc82[s]
        own_ok = bool((split["owner_tile"][idx["ci"], idx["cj"]] == store.index["tile"][idx["row"]]).all())
        sets[s] = {"tiles": int(np.unique(idx["row"]).size), "subtiles": int(idx["row"].size),
                   "positive": int((idx["cls"] == 1).sum()), "negative": int((idx["cls"] == 0).sum()),
                   "ignore": int((idx["cls"] == -1).sum()),
                   "positive_per_band_1_2_4_8_km": [int((idx["band"][idx["cls"] == 1] == b).sum()) for b in range(3)],
                   "check_owner_of_every_cell_is_its_tile": own_ok,
                   "check_equal_qc82": bool(np.unique(idx["row"]).size == ref["tiles_used"] and idx["row"].size == ref["subtiles"]
                                            and (idx["cls"] == 1).sum() == ref["positive"]
                                            and (idx["cls"] == 0).sum() == ref["negative"]
                                            and (idx["cls"] == -1).sum() == ref["ignore"])}
    rng = np.random.default_rng(int(cfg["seed"]))
    rows = np.sort(rng.choice(len(store.index["tile"]), n_tiles, replace=False))
    variants = {}
    for vid, v in VARIANTS.items():
        r = {"description": v.description, "dim": v.dim, "trained": v.trained}
        if not v.trained:
            m = variant_maps(store, vid, rows[::-1])[::-1]               # unsorted rows: order kept
            ref = np.concatenate([np.asarray(store.caches[c][rows], dtype=np.float32) for c in v.caches], axis=-1)
            r.update({"caches": list(v.caches), "shape_per_tile": list(m.shape[1:]),
                      "check_equal_to_caches": bool(np.array_equal(m, ref)), "check_finite": bool(np.isfinite(m).all())})
        else:
            r["status"] = "trained in QF2-QF3"
        variants[vid] = r
    # owned_samples against a direct read of the cache, on the first tiles of S1-val
    x, idx = owned_samples(cfg, store, "F-C", "s1_val", chunk=7, max_tiles=20)
    sel = np.arange(idx["row"].size)
    direct = np.concatenate([np.asarray(store.caches[c][idx["row"][sel], idx["i"][sel], idx["j"][sel]], dtype=np.float32)
                             for c in ("wac", "dem")], axis=-1)
    report = {"per_set": sets, "variants": variants, "check_tiles": n_tiles,
              "check_owned_samples_equal_direct_read_f_c_20_tiles": bool(np.array_equal(x[sel], direct))}
    save_report(cfg, "QF1", report)
    return report


# ---------------------------------------------------------------------------
# QF2 - Cross-attention fusion F-X
# ---------------------------------------------------------------------------

MODALITIES = ("wac", "dem")


def sincos_2d(grid: int, dim: int):
    """Fixed 2-D sine-cosine positional embedding (grid * grid, dim), MAE style: half of the
    channels encode the row, half the column; token k = row k // grid, column k % grid."""
    import torch

    if dim % 4:
        raise ValueError("dim must be a multiple of 4")
    omega = 1.0 / 10000 ** (torch.arange(dim // 4, dtype=torch.float64) / (dim // 4))
    r, c = torch.meshgrid(torch.arange(grid, dtype=torch.float64), torch.arange(grid, dtype=torch.float64), indexing="ij")
    parts = [f(x.reshape(-1, 1) * omega[None]) for x in (r, c) for f in (torch.sin, torch.cos)]
    return torch.cat(parts, dim=1).float()


def _fusion_modules():
    """The modules of F-X (built on first use, so that the rest of the script does not need torch)."""
    import torch
    from torch import nn

    class Attention(nn.Module):
        """Pre-norm residual attention: x + MHA(norm(x), norm_kv(kv), norm_kv(kv))."""

        def __init__(self, d: int, heads: int):
            super().__init__()
            self.norm_q, self.norm_kv = nn.LayerNorm(d), nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)

        def forward(self, x, kv, need_weights: bool = False):
            k = self.norm_kv(kv)
            y, w = self.attn(self.norm_q(x), k, k, need_weights=need_weights, average_attn_weights=False)
            return x + y, w

    class FeedForward(nn.Module):
        def __init__(self, d: int, ratio: int):
            super().__init__()
            self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ratio * d), nn.GELU(), nn.Linear(ratio * d, d))

        def forward(self, x):
            return x + self.net(x)

    class FusionBlock(nn.Module):
        """QF2.2: each stream attends to the other one (cross-attention, both computed from the input
        state of the block, so the update is symmetric), then to itself (self-attention), then FFN."""

        def __init__(self, d: int, heads: int, ffn_ratio: int):
            super().__init__()
            self.cross = nn.ModuleDict({m: Attention(d, heads) for m in MODALITIES})
            self.self_attn = nn.ModuleDict({m: Attention(d, heads) for m in MODALITIES})
            self.ffn = nn.ModuleDict({m: FeedForward(d, ffn_ratio) for m in MODALITIES})

        def forward(self, x: dict, need_weights: bool = False):
            other = {"wac": "dem", "dem": "wac"}
            out, attn = {}, {}
            for m in MODALITIES:
                y, attn[m] = self.cross[m](x[m], x[other[m]], need_weights)
                out[m] = y
            for m in MODALITIES:
                out[m] = self.ffn[m](self.self_attn[m](out[m], out[m])[0])
            return out, attn

    class FusionX(nn.Module):
        """F-X (QF2): two 16 x 16 x 384 token maps (WAC, DEM) -> 16 x 16 x 512 fused map.
        QF2.1 linear projection to d per modality; masked positions (QF3) replaced by a learned mask
        token per modality after the projection; + learned modality embedding + fixed 2-D sine-cosine
        positional embedding.  QF2.2 ``blocks`` fusion blocks.  QF2.3 final norm per stream,
        concatenation, linear layer to ``out_dim``."""

        def __init__(self, in_dim: int = 384, d: int = 256, heads: int = 4, blocks: int = 2, ffn_ratio: int = 4,
                     out_dim: int = 512, grid: int = 16):
            super().__init__()
            self.grid, self.d = grid, d
            self.proj = nn.ModuleDict({m: nn.Linear(in_dim, d) for m in MODALITIES})
            self.mask_token = nn.ParameterDict({m: nn.Parameter(torch.zeros(d)) for m in MODALITIES})
            self.modality_emb = nn.ParameterDict({m: nn.Parameter(torch.zeros(d)) for m in MODALITIES})
            self.register_buffer("pos", sincos_2d(grid, d), persistent=False)
            self.blocks = nn.ModuleList(FusionBlock(d, heads, ffn_ratio) for _ in range(blocks))
            self.norm = nn.ModuleDict({m: nn.LayerNorm(d) for m in MODALITIES})
            self.out = nn.Linear(2 * d, out_dim)
            for p in list(self.mask_token.values()) + list(self.modality_emb.values()):
                nn.init.trunc_normal_(p, std=0.02)
            for mod in self.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.xavier_uniform_(mod.weight)
                    nn.init.zeros_(mod.bias)

        def encode(self, tokens: dict, masks: dict | None = None, need_weights: bool = False):
            """tokens {modality: (B, 16, 16, in_dim) or (B, 256, in_dim)}, masks {modality: (B, 256) bool,
            True = masked} -> (streams {modality: (B, 256, d)} after the final norm, fused (B, 256, out_dim),
            cross-attention weights per block if ``need_weights``)."""
            x = {}
            for m in MODALITIES:
                t = tokens[m].reshape(tokens[m].shape[0], -1, tokens[m].shape[-1]).float()
                h = self.proj[m](t)
                if masks is not None and masks.get(m) is not None:
                    h = torch.where(masks[m][..., None], self.mask_token[m].to(h.dtype), h)
                x[m] = h + self.modality_emb[m] + self.pos
            weights = []
            for blk in self.blocks:
                x, w = blk(x, need_weights)
                weights.append(w)
            x = {m: self.norm[m](x[m]) for m in MODALITIES}
            return x, self.out(torch.cat([x["wac"], x["dem"]], dim=-1)), weights

        def forward(self, tokens: dict, masks: dict | None = None):
            """Fused map (B, 16, 16, out_dim)."""
            fused = self.encode(tokens, masks)[1]
            return fused.reshape(fused.shape[0], self.grid, self.grid, -1)

    class Decoder(nn.Module):
        """QF3.3: light decoder of the self-supervised objective (discarded after training).  Fused map
        (B, 16, 16, in_dim) -> linear to d + fixed 2-D sine-cosine position -> one pre-norm transformer
        layer (self-attention + FFN) -> norm -> one linear head per modality predicting its tokens."""

        def __init__(self, in_dim: int = 512, d: int = 256, heads: int = 4, ffn_ratio: int = 4, out_dim: int = 384,
                     grid: int = 16):
            super().__init__()
            self.inp = nn.Linear(in_dim, d)
            self.register_buffer("pos", sincos_2d(grid, d), persistent=False)
            self.attn, self.ffn, self.norm = Attention(d, heads), FeedForward(d, ffn_ratio), nn.LayerNorm(d)
            self.heads = nn.ModuleDict({m: nn.Linear(d, out_dim) for m in MODALITIES})
            for mod in self.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.xavier_uniform_(mod.weight)
                    nn.init.zeros_(mod.bias)

        def forward(self, fused):
            """fused (B, 16, 16, in_dim) or (B, 256, in_dim) -> {modality: (B, 256, out_dim)}."""
            x = self.inp(fused.reshape(fused.shape[0], -1, fused.shape[-1])) + self.pos
            x = self.norm(self.ffn(self.attn(x, x)[0]))
            return {m: self.heads[m](x) for m in MODALITIES}

    class FusionCM(nn.Module):
        """F-CM (QF1, control of F-X): per token, masked positions replaced by a learned mask token per
        modality (input space, QF3.1), concatenation [WAC, DEM] -> MLP (Linear, GELU, Linear) -> out_dim.
        No attention: every position is fused alone."""

        def __init__(self, in_dim: int = 384, hidden: int = 512, out_dim: int = 512, grid: int = 16):
            super().__init__()
            self.grid = grid
            self.mask_token = nn.ParameterDict({m: nn.Parameter(torch.zeros(in_dim)) for m in MODALITIES})
            self.mlp = nn.Sequential(nn.Linear(2 * in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
            for p in self.mask_token.values():
                nn.init.trunc_normal_(p, std=0.02)
            for mod in self.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.xavier_uniform_(mod.weight)
                    nn.init.zeros_(mod.bias)

        def forward(self, tokens: dict, masks: dict | None = None):
            """Same interface as ``FusionX.forward``: fused map (B, 16, 16, out_dim)."""
            x = []
            for m in MODALITIES:
                t = tokens[m].reshape(tokens[m].shape[0], -1, tokens[m].shape[-1]).float()
                if masks is not None and masks.get(m) is not None:
                    t = torch.where(masks[m][..., None], self.mask_token[m].to(t.dtype), t)
                x.append(t)
            out = self.mlp(torch.cat(x, dim=-1))
            return out.reshape(out.shape[0], self.grid, self.grid, -1)

    return {"FusionX": FusionX, "Decoder": Decoder, "FusionCM": FusionCM}


def build_fusion_x(cfg: dict):
    """F-X with the hyper-parameters of ``fusion`` in ``qc_config.yaml`` (QD4, QF2)."""
    f = cfg["fusion"]
    return _fusion_modules()["FusionX"](in_dim=int(f["in_dim"]), d=int(f["d_model"]), heads=int(f["heads"]), blocks=int(f["blocks"]),
                             ffn_ratio=int(f["ffn_ratio"]), out_dim=int(f["out_dim"]), grid=int(cfg["tiling"]["tile_px"]) // int(cfg["tiling"]["cell_px"]))


def build_fusion(cfg: dict, vid: str):
    """Trained fusion ``vid`` (``F-X`` or ``F-CM``) with the hyper-parameters of ``fusion`` in ``qc_config.yaml``."""
    if vid == "F-X":
        return build_fusion_x(cfg)
    if vid == "F-CM":
        f = cfg["fusion"]
        return _fusion_modules()["FusionCM"](in_dim=int(f["in_dim"]), hidden=int(f["cm_hidden"]), out_dim=int(f["out_dim"]),
                                             grid=int(cfg["tiling"]["tile_px"]) // int(cfg["tiling"]["cell_px"]))
    raise ValueError(f"{vid} is not a trained fusion")


def build_decoder(cfg: dict):
    """Decoder of QF3.3 with the hyper-parameters of ``fusion.decoder`` in ``qc_config.yaml``."""
    f, d = cfg["fusion"], cfg["fusion"]["decoder"]
    return _fusion_modules()["Decoder"](in_dim=int(f["out_dim"]), d=int(d["d_model"]), heads=int(d["heads"]),
                                        ffn_ratio=int(d["ffn_ratio"]), out_dim=int(f["in_dim"]),
                                        grid=int(cfg["tiling"]["tile_px"]) // int(cfg["tiling"]["cell_px"]))


def reconstruction_loss(pred: dict, tokens: dict, masks: dict):
    """QF3.3: MSE between the predicted and the original tokens, on the masked positions only.  The
    targets are the frozen tokens layer-normalised per token (no affine).  Per modality the mean over
    its masked tokens in the batch; the loss is the average of the two modalities.
    Returns (loss, {modality: loss term})."""
    import torch.nn.functional as F

    per = {}
    for m in MODALITIES:
        t = tokens[m].reshape(tokens[m].shape[0], -1, tokens[m].shape[-1]).float()
        err = (pred[m] - F.layer_norm(t, t.shape[-1:])).pow(2).mean(-1)
        w = masks[m].to(err.dtype)
        per[m] = (err * w).sum() / w.sum().clamp(min=1)
    return sum(per.values()) / len(per), per


def run_model_check(cfg: dict, n_tiles: int = 8) -> dict:
    """QF2: parameters of F-X and checks of its forward pass on real tokens (CPU, untrained model)."""
    import time

    import torch

    torch.manual_seed(int(cfg["seed"]))
    model = build_fusion_x(cfg).eval()
    store = TokenStore(cfg)
    rows = np.sort(np.random.default_rng(int(cfg["seed"])).choice(len(store.index["tile"]), n_tiles, replace=False))
    tok = {m: torch.from_numpy(np.asarray(store.caches[m][rows], dtype=np.float32)) for m in MODALITIES}
    n_tok = model.grid ** 2
    count = lambda mod: int(sum(p.numel() for p in mod.parameters()))
    params = {"total": count(model), "projections": count(model.proj), "blocks": count(model.blocks),
              "per_block_cross_attention": count(model.blocks[0].cross), "per_block_self_attention": count(model.blocks[0].self_attn),
              "per_block_ffn": count(model.blocks[0].ffn), "output": count(model.out),
              "embeddings_and_norms": count(model.mask_token) + count(model.modality_emb) + count(model.norm)}
    with torch.no_grad():
        streams, fused, w = model.encode(tok, need_weights=True)
        out = model(tok)
        # (1) cross-modal: changing the DEM tokens changes the WAC stream (and the reverse)
        tok_d = {"wac": tok["wac"], "dem": tok["dem"].roll(1, dims=0)}
        s_d = model.encode(tok_d)[0]
        # (2) mask token: with the WAC fully masked, the WAC values do not matter
        all_w = {"wac": torch.ones(n_tiles, n_tok, dtype=torch.bool), "dem": None}
        a = model(tok, all_w)
        b = model({"wac": torch.randn_like(tok["wac"]), "dem": tok["dem"]}, all_w)
        # (3) position: a spatially shuffled input does not give the same shuffled output
        perm = torch.randperm(n_tok)
        flat = {m: tok[m].reshape(n_tiles, n_tok, -1) for m in MODALITIES}
        shuf = model({m: flat[m][:, perm] for m in MODALITIES}).reshape(n_tiles, n_tok, -1)
    # (4) every parameter gets a gradient (with masks on half of the positions)
    model.train()
    masks = {m: torch.rand(n_tiles, n_tok) < 0.5 for m in MODALITIES}
    model(tok, masks).pow(2).mean().backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
    model.eval()
    big = {m: tok[m].repeat(128 // n_tiles, 1, 1, 1) for m in MODALITIES}
    t0 = time.perf_counter()
    with torch.no_grad():
        model(big)
    report = {
        "hyper_parameters": cfg["fusion"], "parameters": params,
        "check_tiles": [int(t) for t in store.index["tile"][rows]],
        "output_shape": list(out.shape), "check_finite": bool(torch.isfinite(out).all()),
        # encode(need_weights=True) takes another attention kernel than forward: equal up to rounding
        "max_abs_diff_forward_vs_encode_with_weights": float(f"{float((out.reshape(n_tiles, n_tok, -1) - fused).abs().max()):.2e}"),
        "check_forward_equals_encode": bool(torch.allclose(out.reshape(n_tiles, n_tok, -1), fused, atol=1e-4)),
        "cross_attention_weights_shape": list(w[0]["wac"].shape),
        "check_cross_attention_rows_sum_to_1": bool(torch.allclose(w[0]["wac"].sum(-1), torch.ones(1))),
        "wac_stream_change_when_dem_changes": round(float((s_d["wac"] - streams["wac"]).abs().mean()), 4),
        "check_dem_changes_wac_stream": bool((s_d["wac"] - streams["wac"]).abs().mean() > 1e-3),
        "check_masked_wac_ignores_wac_values": bool(torch.allclose(a, b, atol=1e-5)),
        "check_position_matters": bool(not torch.allclose(shuf, out.reshape(n_tiles, n_tok, -1)[:, perm], atol=1e-3)),
        "parameters_without_gradient": no_grad,
        "seconds_forward_128_tiles_cpu": round(time.perf_counter() - t0, 3),
    }
    save_report(cfg, "QF2", report)
    return report


# ---------------------------------------------------------------------------
# QF3 - Self-supervised objective
# ---------------------------------------------------------------------------


def sample_masks(n: int, n_tok: int, ratio_range, dropout: float = 0.0, generator=None, device="cpu") -> dict:
    """QF3.1: for each of ``n`` tiles and each modality an independent random mask (True = masked):
    ratio r ~ U[lo, hi], then round(r * n_tok) positions drawn uniformly without replacement (MAE style,
    exact count).  The masks of the two modalities are independent, so a position can be visible in
    one modality and masked in the other (cross-modal reconstruction).
    QF3.2 modality drop-out: with probability ``dropout`` a tile has one whole modality masked (WAC or
    DEM, 50/50) and the other one fully visible (rebuild a modality from the other one).  The drop-out
    draws come after the partial masks, so the tiles without drop-out keep the masks of ``dropout=0``."""
    import torch

    lo, hi = (float(v) for v in ratio_range)
    masks = {}
    for m in MODALITIES:
        k = torch.round((lo + (hi - lo) * torch.rand(n, generator=generator, device=device)) * n_tok).long()
        rank = torch.rand(n, n_tok, generator=generator, device=device).argsort(1).argsort(1)
        masks[m] = rank < k[:, None]
    if dropout > 0:
        drop = torch.rand(n, generator=generator, device=device) < dropout
        wac = torch.rand(n, generator=generator, device=device) < 0.5        # which modality is dropped
        for m, d in (("wac", drop & wac), ("dem", drop & ~wac)):
            other = "dem" if m == "wac" else "wac"
            masks[m][d] = True
            masks[other][d] = False
    return masks


def run_masking_check(cfg: dict, n_masks: int = 20000, n_tiles: int = 8) -> dict:
    """QF3.1: statistics of ``sample_masks`` and, on real tokens, the values of the masked tokens do not
    reach the output of F-X (they are replaced by the mask token), those of the visible tokens do."""
    import torch

    lo, hi = (float(v) for v in cfg["fusion"]["mask_ratio"])
    n_tok = (int(cfg["tiling"]["tile_px"]) // int(cfg["tiling"]["cell_px"])) ** 2
    g = lambda: torch.Generator().manual_seed(int(cfg["seed"]))
    masks = sample_masks(n_masks, n_tok, (lo, hi), generator=g())
    again = sample_masks(n_masks, n_tok, (lo, hi), generator=g())
    ratio = {m: masks[m].float().mean(1) for m in MODALITIES}
    k_lo, k_hi = round(lo * n_tok), round(hi * n_tok)
    stats = {}
    for m in MODALITIES:
        r, pos = ratio[m], masks[m].float().mean(0)
        stats[m] = {"ratio_min_mean_max": [round(float(r.min()), 4), round(float(r.mean()), 4), round(float(r.max()), 4)],
                    "ratio_quartiles": [round(float(q), 4) for q in torch.quantile(r, torch.tensor([0.25, 0.5, 0.75]))],
                    "per_position_masked_frequency_min_max": [round(float(pos.min()), 4), round(float(pos.max()), 4)],
                    "check_masked_count_in_range": bool(((r * n_tok).round() >= k_lo).all() and ((r * n_tok).round() <= k_hi).all())}
    both = (masks["wac"] & masks["dem"]).float().mean()
    expected = (ratio["wac"] * ratio["dem"]).mean()
    corr = float(torch.corrcoef(torch.stack([ratio["wac"], ratio["dem"]]))[0, 1])
    # F-X on real tokens: masked values are ignored, visible values are used
    torch.manual_seed(int(cfg["seed"]))
    model = build_fusion_x(cfg).eval()
    store = TokenStore(cfg)
    rows = np.sort(np.random.default_rng(int(cfg["seed"])).choice(len(store.index["tile"]), n_tiles, replace=False))
    tok = {m: torch.from_numpy(np.asarray(store.caches[m][rows], dtype=np.float32)).reshape(n_tiles, n_tok, -1) for m in MODALITIES}
    mk = sample_masks(n_tiles, n_tok, (lo, hi), generator=g())
    noise = {m: torch.randn_like(tok[m]) for m in MODALITIES}
    with torch.no_grad():
        ref = model(tok, mk)
        on_masked = model({m: torch.where(mk[m][..., None], noise[m], tok[m]) for m in MODALITIES}, mk)
        on_visible = model({m: torch.where(mk[m][..., None], tok[m], noise[m]) for m in MODALITIES}, mk)
    report = {
        "mask_ratio_range": [lo, hi], "tokens_per_tile": n_tok, "masked_count_range": [k_lo, k_hi], "masks_sampled": n_masks,
        "per_modality": stats,
        "fraction_masked_in_both_modalities": round(float(both), 4),
        "expected_if_independent": round(float(expected), 4),
        "correlation_of_the_ratios_wac_dem": round(corr, 4),
        "check_modalities_independent": bool(abs(float(both - expected)) < 0.01 and abs(corr) < 0.05),
        "check_reproducible_with_seed": bool(all(torch.equal(masks[m], again[m]) for m in MODALITIES)),
        "check_tiles": [int(t) for t in store.index["tile"][rows]],
        "check_masked_values_do_not_reach_output": bool(torch.allclose(ref, on_masked, atol=1e-5)),
        "check_visible_values_reach_output": bool(not torch.allclose(ref, on_visible, atol=1e-3)),
    }
    save_report(cfg, "QF3.1", report)
    report_d = run_dropout_check(cfg, model, tok, n_masks)
    return {"QF3.1": report, "QF3.2": report_d}


def run_dropout_check(cfg: dict, model, tok: dict, n_masks: int) -> dict:
    """QF3.2: frequency of the modality drop-out of ``sample_masks``, the other modality fully visible,
    tiles without drop-out unchanged, and on real tokens F-X ignores the dropped modality."""
    import torch

    p, ratio = float(cfg["fusion"]["modality_dropout"]), cfg["fusion"]["mask_ratio"]
    n_tiles, n_tok = tok["wac"].shape[:2]
    g = lambda: torch.Generator().manual_seed(int(cfg["seed"]))
    base = sample_masks(n_masks, n_tok, ratio, 0.0, generator=g())
    mk = sample_masks(n_masks, n_tok, ratio, p, generator=g())
    full = {m: mk[m].all(1) for m in MODALITIES}
    none = {m: (~mk[m]).all(1) for m in MODALITIES}
    drop = full["wac"] | full["dem"]
    keep = ~drop
    # F-X with every tile dropped (dropout 1): noise on the dropped modality does not reach the output
    md = sample_masks(n_tiles, n_tok, ratio, 1.0, generator=g())
    noise = {m: torch.where(md[m][..., None], torch.randn_like(tok[m]), tok[m]) for m in MODALITIES}
    with torch.no_grad():
        same = torch.allclose(model(tok, md), model(noise, md), atol=1e-5)
    report = {
        "modality_dropout": p, "masks_sampled": n_masks,
        "fraction_tiles_dropped": round(float(drop.float().mean()), 4),
        "fraction_of_dropped_with_wac_dropped": round(float(full["wac"][drop].float().mean()), 4),
        "check_dropout_frequency": bool(abs(float(drop.float().mean()) - p) < 0.01),
        "check_wac_dem_balanced": bool(abs(float(full["wac"][drop].float().mean()) - 0.5) < 0.03),
        "check_other_modality_fully_visible": bool((none["dem"][full["wac"]]).all() and (none["wac"][full["dem"]]).all()),
        "check_never_both_dropped": bool(not (full["wac"] & full["dem"]).any()),
        "check_tiles_without_dropout_unchanged": bool(all(torch.equal(mk[m][keep], base[m][keep]) for m in MODALITIES)),
        "fraction_of_all_tokens_masked_wac_dem": [round(float(mk[m].float().mean()), 4) for m in MODALITIES],
        "check_dropped_modality_values_do_not_reach_output": bool(same),
    }
    save_report(cfg, "QF3.2", report)
    return report


def run_decoder_check(cfg: dict, n_tiles: int = 8, steps: int = 100) -> dict:
    """QF3.3: decoder and loss on real tokens (CPU): shapes, layer-normalised targets, the loss sees only
    the masked positions and equals a direct NumPy computation, every parameter of F-X and decoder gets
    a gradient, and a short fit on ``n_tiles`` tiles (fixed masks) lowers the loss (learnable objective)."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(int(cfg["seed"]))
    enc, dec = build_fusion_x(cfg), build_decoder(cfg)
    store = TokenStore(cfg)
    rows = np.sort(np.random.default_rng(int(cfg["seed"])).choice(len(store.index["tile"]), n_tiles, replace=False))
    tok = {m: torch.from_numpy(np.asarray(store.caches[m][rows], dtype=np.float32)) for m in MODALITIES}
    n_tok, c = enc.grid ** 2, tok["wac"].shape[-1]
    gen = torch.Generator().manual_seed(int(cfg["seed"]))
    f = cfg["fusion"]
    mk = sample_masks(n_tiles, n_tok, f["mask_ratio"], float(f["modality_dropout"]), generator=gen)
    pred = dec(enc(tok, mk))
    loss, per = reconstruction_loss(pred, tok, mk)
    # direct computation in NumPy
    ref = {}
    for m in MODALITIES:
        t = tok[m].reshape(n_tiles, n_tok, c).numpy().astype(np.float64)
        t = (t - t.mean(-1, keepdims=True)) / np.sqrt(t.var(-1, keepdims=True) + 1e-5)
        e = ((pred[m].detach().numpy() - t) ** 2).mean(-1)
        ref[m] = e[mk[m].numpy()].mean()
    # predictions changed on the visible positions only: same loss
    alt = {m: torch.where(mk[m][..., None], pred[m], torch.randn_like(pred[m])) for m in MODALITIES}
    loss_alt = reconstruction_loss(alt, tok, mk)[0]
    tgt = F.layer_norm(tok["wac"].reshape(n_tiles, n_tok, c), (c,))
    loss.backward()
    params = list(enc.named_parameters()) + [("decoder." + n, p) for n, p in dec.named_parameters()]
    no_grad = [n for n, p in params if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
    # short fit: AdamW, lr 3e-4, fixed masks
    torch.manual_seed(int(cfg["seed"]))
    enc, dec = build_fusion_x(cfg), build_decoder(cfg)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=3e-4)
    curve = []
    for _ in range(steps):
        opt.zero_grad()
        l, _ = reconstruction_loss(dec(enc(tok, mk)), tok, mk)
        l.backward()
        opt.step()
        curve.append(l.item())
    report = {
        "hyper_parameters": f["decoder"], "decoder_parameters": int(sum(p.numel() for p in dec.parameters())),
        "check_tiles": [int(t) for t in store.index["tile"][rows]],
        "prediction_shape_per_modality": list(pred["wac"].shape),
        "target_per_token_mean_abs_max_std_mean": [round(float(tgt.mean(-1).abs().max()), 6), round(float(tgt.std(-1, unbiased=False).mean()), 4)],
        "loss_untrained_total_wac_dem": [round(loss.item(), 4), round(per["wac"].item(), 4), round(per["dem"].item(), 4)],
        "loss_of_predicting_zero": round(float(reconstruction_loss({m: torch.zeros_like(pred[m]) for m in MODALITIES}, tok, mk)[0]), 4),
        "check_loss_equals_numpy": bool(all(abs(per[m].item() - ref[m]) < 1e-4 for m in MODALITIES)),
        "check_loss_ignores_visible_positions": bool(torch.allclose(loss_alt, loss)),
        "parameters_without_gradient": no_grad,
        "short_fit_steps_lr": [steps, 3e-4],
        "short_fit_loss_first_last": [round(curve[0], 4), round(curve[-1], 4)],
        "check_short_fit_lowers_loss": bool(curve[-1] < 0.8 * curve[0]),
    }
    save_report(cfg, "QF3.3", report)
    return report



def fusion_dir(cfg: dict) -> Path:
    """Trained fusion models of QF3.4: ``<output>/fusion/``."""
    return cfg["paths"]["output"] / "fusion"


def run_name(vid: str, lr: float, seed: int) -> str:
    return f"{vid}_lr{lr:.0e}_s{seed}"


def set_tokens(cfg: dict, store: TokenStore, set_name: str, device, max_tiles: int | None = None) -> dict:
    """Tokens {modality: (n, 256, 384) float16 tensor on ``device``} of the tiles of ``set_name``."""
    import torch

    rows = np.flatnonzero(store.index["set"] == set_name)[:max_tiles]
    return {m: torch.from_numpy(np.asarray(store.caches[m][rows])).reshape(rows.size, -1, store.caches[m].shape[-1]).to(device)
            for m in MODALITIES}


def evaluate(enc, dec, tokens: dict, masks: dict, batch: int = 256) -> dict:
    """Reconstruction loss of ``reconstruction_loss`` over a whole set (per modality the mean over all its
    masked tokens, then the average): {"total", "wac", "dem"}."""
    import torch

    enc.eval(), dec.eval()
    num, den = {m: 0.0 for m in MODALITIES}, {m: 0.0 for m in MODALITIES}
    with torch.no_grad():
        for s in range(0, tokens["wac"].shape[0], batch):
            tok = {m: tokens[m][s:s + batch] for m in MODALITIES}
            mk = {m: masks[m][s:s + batch] for m in MODALITIES}
            per = reconstruction_loss(dec(enc(tok, mk)), tok, mk)[1]
            for m in MODALITIES:
                c = float(mk[m].sum())
                num[m] += per[m].item() * c
                den[m] += c
    out = {m: num[m] / max(den[m], 1.0) for m in MODALITIES}
    out["total"] = sum(out[m] for m in MODALITIES) / len(MODALITIES)
    return out


def train_fusion(cfg: dict, vid: str, lr: float, seed: int, train: dict, val: dict, val_masks: dict,
                 out_dir: Path, epochs: int | None = None, stop_after: int | None = None, record: bool = True) -> dict:
    """QF3.4: self-supervised training of ``vid`` with the decoder of QF3.3 (masks of QF3.1-QF3.2 drawn
    anew every batch; fixed masks on S1-val).  AdamW, linear warm-up then cosine decay (per step), early
    stopping on the S1-val loss.  ``out_dir/<run>.pt`` holds the best weights (fusion and decoder), the
    per-epoch curve and the state to resume, rewritten every epoch; a finished run is not trained again.
    ``stop_after``: leave after that many epochs of this call, as a killed job would (test of the resume)."""
    import math
    import time

    import torch

    t = cfg["fusion"]["train"]
    epochs = int(epochs or t["epochs"])
    device = train["wac"].device
    path = out_dir / f"{run_name(vid, lr, seed)}.pt"
    torch.manual_seed(seed)
    enc, dec = build_fusion(cfg, vid).to(device), build_decoder(cfg).to(device)
    params = [(n, p) for mod in (enc, dec) for n, p in mod.named_parameters()]
    groups = [{"params": [p for n, p in params if p.ndim >= 2], "weight_decay": float(t["weight_decay"])},
              {"params": [p for n, p in params if p.ndim < 2], "weight_decay": 0.0}]   # biases, norms, tokens
    opt = torch.optim.AdamW(groups, lr=lr, betas=tuple(float(b) for b in t["betas"]))
    n, batch = train["wac"].shape[0], int(t["batch"])
    spe = math.ceil(n / batch)
    total, warm = epochs * spe, int(t["warmup_epochs"]) * spe
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    g_perm = torch.Generator().manual_seed(seed)
    g_mask = torch.Generator(device=device).manual_seed(seed)
    n_tok = train["wac"].shape[1]
    f = cfg["fusion"]
    state = {"vid": vid, "lr": lr, "seed": seed, "epochs_max": epochs, "curve": [], "best": None, "done": False}
    if path.exists():                                                   # resume or skip
        ck = torch.load(path, map_location=device, weights_only=False)
        if ck["done"]:
            return {k: ck[k] for k in ("vid", "lr", "seed", "curve", "best", "stopped_early")}
        state = ck
        enc.load_state_dict(ck["last"]["enc"]), dec.load_state_dict(ck["last"]["dec"])
        opt.load_state_dict(ck["last"]["opt"]), sched.load_state_dict(ck["last"]["sched"])
        g_perm.set_state(ck["last"]["g_perm"]), g_mask.set_state(ck["last"]["g_mask"].to(g_mask.get_state().device))
    stopped, start = False, len(state["curve"])
    for epoch in range(start, epochs):
        t0 = time.perf_counter()
        enc.train(), dec.train()
        perm, run = torch.randperm(n, generator=g_perm).to(device), 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            tok = {m: train[m][idx] for m in MODALITIES}
            mk = sample_masks(idx.numel(), n_tok, f["mask_ratio"], float(f["modality_dropout"]), generator=g_mask, device=device)
            loss = reconstruction_loss(dec(enc(tok, mk)), tok, mk)[0]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item() * idx.numel()
        v = evaluate(enc, dec, val, val_masks)
        state["curve"].append({"epoch": epoch + 1, "train": round(run / n, 5), "val": round(v["total"], 5),
                               "val_wac": round(v["wac"], 5), "val_dem": round(v["dem"], 5),
                               "lr_end": float(f"{sched.get_last_lr()[0]:.3e}"), "seconds": round(time.perf_counter() - t0, 1)})
        if state["best"] is None or v["total"] < state["best"]["val"]:
            state["best"] = {"epoch": epoch + 1, "val": v["total"], "val_wac": v["wac"], "val_dem": v["dem"]}
            state["best_state"] = {"enc": {k: x.detach().cpu().clone() for k, x in enc.state_dict().items()},
                                   "dec": {k: x.detach().cpu().clone() for k, x in dec.state_dict().items()}}
        c = state["curve"][-1]
        print(f"  {run_name(vid, lr, seed)} epoch {epoch + 1:2d}  train {c['train']:.4f}  val {c['val']:.4f} "
              f"(wac {c['val_wac']:.4f}, dem {c['val_dem']:.4f})  {c['seconds']} s", flush=True)
        stopped = epoch + 1 - state["best"]["epoch"] >= int(t["patience"])
        state["done"] = stopped or epoch + 1 == epochs
        state["stopped_early"] = stopped
        state["last"] = None if state["done"] else {
            "enc": enc.state_dict(), "dec": dec.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
            "g_perm": g_perm.get_state(), "g_mask": g_mask.get_state()}
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)                          # atomic: a killed job leaves a valid file
        if state["done"] or (stop_after is not None and epoch + 1 - start >= stop_after):
            break
    if record:
        written(path)
    return {k: state[k] for k in ("vid", "lr", "seed", "curve", "best", "stopped_early")}


def model_path(cfg: dict, vid: str, lr: float | None = None, seed: int | None = None) -> Path:
    """File of a trained fusion of QF3.4.  Default: the model used downstream = first seed of
    ``fusion.train.seeds`` at the lr chosen in QF3.4."""
    seed = int(cfg["fusion"]["train"]["seeds"][0]) if seed is None else seed
    lr = load_report(cfg, "QF3.4")["models"][vid]["chosen_lr"] if lr is None else lr
    return fusion_dir(cfg) / f"{run_name(vid, lr, seed)}.pt"


def load_fusion(cfg: dict, vid: str, lr: float | None = None, seed: int | None = None, device="cpu",
                decoder: bool = False):
    """Trained fusion ``vid`` (best weights of QF3.4, ``model_path``), eval mode; with ``decoder`` also
    its decoder: (fusion, decoder)."""
    import torch

    path = model_path(cfg, vid, lr, seed)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not ck["done"]:
        raise ValueError(f"training of {path.stem} not finished")
    enc = build_fusion(cfg, vid)
    enc.load_state_dict(ck["best_state"]["enc"])
    enc = enc.to(device).eval()
    if not decoder:
        return enc
    dec = build_decoder(cfg)
    dec.load_state_dict(ck["best_state"]["dec"])
    return enc, dec.to(device).eval()


def run_train(cfg: dict, variants, device: str, smoke: bool = False, sweep: bool = False) -> dict:
    """QF3.4: per model, the lrs of ``fusion.train.lrs`` with the first seed, the lr with the lowest best
    S1-val loss chosen, then the other seeds at that lr -> report section QF3.4 (per run best epoch and
    losses, curve; per model chosen lr and spread over the seeds).  ``smoke``: 64 tiles, 2 epochs, first
    lr and seed, models in a temporary folder, no report (test of the code before the GPU job).
    ``sweep``: QF3.5, the recipe of ``fusion.train`` with the overrides of ``fusion.train_sweep``, models in
    ``fusion/sweep/``, report section QF3.5 with the comparison to QF3.4 and the switch rule."""
    import copy
    import tempfile
    import time

    import torch

    if sweep:
        cfg = copy.deepcopy(cfg)
        over = {k: v for k, v in cfg["fusion"]["train_sweep"].items() if k != "switch_min_gain"}
        cfg["fusion"]["train"] = {**cfg["fusion"]["train"], **over}
    t = cfg["fusion"]["train"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    store = TokenStore(cfg)
    mx = 64 if smoke else None
    t0 = time.perf_counter()
    train, val = set_tokens(cfg, store, "s1_train", device, mx), set_tokens(cfg, store, "s1_val", device, mx)
    load_s = round(time.perf_counter() - t0, 1)
    n_tok = train["wac"].shape[1]
    f = cfg["fusion"]
    val_masks = sample_masks(val["wac"].shape[0], n_tok, f["mask_ratio"], float(f["modality_dropout"]),
                             generator=torch.Generator().manual_seed(int(cfg["seed"])))      # same for every run
    val_masks = {m: x.to(device) for m, x in val_masks.items()}
    out_dir = Path(tempfile.mkdtemp(dir=cfg["paths"]["output"])) if smoke else fusion_dir(cfg) / ("sweep" if sweep else "")
    lrs, seeds = [float(x) for x in t["lrs"]], [int(x) for x in t["seeds"]]
    if smoke:
        lrs, seeds = lrs[:1], seeds[:1]
    models = {}
    for vid in variants:
        t1 = time.perf_counter()
        ep, rec = (2, False) if smoke else (None, True)
        runs = [train_fusion(cfg, vid, lr, seeds[0], train, val, val_masks, out_dir, ep, record=rec) for lr in lrs]
        chosen = min(runs, key=lambda r: r["best"]["val"])["lr"]
        runs += [train_fusion(cfg, vid, chosen, s, train, val, val_masks, out_dir, ep, record=rec) for s in seeds[1:]]
        at = [r for r in runs if r["lr"] == chosen]
        vals = np.array([[r["best"][k] for k in ("val", "val_wac", "val_dem")] for r in at])
        models[vid] = {
            "parameters": int(sum(p.numel() for p in build_fusion(cfg, vid).parameters())),
            "chosen_lr": chosen, "model_used_downstream": run_name(vid, chosen, seeds[0]),
            "best_val_mean_std_over_seeds_total_wac_dem": [[round(float(a), 4), round(float(b), 4)]
                                                           for a, b in zip(vals.mean(0), vals.std(0))],
            "minutes": round((time.perf_counter() - t1) / 60, 1),
            "runs": [{"run": run_name(r["vid"], r["lr"], r["seed"]), "lr": r["lr"], "seed": r["seed"],
                      "epochs_run": len(r["curve"]), "stopped_early": bool(r["stopped_early"]),
                      "best_epoch": r["best"]["epoch"],
                      "best_val_total_wac_dem": [round(r["best"][k], 4) for k in ("val", "val_wac", "val_dem")],
                      "curve_train_val_val_wac_val_dem": [[c["train"], c["val"], c["val_wac"], c["val_dem"]] for c in r["curve"]],
                      "seconds_per_epoch_median": float(np.median([c["seconds"] for c in r["curve"]]))} for r in runs]}
        print(f"{vid}: chosen lr {chosen:.0e}, best val (total, wac, dem) mean/std over seeds "
              f"{models[vid]['best_val_mean_std_over_seeds_total_wac_dem']}", flush=True)
    report = {"device": torch.cuda.get_device_name() if str(device).startswith("cuda") else "cpu",
              "train_tiles": int(train["wac"].shape[0]), "val_tiles": int(val["wac"].shape[0]),
              "seconds_loading_tokens": load_s, "recipe": t, "val_masks_seed": int(cfg["seed"]),
              "models_folder": str(out_dir), "models": models}
    if smoke:
        # resume: 3 epochs stopped after 1 and resumed (seed + 1, new file) = 3 epochs in one go
        import shutil

        train_fusion(cfg, variants[0], lrs[0], seeds[0] + 1, train, val, val_masks, out_dir, 3, stop_after=1, record=False)
        a = train_fusion(cfg, variants[0], lrs[0], seeds[0] + 1, train, val, val_masks, out_dir, 3, record=False)
        (out_dir / "straight").mkdir()
        b = train_fusion(cfg, variants[0], lrs[0], seeds[0] + 1, train, val, val_masks, out_dir / "straight", 3, record=False)
        key = lambda r: [(c["train"], c["val"]) for c in r["curve"]]
        report["smoke_resume_equals_straight_run"] = key(a) == key(b)
        report["smoke_resume_curves"] = [key(a), key(b)]
        shutil.rmtree(out_dir)
    elif sweep:
        report["comparison_with_qf34"] = sweep_comparison(cfg, models)
        save_report(cfg, "QF3.5", report)
    else:
        save_report(cfg, "QF3.4", report)
    return report


def sweep_comparison(cfg: dict, models: dict) -> dict:
    """QF3.5: best S1-val loss of the sweep against the first recipe of QF3.4 (section QF3.4-r1) with the
    same seed, per model, and the switch
    rule fixed before the run: the F-X loss >= ``switch_min_gain`` lower and the run stable (finite, last
    epoch within 1 % of the best)."""
    qf34 = load_report(cfg, "QF3.4-r1")["models"]                       # the first recipe (superseded after QF3.5)
    gain_min = float(cfg["fusion"]["train_sweep"]["switch_min_gain"])
    out = {}
    for vid, m in models.items():
        seed = m["runs"][0]["seed"]
        ref = min(r["best_val_total_wac_dem"][0] for r in qf34[vid]["runs"] if r["seed"] == seed and r["lr"] == qf34[vid]["chosen_lr"])
        best = min(m["runs"], key=lambda r: r["best_val_total_wac_dem"][0])
        last = best["curve_train_val_val_wac_val_dem"][-1][1]
        stable = bool(np.isfinite(last) and last <= 1.01 * best["best_val_total_wac_dem"][0])
        out[vid] = {"qf34_best_val_same_seed": ref, "sweep_best_run": best["run"], "sweep_best_val": best["best_val_total_wac_dem"][0],
                    "relative_gain": round(1 - best["best_val_total_wac_dem"][0] / ref, 4), "stable": stable,
                    "all_runs_best_val": {r["run"]: r["best_val_total_wac_dem"][0] for r in m["runs"]}}
    fx = out.get("F-X")
    out["switch_min_gain"] = gain_min
    out["switch_rule_met"] = bool(fx and fx["relative_gain"] >= gain_min and fx["stable"])
    return out


# ---------------------------------------------------------------------------
# QF4 - Checks of the trained fusions, cache of the fused maps
# ---------------------------------------------------------------------------


def tile_classes(cfg: dict, store: TokenStore, rows) -> np.ndarray:
    """(len(rows), 256) class (QB2.3: 1 / 0 / -1) of every sub-tile of the tiles at ``rows``, token order."""
    cls = np.load(cfg["paths"]["output"] / "cell_labels.npz")["cls"]
    g = int(cfg["tiling"]["tile_px"]) // int(cfg["tiling"]["cell_px"])
    k = int(cfg["tiling"]["stride_px"]) // int(cfg["tiling"]["cell_px"])
    a = np.arange(g)
    ci = store.index["ti"][rows, None, None] * k + a[None, :, None]
    cj = store.index["tj"][rows, None, None] * k + a[None, None, :]
    return cls[ci, cj].reshape(len(rows), -1)


def attention_stats(enc, tokens: dict, cls: np.ndarray, batch: int = 64) -> dict:
    """QF4.2: cross-attention of F-X (head average) per block and direction (``wac`` = WAC queries over
    DEM keys), on all the tiles of ``tokens``: mass in the 3 x 3 cells around the query for positive and
    negative query cells (uniform = 9/256), and lift on the positive cells = per tile mean mass on its
    positive cells / their fraction, averaged over the tiles with positives (1 = no preference)."""
    import torch

    g = enc.grid
    r, c = np.divmod(np.arange(g * g), g)
    near = torch.from_numpy(((np.abs(r[:, None] - r[None]) <= 1) & (np.abs(c[:, None] - c[None]) <= 1)).astype(np.float32))
    n_blocks = len(enc.blocks)
    acc = {(b, m): {"pos": [0.0, 0], "neg": [0.0, 0], "lift": []} for b in range(n_blocks) for m in MODALITIES}
    with torch.no_grad():
        for s in range(0, tokens["wac"].shape[0], batch):
            tok = {m: tokens[m][s:s + batch] for m in MODALITIES}
            w = enc.encode(tok, need_weights=True)[2]
            lab = torch.from_numpy(cls[s:s + batch])
            pos = (lab == 1).float()
            frac = pos.mean(1)
            for b in range(n_blocks):
                for m in MODALITIES:
                    a = w[b][m].float().mean(1).cpu()                                     # (B, 256, 256)
                    local = (a * near).sum(-1)
                    for key, sel in (("pos", lab == 1), ("neg", lab == 0)):
                        acc[b, m][key][0] += float(local[sel].sum())
                        acc[b, m][key][1] += int(sel.sum())
                    has = frac > 0
                    if has.any():
                        acc[b, m]["lift"] += ((a[has] * pos[has][:, None, :]).sum(-1).mean(1) / frac[has]).tolist()
    out = {}
    for (b, m), v in acc.items():
        out[f"block{b + 1}_{m}_queries"] = {
            "mass_3x3_positive_queries": round(v["pos"][0] / max(v["pos"][1], 1), 4),
            "mass_3x3_negative_queries": round(v["neg"][0] / max(v["neg"][1], 1), 4),
            "lift_on_positive_cells_mean_median": [round(float(np.mean(v["lift"])), 3), round(float(np.median(v["lift"])), 3)]}
    out["uniform_mass_3x3"] = round(9 / (g * g), 4)
    out["tiles_with_positives"] = len(acc[0, "wac"]["lift"])
    return out


def run_fusion_checks(cfg: dict, device: str = "cpu") -> dict:
    """QF4.1: S1-val reconstruction loss of F-X and F-CM (every seed at the chosen lr) with the masks of
    QF3.4, with one modality fully masked (rebuilt from the other one) and with both masked (no input:
    reference).  QF4.2: the fused maps of the downstream models are not constant (per-channel and
    spatial spread, PCA), and the cross-attention of F-X on the crater cells (``attention_stats``,
    trained vs untrained), with the figure ``figures/qf_attention.png`` (tiles of Fig. D3)."""
    import time

    import torch

    t0 = time.perf_counter()
    f, t = cfg["fusion"], cfg["fusion"]["train"]
    store = TokenStore(cfg)
    val = set_tokens(cfg, store, "s1_val", device)
    nv, n_tok = val["wac"].shape[:2]
    rand = sample_masks(nv, n_tok, f["mask_ratio"], float(f["modality_dropout"]),
                        generator=torch.Generator().manual_seed(int(cfg["seed"])))           # the S1-val masks of QF3.4
    full = lambda w, d: {"wac": torch.full((nv, n_tok), w), "dem": torch.full((nv, n_tok), d)}
    conds = {"random_masks": rand, "wac_from_dem": full(True, False), "dem_from_wac": full(False, True),
             "both_masked": full(True, True)}
    conds = {k: {m: x.to(device) for m, x in v.items()} for k, v in conds.items()}
    pick = {"random_masks": ("total", "wac", "dem"), "wac_from_dem": ("wac",), "dem_from_wac": ("dem",),
            "both_masked": ("total", "wac", "dem")}
    qf34 = load_report(cfg, "QF3.4")["models"]
    q41 = {}
    for vid in ("F-X", "F-CM"):
        lr = qf34[vid]["chosen_lr"]
        per_seed = {}
        for seed in (int(x) for x in t["seeds"]):
            enc, dec = load_fusion(cfg, vid, lr, seed, device, decoder=True)
            per_seed[seed] = {c: evaluate(enc, dec, val, mk) for c, mk in conds.items()}
        res = {}
        for c, keys in pick.items():
            for k in keys:
                v = np.array([per_seed[sd][c][k] for sd in per_seed])
                res[f"{c}_{k}"] = [round(float(v.mean()), 4), round(float(v.std()), 4)]
        # share of the no-input loss removed by the other modality (0 = no help, 1 = perfect)
        for m, c in (("wac", "wac_from_dem"), ("dem", "dem_from_wac")):
            v = np.array([1 - per_seed[sd][c][m] / per_seed[sd]["both_masked"][m] for sd in per_seed])
            res[f"{c}_gain_vs_no_input"] = [round(float(v.mean()), 4), round(float(v.std()), 4)]
        best = {r["seed"]: r["best_val_total_wac_dem"][0] for r in qf34[vid]["runs"] if r["lr"] == lr}
        diff = max(abs(per_seed[sd]["random_masks"]["total"] - best[sd]) for sd in per_seed)
        q41[vid] = {"lr": lr, "seeds": list(per_seed), "loss_mean_std_over_seeds": res,
                    "max_abs_diff_random_masks_vs_qf34": float(f"{diff:.1e}"),
                    "check_random_masks_equal_qf34": bool(diff < 1e-3)}
    q41["note"] = "losses on S1-val (1.0 = zero prediction); *_from_*: the target modality fully masked, the other fully visible"
    save_report(cfg, "QF4.1", q41)

    # QF4.2 (a) spread of the fused maps, downstream models, no masks
    spread = {}
    for vid in ("F-X", "F-CM"):
        enc = load_fusion(cfg, vid, device=device)
        with torch.no_grad():
            x = torch.cat([enc({m: val[m][s:s + 128] for m in MODALITIES}).reshape(-1, n_tok, f["out_dim"]).float().cpu()
                           for s in range(0, nv, 128)])                                       # (nv, 256, 512)
        flat = x.reshape(-1, x.shape[-1])
        std = flat.std(0)
        spatial = x.std(1).mean(1)                                                             # per tile
        ev = torch.linalg.eigvalsh(torch.cov(flat.T.double())).flip(0).clamp(min=0)
        cum = (ev.cumsum(0) / ev.sum()).numpy()
        spread[vid] = {"channel_std_min_median_max": [round(float(std.min()), 4), round(float(std.median()), 4), round(float(std.max()), 4)],
                       "channels_std_below_1pct_of_median": int((std < 0.01 * std.median()).sum()),
                       "spatial_std_per_tile_p1_median": [round(float(spatial.quantile(0.01)), 4), round(float(spatial.median()), 4)],
                       "tiles_spatially_constant": int((spatial < 0.01 * std.median()).sum()),
                       "pca_variance_first_16": round(float(cum[15]), 4),
                       "pca_components_for_90pct": int(np.searchsorted(cum, 0.9) + 1),
                       "check_not_constant": bool((std > 0.01 * std.median()).all() and (spatial > 0.01 * std.median()).all())}
    # (b) attention of F-X on the crater cells, trained (downstream) vs untrained (same init)
    rows_val = np.flatnonzero(store.index["set"] == "s1_val")
    cls = tile_classes(cfg, store, rows_val)
    enc = load_fusion(cfg, "F-X", device=device)
    torch.manual_seed(int(t["seeds"][0]))
    init = build_fusion_x(cfg).to(device).eval()
    attn = {"trained": attention_stats(enc, val, cls), "untrained_same_init": attention_stats(init, val, cls)}
    # (c) figure: tiles of Fig. D3, query = the owned positive cell with the largest crater
    figure_tiles = attention_example(cfg, store, enc, device)
    q42 = {"val_tiles": int(nv), "fused_map_spread": spread, "cross_attention_s1_val": attn,
           "figure": "figures/qf_attention.png", "figure_queries": figure_tiles,
           "seconds": round(time.perf_counter() - t0, 1)}
    save_report(cfg, "QF4.2", q42)
    return {"QF4.1": q41, "QF4.2": q42}


def attention_example(cfg: dict, store: TokenStore, enc, device) -> dict:
    """QF4.2 figure: on the 3 tiles of Fig. D3 the last-block cross-attention of F-X (head average) of
    one query, the owned positive cell with the largest crater, in both directions."""
    import pandas as pd
    import torch

    from qc_figures import draw_attention_check
    from qc_tiles import TileReader, owned_cells

    reader = TileReader(cfg)
    tiles = pd.read_csv(cfg["paths"]["output"] / "tiles.csv")
    lab = np.load(cfg["paths"]["output"] / "cell_labels.npz")
    g = enc.grid
    k = int(cfg["tiling"]["stride_px"]) // int(cfg["tiling"]["cell_px"])
    (lr_, lc_), _ = owned_cells(0, 0, int(cfg["tiling"]["tile_px"]), int(cfg["tiling"]["stride_px"]), int(cfg["tiling"]["cell_px"]))
    shown, queries = [], {}
    for name, v in load_report(cfg, "QD-A3")["tiles"].items():
        tile = int(v["tile"])
        row = store.row_of_tile[tile]
        ti, tj = int(tiles.at[tile, "ti"]), int(tiles.at[tile, "tj"])
        cls = tile_classes(cfg, store, np.array([row]))[0].reshape(g, g)
        diam = lab["diam_km"][ti * k:ti * k + g, tj * k:tj * k + g]
        own = np.zeros((g, g), bool)
        own[lr_, lc_] = True
        cand = np.where((cls == 1) & own, np.nan_to_num(diam, nan=-1), -1)
        qi, qj = np.unravel_index(int(np.argmax(cand)), cand.shape)
        tok = {m: torch.from_numpy(np.asarray(store.caches[m][[row]], dtype=np.float32)).to(device) for m in MODALITIES}
        with torch.no_grad():
            w = enc.encode(tok, need_weights=True)[2][-1]
        att = {m: w[m][0].float().mean(0)[qi * g + qj].reshape(g, g).cpu().numpy() for m in MODALITIES}
        img = reader.read(ti, tj, unit=True)["inputs"]
        shown.append((name, tile, tiles.loc[tile], {"wac": img["wac"][0], "hill": img["dem"][2], "lab": cls,
                                                   "query": (int(qi), int(qj)), "attn": att}))
        queries[name] = {"tile": tile, "query_cell": [int(qi), int(qj)], "query_crater_km": round(float(diam[qi, qj]), 2),
                         "attention_on_query_3x3_wac_dem": [round(float(att[m][max(qi - 1, 0):qi + 2, max(qj - 1, 0):qj + 2].sum()), 3)
                                                            for m in MODALITIES]}
    draw_attention_check(cfg, shown)
    return queries


def run_fused_cache(cfg: dict, device: str = "cpu", batch: int = 256, n_check: int = 16) -> dict:
    """QF4.3: fused maps of the downstream F-X and F-CM (``model_path``) for every used tile, written as
    float16 memory-mapped ``<output>/tokens/fx.npy`` and ``fcm.npy`` (N, 16, 16, 512), same rows as the
    token cache, resumable (``progress.npz``); then ``variant_maps`` reads them like the other variants.
    Checks: every row written and finite, ``n_check`` random tiles recomputed in float32."""
    import hashlib
    import time

    import torch

    from qc_features import tokens_dir

    d = tokens_dir(cfg)
    store = TokenStore(cfg)
    n, g, dim = len(store.index["tile"]), int(store.caches["wac"].shape[1]), int(cfg["fusion"]["out_dim"])
    shape = (n, g, g, dim)
    progress = dict(np.load(d / "progress.npz"))
    report = {}
    for vid, name in (("F-X", "fx"), ("F-CM", "fcm")):
        t0 = time.perf_counter()
        path, mpath = d / f"{name}.npy", model_path(cfg, vid)
        md5 = hashlib.md5(mpath.read_bytes()).hexdigest()
        enc = load_fusion(cfg, vid, device=device)
        if path.exists() and name in progress and str(progress.get(name + "_model", "")) == md5:
            cache = np.load(path, mmap_mode="r+")
        else:                                                             # new, or written by another model
            store.caches.pop(name, None)
            path.unlink(missing_ok=True)                                  # new file: the old one may be memory-mapped
            cache = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
            progress[name], progress[name + "_model"] = np.zeros(n, bool), np.array(md5)
        todo = np.flatnonzero(~progress[name])
        print(f"{vid}: {len(todo)} of {n} tiles to write", flush=True)
        for c, s in enumerate(range(0, len(todo), batch)):
            rows = todo[s:s + batch]
            tok = {m: torch.from_numpy(np.asarray(store.caches[m][rows], dtype=np.float32)).to(device) for m in MODALITIES}
            with torch.no_grad():
                cache[rows] = enc(tok).cpu().numpy().astype(np.float16)
            progress[name][rows] = True
            if c % 10 == 0 or s + batch >= len(todo):
                cache.flush()
                np.savez(d / "progress.npz", **progress)
                print(f"  {int(progress[name].sum())}/{n} tiles, {time.perf_counter() - t0:.0f} s", flush=True)
        written(path)
        written(d / "progress.npz")
        finite = all(bool(np.isfinite(np.asarray(cache[s:s + 512], dtype=np.float32)).all()) for s in range(0, n, 512))
        chk = np.sort(np.random.default_rng(int(cfg["seed"])).choice(n, n_check, replace=False))
        tok = {m: torch.from_numpy(np.asarray(store.caches[m][chk], dtype=np.float32)).to(device) for m in MODALITIES}
        with torch.no_grad():
            ref = enc(tok).cpu()
        c16 = torch.from_numpy(np.asarray(cache[chk], dtype=np.float32))
        cos = torch.nn.functional.cosine_similarity(ref, c16, dim=-1)
        report[vid] = {"file": f"tokens/{name}.npy", "model": mpath.name, "model_md5": md5, "shape": list(shape),
                       "size_gb": round(path.stat().st_size / 1e9, 2), "check_all_rows_done": bool(progress[name].all()),
                       "check_finite": finite, "check_tiles": n_check,
                       "max_abs_diff_float16_vs_float32": float(f"{float((ref - c16).abs().max()):.1e}"),
                       "min_cosine_float16_vs_float32": round(float(cos.min()), 6),
                       "minutes": round((time.perf_counter() - t0) / 60, 1)}
    # the variants now read the caches: owned samples of a few S1-val tiles equal the model output
    store = TokenStore(cfg)
    for vid in ("F-X", "F-CM"):
        x, idx = owned_samples(cfg, store, vid, "s1_val", max_tiles=4)
        enc = load_fusion(cfg, vid, device=device)
        rows = np.unique(idx["row"])
        tok = {m: torch.from_numpy(np.asarray(store.caches[m][rows], dtype=np.float32)).to(device) for m in MODALITIES}
        with torch.no_grad():
            ref = enc(tok).cpu().numpy()
        r = ref[np.searchsorted(rows, idx["row"]), idx["i"], idx["j"]]
        report[vid]["check_owned_samples_equal_model_4_val_tiles"] = bool(np.allclose(x, r, atol=2e-2, rtol=1e-2))
    save_report(cfg, "QF4.3", report)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["variants", "model", "masking", "decoder", "train", "checks", "cache"])
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--variants", nargs="+", default=["F-X", "F-CM"], choices=["F-X", "F-CM"], help="train: models")
    ap.add_argument("--device", default=None, help="train / checks / cache: torch device (default: cuda for train, cpu otherwise)")
    ap.add_argument("--smoke", action="store_true", help="train: 64 tiles, 2 epochs, no report (code test)")
    ap.add_argument("--sweep", action="store_true", help="train: QF3.5 exploratory recipe (fusion.train_sweep)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "variants":
        print(json.dumps(run_variants_check(cfg), indent=1))
    elif args.command == "model":
        print(json.dumps(run_model_check(cfg), indent=1))
    elif args.command == "masking":
        print(json.dumps(run_masking_check(cfg), indent=1))
    elif args.command == "decoder":
        print(json.dumps(run_decoder_check(cfg), indent=1))
    elif args.command == "train":
        r = run_train(cfg, args.variants, args.device or "cuda", args.smoke, args.sweep)
        for m in r["models"].values():
            for x in m["runs"]:
                x.pop("curve_train_val_val_wac_val_dem")
        print(json.dumps(r, indent=1))
    elif args.command == "checks":
        print(json.dumps(run_fusion_checks(cfg, args.device or "cpu"), indent=1))
    elif args.command == "cache":
        print(json.dumps(run_fused_cache(cfg, args.device or "cpu"), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
