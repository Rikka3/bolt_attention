"""Bolt-3 Language Model & Checkpoint Loader."""
from __future__ import annotations

import gc
import os

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .attention import Bolt3Attn
from .nn import GatedDeltaNet, RMSNorm, SoftmaxAttn, SwiGLUMLP

# Parameters always maintained in fp32 (checkpoints, healing master weights, and inference)
FP32_KEYS = ("log_gamma", "kappa", "beta_proj", "eta_proj", "res_norm", "A_log", "dt_bias")


class Bolt3Block(nn.Module):
    def __init__(self, cfg: dict, layer_type: str = "bolt3", layer_idx: int = 0):
        super().__init__()
        norm_off = float(cfg.get("rms_norm_offset", 0.0))
        self.norm1 = RMSNorm(int(cfg["dim"]), float(cfg["eps"]), weight_offset=norm_off)
        if layer_type in ("bolt3", "vsa"):
            self.attn = Bolt3Attn(
                cfg,
                max_T=int(cfg.get("max_T", 2048)),
                S=int(cfg.get("bolt3_S", 128)),
                C=int(cfg.get("bolt3_C", 128)),
                layer_idx=layer_idx,
            )
        elif layer_type in ("linear_attention", "gated_deltanet"):
            self.attn = GatedDeltaNet(cfg, layer_idx=layer_idx)
        else:
            self.attn = SoftmaxAttn(cfg)
        self.norm2 = RMSNorm(int(cfg["dim"]), float(cfg["eps"]), weight_offset=norm_off)
        self.mlp = SwiGLUMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Bolt3LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate: int,
        max_T: int = 2048,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        eps: float = 1e-6,
        head_dim: int | None = None,
        qkv_bias: bool = False,
        tie_word_embeddings: bool = True,
        **extra_cfg,
    ):
        super().__init__()
        head_dim = int(head_dim or (dim // num_heads))
        extra_cfg.pop("is_bolt3", None)
        layer_types = extra_cfg.pop("layer_types", None)
        if not layer_types:
            layer_types = ["bolt3"] * int(num_layers)
        else:
            # Normalize any legacy "vsa" entries to "bolt3"
            layer_types = ["bolt3" if lt in ("bolt3", "vsa") else lt for lt in layer_types]

        self.config = dict(
            vocab_size=int(vocab_size),
            dim=int(dim),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            num_kv_heads=int(num_kv_heads),
            intermediate=int(intermediate),
            max_T=int(max_T),
            rope_theta=float(rope_theta),
            rope_scaling=rope_scaling,
            eps=float(eps),
            head_dim=head_dim,
            qkv_bias=bool(qkv_bias),
            tie_word_embeddings=bool(tie_word_embeddings),
            is_bolt3=True,
            layer_types=layer_types,
            **extra_cfg,
        )
        self.embed_tokens = nn.Embedding(self.config["vocab_size"], self.config["dim"])
        self.blocks = nn.ModuleList([Bolt3Block(self.config, lt, layer_idx=li) for li, lt in enumerate(layer_types)])
        self.norm = RMSNorm(
            self.config["dim"],
            self.config["eps"],
            weight_offset=float(self.config.get("rms_norm_offset", 0.0)),
        )
        self.lm_head = nn.Linear(self.config["dim"], self.config["vocab_size"], bias=False)
        if self.config["tie_word_embeddings"]:
            self.lm_head.weight = self.embed_tokens.weight
        self.grad_checkpoint = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return self.norm(x)


def _remap_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap legacy `blocks.{i}.vsa.*` keys from older checkpoints to `blocks.{i}.attn.*`."""
    out = {}
    for k, v in sd.items():
        if k.startswith("blocks.") and ".vsa." in k:
            k = k.replace(".vsa.", ".attn.", 1)
        out[k] = v
    return out


def load_bolt3_checkpoint(
    ckpt_path: str,
    device: torch.device | str | None = None,
    donor_override: str | None = None,
    eval_mode: bool = True,
):
    """Load a Bolt-3 checkpoint into Bolt3LM. Returns (model, cfg, donor_id, step, meta)."""
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=(os.name != "nt"))
    except Exception:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = dict(ck["config"])
    cfg["is_bolt3"] = True
    donor_id = donor_override or ck.get("donor") or (ck.get("injection") or {}).get("donor")
    meta = {k: v for k, v in ck.items() if k != "model"}
    sd = _remap_state_dict(ck["model"])
    sd_keys = set(sd.keys())
    del ck

    model = Bolt3LM(**cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    del sd
    gc.collect()
    if cfg.get("tie_word_embeddings", True) and "lm_head.weight" in missing:
        model.lm_head.weight = model.embed_tokens.weight
        missing = [m for m in missing if m != "lm_head.weight"]
    if missing:
        raise RuntimeError(f"Checkpoint mismatch! Missing keys: {missing[:10]}")
    if unexpected:
        print(f"[bolt3] Warning: ignored {len(unexpected)} unexpected checkpoint keys (e.g. {unexpected[:5]})", flush=True)

    if cfg.get("tie_word_embeddings", True) or "lm_head.weight" not in sd_keys:
        model.lm_head.weight = model.embed_tokens.weight

    if device is not None:
        dev = torch.device(device)
        for n, p in model.named_parameters():
            if any(k in n for k in FP32_KEYS):
                p.data = p.data.to(dev, dtype=torch.float32)
            else:
                p.data = p.data.to(dev, dtype=(torch.bfloat16 if dev.type == "cuda" else torch.float32))
        for b in model.buffers():
            b.data = b.data.to(dev)

    if eval_mode:
        model.eval()
    return model, cfg, donor_id, meta.get("step", "?"), meta
