"""Shared Transformer building blocks for Bolt-3 student and Softmax teacher replicas."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == self.weight.dtype and hasattr(F, "rms_norm"):
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(dt)


def inv_freq_for_rope(
    head_dim: int,
    theta: float,
    rope_scaling: dict | None = None,
    partial_rotary_factor: float = 1.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, int]:
    rot_dim = int(head_dim * partial_rotary_factor)
    rot_dim = max(2, rot_dim - (rot_dim % 2))
    inv = 1.0 / (
        theta ** (torch.arange(0, rot_dim, 2, dtype=torch.int64, device=device).float() / rot_dim)
    )
    if rope_scaling and rope_scaling.get("rope_type") == "llama3":
        old_ctx = float(rope_scaling["original_max_position_embeddings"])
        factor = float(rope_scaling["factor"])
        low = float(rope_scaling["low_freq_factor"])
        high = float(rope_scaling["high_freq_factor"])
        low_wl = old_ctx / low
        high_wl = old_ctx / high
        wavelen = 2.0 * math.pi / inv
        inv_a = torch.where(wavelen > low_wl, inv / factor, inv)
        smooth = (old_ctx / wavelen - low) / (high - low)
        smoothed = (1.0 - smooth) * inv_a / factor + smooth * inv_a
        medium = ~(wavelen < high_wl) & ~(wavelen > low_wl)
        inv = torch.where(medium, smoothed, inv_a)
    return inv, rot_dim


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope_cos_sin(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rot_dim: int) -> torch.Tensor:
    """Apply precomputed `(cos, sin)` tables of shape `(..., T, rot_dim)` to `(B, H, T, HD)`."""
    if rot_dim < x.shape[-1]:
        xr, xp = x[..., :rot_dim], x[..., rot_dim:]
        return torch.cat(((xr * cos) + (rotate_half(xr) * sin), xp), dim=-1)
    return (x * cos) + (rotate_half(x) * sin)


def apply_rope_pos(x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to (B, H, S, HD) keys with per-slot positions (B, H, S)."""
    rot_dim = inv_freq.numel() * 2
    freqs = pos.unsqueeze(-1).float() * inv_freq.view(1, 1, 1, -1)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(x.dtype)
    sin = emb.sin().to(x.dtype)
    return apply_rope_cos_sin(x, cos, sin, rot_dim)


class RoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,
        theta: float = 10000.0,
        rope_scaling: dict | None = None,
        partial_rotary_factor: float = 1.0,
    ):
        super().__init__()
        buf, rot_dim = inv_freq_for_rope(
            head_dim, theta, rope_scaling, partial_rotary_factor, torch.device("cpu")
        )
        self.rotary_dim = rot_dim
        self.register_buffer("inv_freq", buf, persistent=False)

    def cos_sin(
        self, pos_start: int, T: int, device: torch.device, dtype: torch.dtype, pos_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device)
        if pos_ids is None:
            t = torch.arange(pos_start, pos_start + T, device=device, dtype=torch.float32)
            freqs = t[:, None] * inv_freq
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]
        freqs = pos_ids.unsqueeze(1).unsqueeze(-1).float() * inv_freq.view(1, 1, 1, -1)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, x: torch.Tensor, pos_start: int = 0) -> torch.Tensor:
        cos, sin = self.cos_sin(pos_start, x.shape[2], x.device, x.dtype)
        return apply_rope_cos_sin(x, cos, sin, self.rotary_dim)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, H, T, HD = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, T, HD).reshape(B, H * n_rep, T, HD)


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        D, I = int(cfg["dim"]), int(cfg["intermediate"])
        self.gate_proj = nn.Linear(D, I, bias=False)
        self.up_proj = nn.Linear(D, I, bias=False)
        self.down_proj = nn.Linear(I, D, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SoftmaxAttn(nn.Module):
    """Standard causal Softmax attention used for donor teacher replicas."""

    def __init__(self, cfg: dict):
        super().__init__()
        D, H, HKV = int(cfg["dim"]), int(cfg["num_heads"]), int(cfg["num_kv_heads"])
        HD = int(cfg.get("head_dim") or (D // H))
        qkv_bias = bool(cfg.get("qkv_bias", False))
        self.H, self.HKV, self.HD, self.G = H, HKV, HD, H // HKV
        self.packed_q_gate = bool(cfg.get("packed_q_gate", False))

        q_out_dim = (H * HD * 2) if self.packed_q_gate else (H * HD)
        self.q_proj = nn.Linear(D, q_out_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.v_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.o_proj = nn.Linear(H * HD, D, bias=False)

        if cfg.get("qk_norm", False):
            self.q_norm = RMSNorm(HD, float(cfg["eps"]))
            self.k_norm = RMSNorm(HD, float(cfg["eps"]))
        else:
            self.q_norm, self.k_norm = None, None

        self.g_proj = (
            nn.Linear(D, H * HD, bias=False)
            if (cfg.get("softmax_gate", False) and not self.packed_q_gate)
            else None
        )
        self.rope = RoPE(
            HD,
            float(cfg["rope_theta"]),
            cfg.get("rope_scaling"),
            float(cfg.get("partial_rotary_factor", 1.0)),
        )

    def head_qkv(self, x: torch.Tensor):
        B, T, _ = x.shape
        if self.packed_q_gate:
            q, _ = torch.chunk(self.q_proj(x).view(B, T, self.H, self.HD * 2), 2, dim=-1)
        else:
            q = self.q_proj(x).view(B, T, self.H, self.HD)
        k = self.k_proj(x).view(B, T, self.HKV, self.HD)
        v = self.v_proj(x).view(B, T, self.HKV, self.HD)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        if self.packed_q_gate:
            q, gate = torch.chunk(self.q_proj(x).view(B, T, self.H, self.HD * 2), 2, dim=-1)
            gate = gate.reshape(B, T, self.H * self.HD)
            k = self.k_proj(x).view(B, T, self.HKV, self.HD)
            v = self.v_proj(x).view(B, T, self.HKV, self.HD)
            if self.q_norm is not None:
                q, k = self.q_norm(q), self.k_norm(k)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            q, k = self.rope(q), self.rope(k)
            try:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.G > 1))
            except TypeError:
                y = F.scaled_dot_product_attention(q, repeat_kv(k, self.G), repeat_kv(v, self.G), is_causal=True)
            return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.HD) * torch.sigmoid(gate))

        q, k, v = self.head_qkv(x)
        q, k = self.rope(q), self.rope(k)
        try:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.G > 1))
        except TypeError:
            y = F.scaled_dot_product_attention(q, repeat_kv(k, self.G), repeat_kv(v, self.G), is_causal=True)
        if self.g_proj is not None:
            y = y * torch.sigmoid(self.g_proj(x).view(B, T, self.H, self.HD).transpose(1, 2))
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.HD))
