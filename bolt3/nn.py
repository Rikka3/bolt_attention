"""Shared Transformer building blocks for Bolt-3 student and Softmax teacher replicas."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, weight_offset: float = 0.0):
        super().__init__()
        self.weight_offset = float(weight_offset)
        self.weight = nn.Parameter(torch.zeros(dim) if self.weight_offset != 0.0 else torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_offset == 0.0 and x.dtype == self.weight.dtype and hasattr(F, "rms_norm"):
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        w = (self.weight.float() + self.weight_offset) if self.weight_offset != 0.0 else self.weight.float()
        return (xf * w).to(dt)


class RMSNormGated(nn.Module):
    """Standard gated RMSNorm (`weight * norm(x) * silu(gate)`), matching Qwen3.5 GatedDeltaNet."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dt = hidden_states.dtype
        hs = hidden_states.float()
        hs = hs * torch.rsqrt(hs.pow(2).mean(-1, keepdim=True) + self.eps)
        hs = self.weight * hs.to(dt)
        hs = hs * F.silu(gate.float())
        return hs.to(dt)


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
    if rope_scaling:
        rtype = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        if rtype == "linear":
            inv = inv / float(rope_scaling.get("factor", 1.0))
        elif rtype == "llama3":
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


def layer_rope_params(cfg: dict, layer_idx: int) -> tuple[float, dict | None]:
    """Return per-layer `(rope_theta, rope_scaling)` supporting hybrid local/global configs like Gemma-3."""
    thetas = cfg.get("layer_rope_thetas")
    scalings = cfg.get("layer_rope_scalings")
    theta = float(thetas[layer_idx]) if (thetas and layer_idx < len(thetas)) else float(cfg["rope_theta"])
    scaling = scalings[layer_idx] if (scalings and layer_idx < len(scalings)) else cfg.get("rope_scaling")
    return theta, scaling


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
        self.hidden_act = str(cfg.get("hidden_act", "silu")).lower()
        self.gate_proj = nn.Linear(D, I, bias=False)
        self.up_proj = nn.Linear(D, I, bias=False)
        self.down_proj = nn.Linear(I, D, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_proj(x)
        if self.hidden_act in ("gelu_pytorch_tanh", "gelu_new"):
            act = F.gelu(g, approximate="tanh")
        elif self.hidden_act == "gelu":
            act = F.gelu(g)
        else:
            act = F.silu(g)
        return self.down_proj(act * self.up_proj(x))


class SoftmaxAttn(nn.Module):
    """Standard causal Softmax attention used for donor teacher replicas."""

    def __init__(self, cfg: dict, layer_idx: int = 0):
        super().__init__()
        D, H, HKV = int(cfg["dim"]), int(cfg["num_heads"]), int(cfg["num_kv_heads"])
        HD = int(cfg.get("head_dim") or (D // H))
        qkv_bias = bool(cfg.get("qkv_bias", False))
        self.H, self.HKV, self.HD, self.G = H, HKV, HD, H // HKV
        self.packed_q_gate = bool(cfg.get("packed_q_gate", False))
        q_scalar = cfg.get("query_pre_attn_scalar")
        self.attn_scale = (float(q_scalar) ** -0.5) if q_scalar is not None else (HD ** -0.5)
        sws = cfg.get("layer_sliding_windows")
        self.sliding_window = int(sws[layer_idx]) if (sws and layer_idx < len(sws) and sws[layer_idx]) else None

        q_out_dim = (H * HD * 2) if self.packed_q_gate else (H * HD)
        self.q_proj = nn.Linear(D, q_out_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.v_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.o_proj = nn.Linear(H * HD, D, bias=False)

        if cfg.get("qk_norm", False):
            norm_off = float(cfg.get("rms_norm_offset", 0.0))
            self.q_norm = RMSNorm(HD, float(cfg["eps"]), weight_offset=norm_off)
            self.k_norm = RMSNorm(HD, float(cfg["eps"]), weight_offset=norm_off)
        else:
            self.q_norm, self.k_norm = None, None

        self.g_proj = (
            nn.Linear(D, H * HD, bias=False)
            if (cfg.get("softmax_gate", False) and not self.packed_q_gate)
            else None
        )
        r_theta, r_scaling = layer_rope_params(cfg, int(layer_idx))
        self.rope = RoPE(
            HD,
            r_theta,
            r_scaling,
            float(cfg.get("partial_rotary_factor", 1.0)),
        )

    def head_qkv_gate(self, x: torch.Tensor):
        B, T, _ = x.shape
        if self.packed_q_gate:
            q, gate = torch.chunk(self.q_proj(x).view(B, T, self.H, self.HD * 2), 2, dim=-1)
            gate = gate.reshape(B, T, self.H * self.HD)
        else:
            q = self.q_proj(x).view(B, T, self.H, self.HD)
            gate = None
        k = self.k_proj(x).view(B, T, self.HKV, self.HD)
        v = self.v_proj(x).view(B, T, self.HKV, self.HD)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), gate

    def head_qkv(self, x: torch.Tensor):
        q, k, v, _ = self.head_qkv_gate(x)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v, gate = self.head_qkv_gate(x)
        q, k = self.rope(q), self.rope(k)
        if getattr(self, "_capture_evict_target", False):
            C = int(getattr(self, "_evict_C", 128))
            loc_chunks = int(getattr(self, "_evict_loc_chunks", 4))
            n_sinks = int(getattr(self, "_evict_n_sinks", 4))
            ke = repeat_kv(k, self.G) if self.G > 1 else k
            ve = repeat_kv(v, self.G) if self.G > 1 else v
            pos = torch.arange(T, device=x.device)
            y_parts, w_parts = [], []
            with torch.no_grad():
                for qs in range(0, T, 256):
                    qe = min(T, qs + 256)
                    q_s = q[:, :, qs:qe]
                    scores = torch.matmul(q_s.float(), ke[:, :, :qe].transpose(-2, -1).float()) * self.attn_scale
                    qp = pos[qs:qe].view(1, 1, qe - qs, 1)
                    kp = pos[:qe].view(1, 1, 1, qe)
                    causal = kp <= qp
                    if self.sliding_window is not None and T > self.sliding_window:
                        causal = causal & (kp > (qp - self.sliding_window))
                    probs = F.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
                    y_parts.append(torch.matmul(probs.to(ve.dtype), ve[:, :, :qe]))
                    lo_k = ((qp // C) - (loc_chunks - 1)).clamp_min(0) * C
                    ev_m = (kp >= n_sinks) & (kp < lo_k)
                    w_parts.append(probs.masked_fill(~ev_m, 0.0).sum(dim=-1).mean(dim=1))
            y = torch.cat(y_parts, dim=2)
            self._evict_aux = (y.transpose(1, 2).reshape(B, T, self.H * self.HD).detach(), torch.cat(w_parts, dim=1).detach())
        else:
            attn_mask = None
            is_causal = True
            if self.sliding_window is not None and T > self.sliding_window:
                qi = torch.arange(T, device=x.device)[:, None]
                kj = torch.arange(T, device=x.device)[None, :]
                attn_mask = (kj <= qi) & (kj > (qi - self.sliding_window))
                is_causal = False
            try:
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=self.attn_scale, enable_gqa=(self.G > 1)
                )
            except TypeError:
                y = F.scaled_dot_product_attention(
                    q, repeat_kv(k, self.G), repeat_kv(v, self.G), attn_mask=attn_mask, is_causal=is_causal, scale=self.attn_scale
                )
        out = y.transpose(1, 2).reshape(B, T, self.H * self.HD)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        elif self.g_proj is not None:
            out = out * torch.sigmoid(self.g_proj(x))
        return self.o_proj(out)


def _l2norm_last(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked UT-solve GatedDeltaNet recurrence (float32 state, matches Qwen3.5 HF)."""
    init_dt = query.dtype
    b, seq_len, _, hk = key.shape
    nv, hv = value.shape[-2:]
    query, key, value, beta, decay = [
        t.transpose(1, 2).to(torch.float32).contiguous() for t in (query, key, value, beta, g)
    ]
    query = _l2norm_last(query) * (hk ** -0.5)
    key = _l2norm_last(key)

    pad_sz = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_sz:
        query, key, value = (F.pad(t, (0, 0, 0, pad_sz)) for t in (query, key, value))
        beta, decay = (F.pad(t, (0, pad_sz)) for t in (beta, decay))

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [
        t.reshape(b, nv, -1, chunk_size, t.shape[-1]) for t in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(b, nv, -1, chunk_size)
    n_chunks = decay.shape[2]

    triu_m = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cum_dec = decay.cumsum(dim=3)
    pw_dec = (cum_dec.unsqueeze(4) - cum_dec.unsqueeze(3)).masked_fill(triu_m, float("-inf")).exp()

    ut_sys = (k_beta @ key.transpose(-1, -2)) * pw_dec
    intra = (query @ key.transpose(-1, -2)) * pw_dec
    dec_kb = k_beta * cum_dec.exp().unsqueeze(-1)

    new_values = torch.linalg.solve_triangular(ut_sys, v_beta, upper=False, unitriangular=True)
    k_cumdec = torch.linalg.solve_triangular(ut_sys, dec_kb, upper=False, unitriangular=True)

    rec = (
        torch.zeros(b, nv, hk, hv, dtype=torch.float32, device=query.device)
        if initial_state is None
        else initial_state.to(torch.float32)
    )
    out = torch.empty_like(new_values)
    query = query * cum_dec.exp().unsqueeze(-1)
    key = key * (cum_dec[..., -1:] - cum_dec).exp().unsqueeze(-1)
    chk_dec = cum_dec[..., -1].exp()[..., None, None]

    for i in range(n_chunks):
        v_new = new_values[:, :, i] - k_cumdec[:, :, i] @ rec
        out[:, :, i] = query[:, :, i] @ rec + intra[:, :, i] @ v_new
        rec = rec * chk_dec[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new

    out = out.reshape(b, nv, -1, hv)[:, :, :seq_len].transpose(1, 2).contiguous().to(init_dt)
    return out, (rec if output_final_state else None)


class GatedDeltaNet(nn.Module):
    """Native O(1)-state GatedDeltaNet linear-attention block for hybrid models (e.g. Qwen3.5)."""

    def __init__(self, cfg: dict, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(cfg["dim"])
        self.num_v_heads = int(cfg.get("linear_num_value_heads", 32))
        self.num_k_heads = int(cfg.get("linear_num_key_heads", 16))
        self.head_k_dim = int(cfg.get("linear_key_head_dim", 128))
        self.head_v_dim = int(cfg.get("linear_value_head_dim", 128))
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.n_rep = self.num_v_heads // self.num_k_heads
        self.conv_kernel_size = int(cfg.get("linear_conv_kernel_dim", 4))
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.eps = float(cfg.get("eps", 1e-6))

        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0.01, 16.0)))
        self.norm = RMSNormGated(self.head_v_dim, eps=self.eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def _expand_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.n_rep <= 1:
            return q, k
        b, t, hk, d = q.shape
        q = q.unsqueeze(3).expand(b, t, hk, self.n_rep, d).reshape(b, t, self.num_v_heads, d)
        k = k.unsqueeze(3).expand(b, t, hk, self.n_rep, d).reshape(b, t, self.num_v_heads, d)
        return q, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.in_proj_qkv(x).transpose(1, 2)
        qkv = F.silu(self.conv1d(qkv)[:, :, :t]).transpose(1, 2)
        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(b, t, self.num_k_heads, self.head_k_dim)
        k = k.view(b, t, self.num_k_heads, self.head_k_dim)
        v = v.view(b, t, self.num_v_heads, self.head_v_dim)
        q, k = self._expand_qk(q, k)

        z = self.in_proj_z(x).view(b, t, self.num_v_heads, self.head_v_dim)
        beta = torch.sigmoid(self.in_proj_b(x))
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias.float())

        core, _ = _chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64, output_final_state=False)
        out = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)).view(b, t, self.value_dim)
        return self.out_proj(out)

    def step_graph(self, x: torch.Tensor, conv_state: torch.Tensor, rec_state: torch.Tensor) -> torch.Tensor:
        """Single-token (`T==1`) CUDA-Graph-safe decode step updating conv_state and rec_state in-place."""
        b = x.shape[0]
        qkv_t = self.in_proj_qkv(x).transpose(1, 2)  # (B, conv_dim, 1)
        comb = torch.cat([conv_state, qkv_t], dim=-1)  # (B, conv_dim, K)
        conv_state.copy_(comb[:, :, -(self.conv_kernel_size - 1) :])
        mixed = F.silu(F.conv1d(comb, self.conv1d.weight, bias=None, padding=0, groups=self.conv_dim).squeeze(-1))
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(b, 1, self.num_k_heads, self.head_k_dim)
        k = k.view(b, 1, self.num_k_heads, self.head_k_dim)
        q, k = self._expand_qk(q, k)
        q_t = (_l2norm_last(q.float()) * (self.head_k_dim ** -0.5)).squeeze(1)
        k_t = _l2norm_last(k.float()).squeeze(1)
        v_t = v.view(b, self.num_v_heads, self.head_v_dim).float()

        beta_t = torch.sigmoid(self.in_proj_b(x).squeeze(1)).float()
        g_t = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).squeeze(1).float() + self.dt_bias.float())

        rec_state.mul_(g_t.exp()[..., None, None])
        kv_mem = (rec_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t.unsqueeze(-1)
        rec_state.add_(k_t.unsqueeze(-1) * delta.unsqueeze(-2))
        core = (rec_state * q_t.unsqueeze(-1)).sum(dim=-2)

        z = self.in_proj_z(x).view(b * self.num_v_heads, self.head_v_dim)
        out = self.norm(core.to(x.dtype).reshape(b * self.num_v_heads, self.head_v_dim), z).view(b, 1, self.value_dim)
        return self.out_proj(out)

    def step_with_state(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        rec_state: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multi-token (`T>=1`) or padded recurrent step updating conv_state and rec_state in-place."""
        b, t, _ = x.shape
        if valid_mask is None and t == 1:
            return self.step_graph(x, conv_state, rec_state)

        x_in = x * valid_mask.unsqueeze(-1).to(x.dtype) if valid_mask is not None else x
        qkv_new = self.in_proj_qkv(x_in).transpose(1, 2)
        comb = torch.cat([conv_state, qkv_new], dim=-1)
        conv_state.copy_(comb[:, :, -(self.conv_kernel_size - 1) :])
        mixed = F.silu(F.conv1d(comb, self.conv1d.weight, bias=None, padding=0, groups=self.conv_dim)).transpose(1, 2)

        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(b, t, self.num_k_heads, self.head_k_dim)
        k = k.view(b, t, self.num_k_heads, self.head_k_dim)
        v = v.view(b, t, self.num_v_heads, self.head_v_dim)
        q, k = self._expand_qk(q, k)

        z = self.in_proj_z(x_in).view(b, t, self.num_v_heads, self.head_v_dim)
        beta = torch.sigmoid(self.in_proj_b(x_in))
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x_in).float() + self.dt_bias.float())
        if valid_mask is not None:
            vm = valid_mask.unsqueeze(-1)
            beta = beta * vm.to(beta.dtype)
            g = g * vm.to(g.dtype)

        core, new_rec = _chunk_gated_delta_rule(
            q, k, v, g, beta, chunk_size=64, initial_state=rec_state, output_final_state=True
        )
        rec_state.copy_(new_rec)
        out = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)).view(b, t, self.value_dim)
        return self.out_proj(out)
