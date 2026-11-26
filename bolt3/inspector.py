"""HuggingFace donor inspection, safetensor loading, and weight mapping for Bolt-3."""
from __future__ import annotations

import glob
import json
import os
import torch


def fetch_any_donor(repo_or_dir: str, hf_token: str | None = None) -> str:
    if os.path.isdir(repo_or_dir):
        return repo_or_dir
    from huggingface_hub import snapshot_download

    tok = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    pats = [
        "config.json",
        "generation_config.json",
        "*.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "added_tokens.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    path = snapshot_download(repo_or_dir, allow_patterns=pats, token=tok)
    print(f"[bolt3-convert] Donor snapshot ready: {path}")
    return path


def load_all_safetensors(donor_dir: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    shards = sorted(glob.glob(os.path.join(donor_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found in {donor_dir}")
    sd = {}
    for sh in shards:
        for k, v in load_file(sh).items():
            if any(
                tag in k
                for tag in (
                    "visual.",
                    "vision_tower.",
                    "multi_modal_projector.",
                    "mtp.",
                )
            ) and not k.startswith(("model.layers.", "language_model.model.layers.")):
                continue
            if k.startswith(("visual.", "model.visual.", "mtp.")) or ".visual." in k or ".mtp." in k:
                continue
            if k.startswith(("vision_tower.", "model.vision_tower.", "multi_modal_projector.", "model.multi_modal_projector.")):
                continue
            if k.startswith("model.language_model."):
                k = "model." + k[len("model.language_model.") :]
            elif k.startswith("language_model.model."):
                k = "model." + k[len("language_model.model.") :]
            elif k.startswith("language_model."):
                k = k[len("language_model.") :]
            sd[k] = v
    return sd


def inspect_donor_config(
    donor_dir: str,
    hf_sd: dict[str, torch.Tensor],
    max_T: int = 2048,
    bolt3_S: int = 128,
    bolt3_C: int = 128,
    bolt3_freeze_proj: bool = True,
    bolt3_cold_layers: list[int] | tuple[int, ...] | None = None,
    bolt3_max_cold_blocks: int = 256,
    bolt3_cold_topk: int = 0,
) -> dict:
    with open(os.path.join(donor_dir, "config.json")) as f:
        raw_cfg = json.load(f)
    hc = raw_cfg.get("text_config", raw_cfg)
    dim = int(hc["hidden_size"])
    num_layers = int(hc["num_hidden_layers"])
    num_heads = int(hc["num_attention_heads"])
    num_kv_heads = int(hc.get("num_key_value_heads", num_heads))
    head_dim = int(hc.get("head_dim") or (dim // num_heads))
    intermediate = int(hc["intermediate_size"])
    vocab_size = int(hc["vocab_size"])
    eps = float(hc.get("rms_norm_eps", 1e-6))
    rope_params = hc.get("rope_parameters") or {}
    rope_theta = float(hc.get("rope_theta", rope_params.get("rope_theta", 10000.0)))
    partial_rotary_factor = float(
        hc.get("partial_rotary_factor", rope_params.get("partial_rotary_factor", 1.0))
    )
    rope_scaling = hc.get("rope_scaling")

    if "lm_head.weight" in hf_sd and not bool(
        hc.get("tie_word_embeddings", raw_cfg.get("tie_word_embeddings", False))
    ):
        tie_emb = False
    else:
        tie_emb = "lm_head.weight" not in hf_sd or bool(
            hc.get("tie_word_embeddings", raw_cfg.get("tie_word_embeddings", True))
        )

    has_qkv_bias = any(k.endswith("self_attn.q_proj.bias") for k in hf_sd)
    has_qk_norm = any(k.endswith("self_attn.q_norm.weight") for k in hf_sd)
    packed_q_gate = bool(hc.get("attn_output_gate", False)) or any(
        k.endswith("self_attn.q_proj.weight") and t.shape[0] == 2 * num_heads * head_dim
        for k, t in hf_sd.items()
    )
    raw_lts = hc.get("layer_types")
    if raw_lts and "linear_attention" in raw_lts:
        layer_types = ["linear_attention" if lt == "linear_attention" else "bolt3" for lt in raw_lts]
    else:
        layer_types = ["bolt3"] * num_layers

    mtype = str(hc.get("model_type") or raw_cfg.get("model_type") or "").lower()
    is_gemma = mtype in ("gemma", "gemma2", "gemma3", "gemma3_text")
    rms_norm_offset = 1.0 if (is_gemma or mtype in ("qwen3_5", "qwen3_5_text")) else 0.0
    embed_scale = float(dim ** 0.5) if is_gemma else 1.0
    hidden_act = str(hc.get("hidden_activation") or hc.get("hidden_act") or "silu")
    post_norms = any(k.endswith(".pre_feedforward_layernorm.weight") for k in hf_sd)
    q_scalar = hc.get("query_pre_attn_scalar")
    query_pre_attn_scalar = float(q_scalar) if q_scalar is not None else None

    sw_pat = hc.get("sliding_window_pattern")
    sw_val = hc.get("sliding_window")
    raw_roles = hc.get("layer_types")
    if raw_roles is None and sw_pat and int(sw_pat) > 1:
        pat = int(sw_pat)
        raw_roles = ["sliding_attention" if bool((i + 1) % pat) else "full_attention" for i in range(num_layers)]

    layer_rope_thetas = None
    layer_rope_scalings = None
    layer_sliding_windows = None
    layer_roles = None
    if raw_roles and ("sliding_attention" in raw_roles and "full_attention" in raw_roles):
        layer_roles = list(raw_roles)
        sl_rp = rope_params.get("sliding_attention") if isinstance(rope_params.get("sliding_attention"), dict) else {}
        fl_rp = rope_params.get("full_attention") if isinstance(rope_params.get("full_attention"), dict) else {}
        sl_theta = float(hc.get("rope_local_base_freq", sl_rp.get("rope_theta", 10000.0)))
        fl_theta = float(fl_rp.get("rope_theta", rope_theta))
        fl_scaling = rope_scaling if rope_scaling is not None else (
            {k: v for k, v in fl_rp.items() if k != "rope_theta"} if fl_rp.get("rope_type", "default") != "default" else None
        )
        sl_scaling = (
            {k: v for k, v in sl_rp.items() if k != "rope_theta"}
            if sl_rp.get("rope_type", "default") != "default"
            else None
        )
        sw_int = int(sw_val) if sw_val else None
        layer_rope_thetas = [sl_theta if r == "sliding_attention" else fl_theta for r in layer_roles]
        layer_rope_scalings = [sl_scaling if r == "sliding_attention" else fl_scaling for r in layer_roles]
        layer_sliding_windows = [sw_int if r == "sliding_attention" else None for r in layer_roles]

    cfg_out = dict(
        donor_arch=raw_cfg.get("model_type", hc.get("model_type", "unknown")),
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate=intermediate,
        max_T=int(max_T),
        rope_theta=rope_theta,
        partial_rotary_factor=partial_rotary_factor,
        rope_scaling=rope_scaling,
        eps=eps,
        rms_norm_offset=rms_norm_offset,
        embed_scale=embed_scale,
        hidden_act=hidden_act,
        post_norms=post_norms,
        query_pre_attn_scalar=query_pre_attn_scalar,
        qkv_bias=has_qkv_bias,
        qk_norm=has_qk_norm,
        packed_q_gate=packed_q_gate,
        tie_word_embeddings=tie_emb,
        is_bolt3=True,
        bolt3_S=int(bolt3_S),
        bolt3_C=int(bolt3_C),
        bolt3_n_sinks=4,
        bolt3_sim_k=0.94,
        bolt3_sim_v=0.88,
        bolt3_gamma_u=0.985,
        bolt3_freeze_proj=bool(bolt3_freeze_proj),
        bolt3_cold_layers=[int(i) for i in (bolt3_cold_layers or ())],
        bolt3_max_cold_blocks=int(bolt3_max_cold_blocks),
        bolt3_cold_topk=int(bolt3_cold_topk),
        layer_types=layer_types,
    )
    if layer_roles is not None:
        cfg_out.update(
            layer_roles=layer_roles,
            layer_rope_thetas=layer_rope_thetas,
            layer_rope_scalings=layer_rope_scalings,
            layer_sliding_windows=layer_sliding_windows,
        )
    if "linear_attention" in layer_types:
        cfg_out.update(
            linear_conv_kernel_dim=int(hc.get("linear_conv_kernel_dim", 4)),
            linear_key_head_dim=int(hc.get("linear_key_head_dim", 128)),
            linear_num_key_heads=int(hc.get("linear_num_key_heads", 16)),
            linear_value_head_dim=int(hc.get("linear_value_head_dim", 128)),
            linear_num_value_heads=int(hc.get("linear_num_value_heads", 32)),
        )
    return cfg_out


def map_hf_to_bolt3(hf_sd: dict[str, torch.Tensor], cfg: dict, target_attn: str = "bolt3") -> dict[str, torch.Tensor]:
    """Map HF Transformer safetensors into Bolt3LM (`target_attn='bolt3'` or `'softmax'` for teacher)."""
    sd: dict[str, torch.Tensor] = {}
    H, HD, D = int(cfg["num_heads"]), int(cfg["head_dim"]), int(cfg["dim"])
    keep_gate = bool(cfg.get("packed_q_gate", False))
    use_post_norms = bool(cfg.get("post_norms", False))

    for k, t in hf_sd.items():
        if k == "model.embed_tokens.weight":
            sd["embed_tokens.weight"] = t.detach().cpu().clone()
        elif k == "model.norm.weight":
            sd["norm.weight"] = t.detach().cpu().clone()
        elif k == "lm_head.weight":
            sd["lm_head.weight"] = t.detach().cpu().clone()
        elif k.startswith("model.layers."):
            parts = k.split(".")
            li, sub = int(parts[2]), ".".join(parts[3:])
            if sub.startswith("self_attn."):
                proj = sub.split(".", 1)[1]
                w = t.detach().cpu().clone()
                if proj == "q_proj.weight" and w.shape[0] == 2 * H * HD and target_attn == "bolt3" and not keep_gate:
                    q_part, _ = torch.chunk(w.view(H, 2, HD, D), 2, dim=1)
                    sd[f"blocks.{li}.attn.q_proj.weight"] = q_part.reshape(H * HD, D).clone()
                elif proj in (
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "o_proj.weight",
                    "q_proj.bias",
                    "k_proj.bias",
                    "v_proj.bias",
                    "q_norm.weight",
                    "k_norm.weight",
                ):
                    sd[f"blocks.{li}.attn.{proj}"] = w
            elif sub.startswith("linear_attn."):
                sub_name = sub.split(".", 1)[1]
                if sub_name in (
                    "A_log",
                    "conv1d.weight",
                    "dt_bias",
                    "in_proj_a.weight",
                    "in_proj_b.weight",
                    "in_proj_qkv.weight",
                    "in_proj_z.weight",
                    "norm.weight",
                    "out_proj.weight",
                ):
                    sd[f"blocks.{li}.attn.{sub_name}"] = t.detach().cpu().clone()
            elif sub == "input_layernorm.weight":
                sd[f"blocks.{li}.norm1.weight"] = t.detach().cpu().clone()
            elif sub == "pre_feedforward_layernorm.weight":
                sd[f"blocks.{li}.norm2.weight"] = t.detach().cpu().clone()
            elif sub == "post_attention_layernorm.weight":
                key_dst = f"blocks.{li}.post_attn_norm.weight" if use_post_norms else f"blocks.{li}.norm2.weight"
                sd[key_dst] = t.detach().cpu().clone()
            elif sub == "post_feedforward_layernorm.weight" and use_post_norms:
                sd[f"blocks.{li}.post_mlp_norm.weight"] = t.detach().cpu().clone()
            elif sub in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
                sd[f"blocks.{li}.{sub}"] = t.detach().cpu().clone()

    if cfg["tie_word_embeddings"] or "lm_head.weight" not in sd:
        sd["lm_head.weight"] = sd["embed_tokens.weight"]
    return sd
