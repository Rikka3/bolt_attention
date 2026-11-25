"""Bolt-3 Donor Converter, Stage-1 Attention Transfer, Stage-2 KD Healer, Stage-3 Chain KD, & Scan."""
from __future__ import annotations

import gc
import math
import os
import random
import statistics
import sys
import threading
import time
import traceback
from array import array

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from .attention import Bolt3Attn, Bolt3Decoder
from .inspector import fetch_any_donor, inspect_donor_config, load_all_safetensors, map_hf_to_bolt3
from .model import FP32_KEYS, Bolt3LM, load_bolt3_checkpoint


def _purge_zombie_vram():
    for attr in ("last_traceback", "last_value", "last_type", "last_exc"):
        if hasattr(sys, attr):
            try:
                tb = getattr(sys, attr)
                if attr == "last_traceback" and tb is not None:
                    traceback.clear_frames(tb)
                elif attr in ("last_value", "last_exc") and hasattr(tb, "__traceback__") and tb.__traceback__ is not None:
                    traceback.clear_frames(tb.__traceback__)
                setattr(sys, attr, None)
            except Exception:
                pass
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _resolve_cold_layers(spec: list[int] | tuple[int, ...] | str | None, num_layers: int) -> list[int]:
    if not spec:
        return []
    if isinstance(spec, str):
        s = spec.strip().lower()
        if not s or s in ("none", "off", "0"):
            return []
        if s == "auto4":
            return [14, 19, 24, 29] if num_layers == 36 else [
                num_layers // 3,
                num_layers // 2,
                (2 * num_layers) // 3,
                max(0, num_layers - 7),
            ]
        if s == "auto6":
            return [12, 16, 20, 24, 28, 32] if num_layers == 36 else [
                max(0, int(round(num_layers * f))) for f in (0.33, 0.44, 0.56, 0.67, 0.78, 0.89)
            ]
        return sorted({int(x) for x in s.split(",") if x.strip()})
    return sorted({int(x) for x in spec})


def convert_donor_model(
    donor_id_or_dir: str,
    out_ckpt: str,
    *,
    max_T: int = 2048,
    bolt3_S: int = 128,
    bolt3_C: int = 128,
    freeze_proj: bool = True,
    cold_layers: list[int] | tuple[int, ...] | str | None = None,
    max_cold_blocks: int = 256,
    cold_topk: int = 0,
    hf_token: str | None = None,
) -> str:
    """Convert any pretrained HF Transformer (Qwen3, Llama, Mistral, etc.) to Bolt-3 (optionally Cold-Armed Option B)."""
    t0 = time.perf_counter()
    donor_dir = fetch_any_donor(donor_id_or_dir, hf_token=hf_token)
    hf_sd = load_all_safetensors(donor_dir)
    cfg = inspect_donor_config(
        donor_dir,
        hf_sd,
        max_T=max_T,
        bolt3_S=bolt3_S,
        bolt3_C=bolt3_C,
        bolt3_freeze_proj=freeze_proj,
        bolt3_max_cold_blocks=max_cold_blocks,
        bolt3_cold_topk=cold_topk,
    )
    resolved_cold = _resolve_cold_layers(cold_layers, int(cfg["num_layers"]))
    cfg["bolt3_cold_layers"] = resolved_cold

    L = cfg["num_layers"]
    print("=" * 80)
    print(f"BOLT-3 FULL CONVERTER: {donor_id_or_dir}")
    print("=" * 80)
    print(
        f"  Donor arch    : {cfg['donor_arch']} (dim={cfg['dim']}, H={cfg['num_heads']}, "
        f"HKV={cfg['num_kv_heads']}, HD={cfg['head_dim']}, L={L})"
    )
    cold_tag = f" | cold_layers={resolved_cold} (max_blks={max_cold_blocks}, topk={cold_topk})" if resolved_cold else ""
    n_b3 = sum(1 for lt in cfg["layer_types"] if lt == "bolt3")
    n_lin = sum(1 for lt in cfg["layer_types"] if lt == "linear_attention")
    layout_str = f"{n_b3}/{L} Bolt3Attn + {n_lin}/{L} GatedDeltaNet" if n_lin else f"{n_b3}/{L} Bolt3Attn layers"
    print(
        f"  Bolt-3 layout : {layout_str} (0 pure softmax) | "
        f"S={bolt3_S}, C={bolt3_C}, max_T={max_T}, freeze_proj={freeze_proj}{cold_tag}"
    )

    stu = Bolt3LM(**cfg)
    sd = map_hf_to_bolt3(hf_sd, cfg, target_attn="bolt3")
    del hf_sd

    missing, unexpected = stu.load_state_dict(sd, strict=False)
    bolt3_init_suffixes = ("kappa", "log_gamma", "beta_proj.weight", "beta_proj.bias", "eta_proj.weight", "eta_proj.bias", "res_norm.weight")
    real_missing = [m for m in missing if not m.endswith(bolt3_init_suffixes)]
    if real_missing or unexpected:
        raise RuntimeError(f"Weight mapping mismatch! Missing: {real_missing[:10]}, Unexpected: {unexpected[:10]}")

    out_dir = os.path.dirname(os.path.abspath(out_ckpt))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_sd = {
        k: (v.cpu().float() if any(f in k for f in FP32_KEYS) else v.cpu().to(torch.bfloat16))
        for k, v in stu.state_dict().items()
    }
    n_params = sum(p.numel() for p in stu.parameters())
    tmp = out_ckpt + ".tmp"
    torch.save(
        {
            "model": save_sd,
            "step": 0,
            "config": stu.config,
            "injection": {
                "donor": donor_id_or_dir,
                "converter": "bolt3.converter v1",
            },
        },
        tmp,
    )
    os.replace(tmp, out_ckpt)
    print(f"  Weight load   : PASS ({n_params/1e9:.3f}B params)")
    print(f"  Saved ckpt    : {out_ckpt} ({time.perf_counter() - t0:.1f}s)")
    print("=" * 80)
    return out_ckpt


def _format_conversation_donor_native(tok, msgs: list[dict]) -> str:
    norm_m = []
    for m in msgs:
        raw_role = str(m.get("from", m.get("role", "user"))).lower()
        r = "system" if raw_role in ("system", "sys") else ("user" if raw_role in ("human", "user") else "assistant")
        c = str(m.get("value", m.get("content", "")))
        c = (
            c.replace("<|begin_of_thought|>", "<think>\n")
            .replace("<|end_of_thought|>", "\n</think>\n")
            .replace("<|begin_of_solution|>", "")
            .replace("<|end_of_solution|>", "")
        )
        norm_m.append({"role": r, "content": c})

    has_cot = any("<think>" in m["content"] and m["role"] == "assistant" for m in norm_m)
    if not has_cot:
        try:
            return tok.apply_chat_template(norm_m, tokenize=False)
        except Exception:
            return "\n".join(f"{x['role']}: {x['content']}" for x in norm_m)

    eos = tok.eos_token or ""
    parts, history = [], []
    for m in norm_m:
        if m["role"] != "assistant":
            history.append(m)
        else:
            try:
                prefix = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            except Exception:
                prefix = "\n".join(f"{x['role']}: {x['content']}" for x in history) + "\nassistant: "
            content = m["content"]
            if prefix.rstrip().endswith("<think>") and content.lstrip().startswith("<think>"):
                content = content.lstrip()[len("<think>") :].lstrip("\n")
            parts.append(prefix + content + eos)
            history = []
    return "\n".join(parts) if parts else "\n".join(f"{x['role']}: {x['content']}" for x in norm_m)


def _build_reasoning_and_eval_caches(
    work: str,
    donor_dir: str,
    donor_repo: str,
    target_tokens: int,
    max_T: int = 2048,
    profile: str | None = None,
):
    import pyarrow.parquet as pq
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    from .profiles import resolve_profile

    tok = AutoTokenizer.from_pretrained(donor_dir)
    prof = resolve_profile(tok, profile, donor_dir)
    safe_tag = donor_repo.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    mix_tag = "" if prof.name == "reasoning" else f"_{prof.name}"
    train_cache = os.path.join(work, f"{safe_tag}_bolt3{mix_tag}_train.pt")
    eval_cache = os.path.join(work, f"{safe_tag}_bolt3{mix_tag}_eval.pt")

    if os.path.isfile(train_cache) and os.path.isfile(eval_cache):
        tr_t = torch.load(train_cache, weights_only=True)
        ev_t = torch.load(eval_cache, weights_only=True)
        if len(tr_t) >= int(target_tokens * 0.8) and len(ev_t) >= 500_000:
            print(f"[bolt3-data] Loaded cache ({len(tr_t)/1e6:.1f}M train, {len(ev_t)/1e6:.2f}M eval)")
            return tr_t, ev_t

    eos_id = tok.eos_token_id if isinstance(tok.eos_token_id, int) else (tok.eos_token_id[0] if tok.eos_token_id else 0)
    bos_id = tok.bos_token_id if isinstance(tok.bos_token_id, int) else eos_id
    api = HfApi()
    t0 = time.perf_counter()

    def _tokenize_docs(docs: list[str], budget: int) -> tuple[array, int]:
        buf = array("i")
        for b_i in range(0, len(docs), 128):
            for ids in tok(docs[b_i : b_i + 128], add_special_tokens=False)["input_ids"]:
                if bos_id is not None:
                    buf.append(int(bos_id))
                buf.extend(ids)
                buf.append(int(eos_id))
            if len(buf) >= budget:
                break
        return buf, len(buf)

    cot_docs, chat_docs, code_docs = [], [], []
    for ds_id, target_list, max_f in (
        ("bespokelabs/Bespoke-Stratos-17k", cot_docs, 4),
        ("HuggingFaceH4/ultrachat_200k", chat_docs, 4),
    ):
        for rev in ("main", "refs/convert/parquet"):
            try:
                files = [
                    f
                    for f in api.list_repo_files(ds_id, revision=rev, repo_type="dataset")
                    if f.endswith(".parquet") and "train" in f
                ]
                if not files:
                    files = [f for f in api.list_repo_files(ds_id, revision=rev, repo_type="dataset") if f.endswith(".parquet")]
                for f in files[:max_f]:
                    lp = hf_hub_download(repo_id=ds_id, filename=f, revision=rev, repo_type="dataset")
                    for row in pq.read_table(lp).to_pylist():
                        msgs = row.get("messages") or row.get("conversations")
                        if isinstance(msgs, list):
                            target_list.append(_format_conversation_donor_native(tok, msgs))
                if target_list:
                    break
            except Exception:
                continue

    for code_ds_id in (
        "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "nickrosh/Evol-Instruct-Code-80k-v1",
        "iamtarun/python_code_instructions_18k_alpaca",
    ):
        try:
            cds = load_dataset(code_ds_id, split="train", streaming=True)
            for ex in cds:
                q_c = str(ex.get("problem") or ex.get("instruction") or ex.get("prompt") or "")
                a_c = str(ex.get("solution") or ex.get("output") or ex.get("response") or "")
                if q_c and a_c:
                    code_docs.append(
                        _format_conversation_donor_native(
                            tok, [{"role": "user", "content": q_c}, {"role": "assistant", "content": a_c}]
                        )
                    )
                if len(code_docs) >= 25_000:
                    break
            if len(code_docs) >= 10_000:
                break
        except Exception:
            continue

    rng = random.Random(7777)
    rng.shuffle(cot_docs)
    rng.shuffle(code_docs)
    rng.shuffle(chat_docs)

    eval_cot_cut = max(50, len(cot_docs) // 20)
    eval_code_cut = max(100, len(code_docs) // 20)
    eval_chat_cut = max(150, len(chat_docs) // 20)

    ev_cot_buf, _ = _tokenize_docs(cot_docs[:eval_cot_cut], 350_000)
    ev_code_buf, _ = _tokenize_docs(code_docs[:eval_code_cut], 350_000)
    ev_chat_buf, _ = _tokenize_docs(chat_docs[:eval_chat_cut], 350_000)
    eval_t = torch.cat(
        [
            torch.frombuffer(ev_cot_buf, dtype=torch.int32).clone(),
            torch.frombuffer(ev_code_buf, dtype=torch.int32).clone(),
            torch.frombuffer(ev_chat_buf, dtype=torch.int32).clone(),
        ]
    )

    cot_budget = int(target_tokens * prof.mix_cot)
    code_budget = int(target_tokens * prof.mix_code)
    chat_budget = max(0, target_tokens - cot_budget - code_budget)

    tr_cot_buf, n_tr_cot = _tokenize_docs(cot_docs[eval_cot_cut:], cot_budget)
    tr_code_buf, n_tr_code = _tokenize_docs(code_docs[eval_code_cut:], code_budget)
    tr_chat_buf, n_tr_chat = _tokenize_docs(chat_docs[eval_chat_cut:], chat_budget)

    T_chunk = max_T + 1
    chunks = []
    for buf in (tr_cot_buf, tr_code_buf, tr_chat_buf):
        t_arr = torch.frombuffer(buf, dtype=torch.int32).clone()
        n_c = len(t_arr) // T_chunk
        if n_c > 0:
            chunks.append(t_arr[: n_c * T_chunk].view(n_c, T_chunk))
    all_chunks = torch.cat(chunks, dim=0)
    perm = torch.randperm(all_chunks.size(0), generator=torch.Generator().manual_seed(7777))
    train_t = all_chunks[perm].reshape(-1)[:target_tokens].contiguous()

    os.makedirs(work, exist_ok=True)
    torch.save(train_t, train_cache)
    torch.save(eval_t, eval_cache)
    tot_tr = max(1, n_tr_cot + n_tr_code + n_tr_chat)
    print(
        f"[bolt3-data] Token mix -> Train: {len(train_t)/1e6:.1f}M "
        f"(CoT: {n_tr_cot/tot_tr*100:.0f}%, Code: {n_tr_code/tot_tr*100:.0f}%, Chat: {n_tr_chat/tot_tr*100:.0f}%) | "
        f"Eval: {len(eval_t)/1e6:.2f}M in {time.perf_counter()-t0:.1f}s!",
        flush=True,
    )
    return train_t, eval_t


def _load_donor_teacher(donor_dir: str, stu_cfg: dict, device: torch.device):
    hf_sd = load_all_safetensors(donor_dir)
    rep_cfg = dict(
        stu_cfg,
        layer_types=["softmax" if lt in ("bolt3", "vsa") else lt for lt in stu_cfg["layer_types"]],
    )
    donor = Bolt3LM(**rep_cfg)
    donor.load_state_dict(map_hf_to_bolt3(hf_sd, rep_cfg, target_attn="softmax"), strict=True)
    del hf_sd
    donor = donor.to(device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    for n_, p in donor.named_parameters():
        if any(k in n_ for k in FP32_KEYS):
            p.data = p.data.to(device, dtype=torch.float32)
        p.requires_grad_(False)
    donor.eval()
    return donor, False


def _load_tied_teacher(donor_repo: str, stu_cfg: dict, stu, device: torch.device):
    donor_dir = fetch_any_donor(donor_repo)
    donor, is_hf_donor = _load_donor_teacher(donor_dir, stu_cfg, device)
    if torch.equal(donor.embed_tokens.weight, stu.embed_tokens.weight):
        donor.embed_tokens.weight = stu.embed_tokens.weight
    if torch.equal(donor.lm_head.weight, stu.lm_head.weight):
        donor.lm_head.weight = stu.lm_head.weight
    _purge_zombie_vram()
    return donor, is_hf_donor, donor.lm_head.weight, donor_dir


def _forward_hf_teacher_body(donor, xm: torch.Tensor, num_layers: int, anchor_every: int):
    inner = getattr(donor, "model", donor)
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    out = inner(input_ids=xm, output_hidden_states=True, use_cache=False)
    h_teacher = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out.hidden_states[-1]).detach()
    t_mids = [
        out.hidden_states[li + 1].detach()
        for li in range(num_layers - 1)
        if anchor_every > 0 and (li + 1) % anchor_every == 0
    ]
    del out
    return h_teacher, t_mids


@torch.no_grad()
def _eval_reasoning_metrics(
    stu,
    donor,
    is_hf_donor: bool,
    eval_tokens: torch.Tensor,
    device: torch.device,
    n_tokens: int = 131_072,
    bs: int = 2,
    bos_id: int | None = None,
) -> tuple[float, float, float, float]:
    stu.eval()
    T, V = stu.config["max_T"], stu.config["vocab_size"]
    avail = len(eval_tokens) - 1
    n_b = max(1, min(n_tokens // (bs * T), avail // (bs * T)))
    ar = torch.arange(T)
    nll_s, nll_d, kl_sum, top1_ok, cnt = 0.0, 0.0, 0.0, 0, 0
    w_lm_s = stu.lm_head.weight
    w_lm_t = donor.get_output_embeddings().weight if is_hf_donor else donor.lm_head.weight
    anchor_every = int(stu.config.get("kd_anchor_every", 4))

    for i in range(n_b):
        s = i * bs * T
        base = s + torch.arange(bs)[:, None] * T + ar[None, :]
        x, y = eval_tokens[base].long().to(device), eval_tokens[base + 1].long().to(device)
        if bos_id is not None:
            x[:, 0] = int(bos_id)
        with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
            hs = stu(x)
            hd, _ = (
                _forward_hf_teacher_body(donor, x, stu.config["num_layers"], anchor_every)
                if is_hf_donor
                else (donor(x), None)
            )
        for c in range(bs):
            slog = F.linear(hs[c : c + 1].float(), w_lm_s.float())
            dlog = F.linear(hd[c : c + 1].float(), w_lm_t.float())
            yv = y[c : c + 1].reshape(-1)
            nll_s += F.cross_entropy(slog.reshape(-1, V), yv, reduction="sum").item()
            nll_d += F.cross_entropy(dlog.reshape(-1, V), yv, reduction="sum").item()
            lsm_s, lsm_d = F.log_softmax(slog, dim=-1), F.log_softmax(dlog, dim=-1)
            p_d = lsm_d.exp()
            kl_sum += float((p_d * (lsm_d - lsm_s)).sum().item())
            top1_ok += int((slog.argmax(dim=-1) == dlog.argmax(dim=-1)).sum().item())
            cnt += yv.numel()
            del slog, dlog, lsm_s, lsm_d, p_d
        del hs, hd, x, y

    return (
        math.exp(nll_s / max(1, cnt)),
        math.exp(nll_d / max(1, cnt)),
        kl_sum / max(1, cnt),
        (top1_ok / max(1, cnt)) * 100.0,
    )


def _attn_transfer_loss(a_hat: torch.Tensor, a_tgt: torch.Tensor) -> torch.Tensor:
    a_hat, a_tgt = a_hat.float(), a_tgt.float()
    rel_mse = F.mse_loss(a_hat, a_tgt) / a_tgt.pow(2).mean().clamp_min(1e-5)
    return rel_mse + (1.0 - F.cosine_similarity(a_hat, a_tgt, dim=-1).mean())


def _stage1_attention_transfer(
    stu,
    donor,
    is_hf_donor: bool,
    train_tokens: torch.Tensor,
    device: torch.device,
    stage1_tokens: int = 16_000_000,
    bs: int = 4,
    on_policy: bool = False,
):
    if stage1_tokens <= 0:
        return
    T = min(1024, int(stu.config["max_T"]))
    steps = max(1, stage1_tokens // (bs * T))
    L = int(stu.config["num_layers"])
    bolt3_indices = [li for li in range(L) if isinstance(stu.blocks[li].attn, Bolt3Attn)]
    if not bolt3_indices:
        return
    if on_policy and is_hf_donor:
        print("  [Stage-1] on-policy mode needs a native donor replica; using teacher-forced.", flush=True)
    on_policy_from = steps // 4 if (on_policy and not is_hf_donor) else None

    for p in stu.parameters():
        p.requires_grad_(False)
    bolt3_params = []
    _pretrained_proj = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
    for li in bolt3_indices:
        for n_p, p in stu.blocks[li].attn.named_parameters():
            if any(k in n_p for k in _pretrained_proj):
                p.requires_grad_(False)
            else:
                p.data = p.data.to(device, dtype=torch.float32)
                p.requires_grad_(True)
                bolt3_params.append(p)

    opt = torch.optim.AdamW(bolt3_params, lr=1e-3, weight_decay=0.0, fused=(device.type == "cuda"))
    n_win = max(1, (len(train_tokens) - 1) // T)
    gen = torch.Generator().manual_seed(101)
    ar = torch.arange(T)
    t0 = time.perf_counter()
    print(
        f"  [Stage-1 Attention Transfer] {stage1_tokens/1e6:.1f}M tokens ({steps} steps) across {len(bolt3_indices)} Bolt-3 layers...",
        flush=True,
    )

    stu.train()
    for s in range(steps):
        idx = torch.randint(0, n_win, (bs,), generator=gen)
        x = train_tokens[idx[:, None] * T + ar[None, :]].long().to(device)
        opt.zero_grad(set_to_none=True)

        if on_policy_from is not None and s == on_policy_from:
            print(f"    [Stage-1] step {s+1}: switching to on-policy student stream", flush=True)
        total_loss = torch.zeros((), device=device, dtype=torch.float32)

        if on_policy_from is not None and s >= on_policy_from:
            bolt3_set = set(bolt3_indices)
            with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
                with torch.no_grad():
                    xs = stu.embed_tokens(x)
                for li, blk in enumerate(stu.blocks):
                    with torch.no_grad():
                        h_n = blk.norm1(xs)
                        a_tgt = donor.blocks[li].attn(h_n)
                    if li in bolt3_set:
                        a_hat = blk.attn(h_n)
                        loss_li = _attn_transfer_loss(a_hat, a_tgt) / len(bolt3_indices)
                        loss_li.backward()
                        total_loss = total_loss + loss_li.detach()
                        xs = xs + a_hat.detach()
                        del a_hat, loss_li
                    else:
                        xs = xs + a_tgt
                    with torch.no_grad():
                        xs = xs + blk.mlp(blk.norm2(xs))
                    del a_tgt
            del xs
        else:
            norm_inputs, attn_targets = {}, {}
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
                xt = donor.embed_tokens(x)
                for li, blk in enumerate(donor.blocks):
                    h_n = blk.norm1(xt)
                    a_out = blk.attn(h_n)
                    if li in bolt3_indices:
                        norm_inputs[li] = h_n.detach()
                        attn_targets[li] = a_out.detach()
                    xt = xt + a_out
                    xt = xt + blk.mlp(blk.norm2(xt))
            with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
                for li, h_n in norm_inputs.items():
                    a_hat = stu.blocks[li].attn(h_n)
                    loss_li = _attn_transfer_loss(a_hat, attn_targets[li]) / max(1, len(norm_inputs))
                    loss_li.backward()
                    total_loss = total_loss + loss_li.detach()
                    del a_hat, loss_li
            del norm_inputs, attn_targets
        torch.nn.utils.clip_grad_norm_(bolt3_params, 1.0)
        opt.step()
        if (s + 1) in (1, max(1, steps // 4), max(1, steps // 2), steps):
            print(
                f"    [Stage-1] step {s+1:4d}/{steps} | attn_rel_mse+cos = {float(total_loss.item()):.4f} ({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )


def _setup_trainable(
    stu,
    stu_cfg: dict,
    device: torch.device,
    freeze_mlp: bool = False,
    train_proj: bool = False,
):
    bolt3_core, bolt3_proj, mlp_and_norm = [], [], []
    freeze_proj = bool(stu_cfg.get("bolt3_freeze_proj", True)) and not train_proj
    b3_layers = {f"blocks.{li}." for li, blk in enumerate(stu.blocks) if isinstance(blk.attn, Bolt3Attn)}
    bolt3_core_keys = ("log_gamma", "kappa", "beta_proj", "eta_proj", "res_norm")

    for n_, p_ in stu.named_parameters():
        in_b3 = any(n_.startswith(pfx) for pfx in b3_layers)
        if in_b3 and any(k in n_ for k in bolt3_core_keys):
            p_.requires_grad_(True)
            bolt3_core.append(p_)
        elif in_b3 and ".attn." in n_ and not freeze_proj:
            p_.requires_grad_(True)
            bolt3_proj.append(p_)
        elif not freeze_mlp and (".mlp." in n_ or ".norm1." in n_ or ".norm2." in n_):
            p_.requires_grad_(True)
            mlp_and_norm.append(p_)
        else:
            p_.requires_grad_(False)

    for n_, p_ in stu.named_parameters():
        keep_fp32 = p_.requires_grad or any(k in n_ for k in FP32_KEYS)
        p_.data = p_.data.to(
            device,
            dtype=(torch.float32 if keep_fp32 else (torch.bfloat16 if device.type == "cuda" else torch.float32)),
        )
    for b in stu.buffers():
        b.data = b.data.to(device)

    groups = [
        {"params": bolt3_core, "weight_decay": 0.0, "lr_scale": 5.0},
        {"params": bolt3_proj, "weight_decay": 0.01, "lr_scale": 1.0},
        {"params": mlp_and_norm, "weight_decay": 0.0, "lr_scale": 0.08},
    ]
    groups = [g for g in groups if g["params"]]
    trainable = bolt3_core + bolt3_proj + mlp_and_norm
    print(
        f"  Trainable     : {sum(p.numel() for p in trainable)/1e6:.2f}M params "
        f"(proj={'frozen' if freeze_proj else 'trained'}, MLPs={'frozen' if freeze_mlp else 'trained @0.08x'})"
    )
    return groups, trainable


def _save_student(stu, stu_cfg: dict, ck: dict, out_ckpt: str, step: int, opt=None, opt_ckpt: str | None = None, **extra):
    tmp = out_ckpt + ".tmp"
    save_sd = {
        k: (v.cpu().float() if any(f in k for f in FP32_KEYS) else v.cpu().to(torch.bfloat16))
        for k, v in stu.state_dict().items()
    }
    torch.save({"model": save_sd, "step": step, "config": stu_cfg, "injection": ck.get("injection", {}), **extra}, tmp)
    os.replace(tmp, out_ckpt)
    if opt is not None and opt_ckpt:
        torch.save(opt.state_dict(), opt_ckpt + ".tmp")
        os.replace(opt_ckpt + ".tmp", opt_ckpt)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return x.mean() if mask is None else (x * mask).sum() / mask.sum().clamp_min(1.0)


def _kd_microbatch_backward(
    stu,
    donor,
    is_hf_donor: bool,
    w_lm_t: torch.Tensor,
    xm: torch.Tensor,
    ym: torch.Tensor,
    w: float,
    kd: float,
    use_ckpt: bool,
    loss_mask: torch.Tensor | None = None,
    tok_slice: int = 512,
    ckpt_stride: int = 1,
) -> torch.Tensor:
    cfg = stu.config
    L_num = int(cfg["num_layers"])
    anchor_every = int(cfg.get("kd_anchor_every", 4))
    device = xm.device
    mask_f = loss_mask.float() if loss_mask is not None else None

    with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
        with torch.no_grad():
            if is_hf_donor:
                h_teacher, t_mids = _forward_hf_teacher_body(donor, xm, L_num, anchor_every)
            else:
                xt = donor.embed_tokens(xm)
                t_mids = []
                for li, blk in enumerate(donor.blocks):
                    xt = blk(xt)
                    if anchor_every > 0 and li < L_num - 1 and (li + 1) % anchor_every == 0:
                        t_mids.append(xt)
                h_teacher = donor.norm(xt)

        xs = stu.embed_tokens(xm)
        s_mids = []
        for li, blk in enumerate(stu.blocks):
            do_ckpt = use_ckpt and (li % max(1, ckpt_stride) == 0)
            xs = torch.utils.checkpoint.checkpoint(blk, xs, use_reentrant=False) if do_ckpt else blk(xs)
            if anchor_every > 0 and li < L_num - 1 and (li + 1) % anchor_every == 0:
                s_mids.append(xs)
        h_student = stu.norm(xs)

        def _cos_loss(a, b):
            return _masked_mean(1.0 - F.cosine_similarity(a.float(), b.float(), dim=-1), mask_f)

        mid_loss = (
            sum(_cos_loss(sm, tm) for sm, tm in zip(s_mids, t_mids)) / max(1, len(s_mids))
            if s_mids
            else torch.zeros((), device=device)
        )
        final_h_loss = _cos_loss(h_student, h_teacher)
        del s_mids, t_mids

    B_m, T_m, D_m = h_student.shape
    N_m = B_m * T_m
    hs_2d, ht_2d, ym_1d = h_student.reshape(N_m, D_m), h_teacher.reshape(N_m, D_m), ym.reshape(N_m)
    m_1d = None if mask_f is None else mask_f.reshape(N_m)
    denom = float(N_m) if m_1d is None else float(m_1d.sum().clamp_min(1.0).item())
    dh_2d = torch.empty_like(hs_2d)
    if stu.config["vocab_size"] > 200_000:
        tok_slice = min(tok_slice, 1024)
    w_lm_s_f32 = stu.lm_head.weight.float()
    w_lm_t_f32 = w_lm_t.float()
    loss = torch.zeros((), device=device, dtype=torch.float32)

    with torch.no_grad():
        for t_lo in range(0, N_m, tok_slice):
            t_hi = min(t_lo + tok_slice, N_m)
            hs_c = hs_2d[t_lo:t_hi].float()
            ht_c = ht_2d[t_lo:t_hi].float()
            ym_c = ym_1d[t_lo:t_hi]
            m_c = None if m_1d is None else m_1d[t_lo:t_hi]

            slog = F.linear(hs_c, w_lm_s_f32)
            lsm_s = F.log_softmax(slog, dim=-1)
            prob_s = lsm_s.exp()

            rlog = F.linear(ht_c, w_lm_t_f32)
            rlsm = F.log_softmax(rlog, dim=-1)
            prob_t = rlsm.exp()
            kd_tok = (prob_t * (rlsm - lsm_s)).sum(-1)
            kd_i = kd_tok.sum() if m_c is None else (kd_tok * m_c).sum()
            del rlog, rlsm, kd_tok

            if kd >= 0.999:
                loss.add_(kd_i * (w / denom))
                g_logits = (prob_s - prob_t) * (w / denom)
            else:
                ce_tok = -lsm_s.gather(-1, ym_c.unsqueeze(-1))[:, 0]
                ce_i = ce_tok.sum() if m_c is None else (ce_tok * m_c).sum()
                loss.add_(((1.0 - kd) * ce_i + kd * kd_i) * (w / denom))
                g_logits = prob_s
                g_logits.sub_(prob_t, alpha=kd)
                g_logits.scatter_add_(
                    -1, ym_c.unsqueeze(-1), torch.full_like(ym_c.unsqueeze(-1), -(1.0 - kd), dtype=torch.float32)
                )
                g_logits.mul_(w / denom)
            if m_c is not None:
                g_logits.mul_(m_c.unsqueeze(-1))

            dh_2d[t_lo:t_hi] = (g_logits @ w_lm_s_f32).to(h_student.dtype)
            del slog, lsm_s, prob_s, prob_t, g_logits

    del h_teacher, ht_2d, w_lm_s_f32, w_lm_t_f32
    torch.autograd.backward(
        [h_student, (mid_loss + final_h_loss) * (0.20 * w)],
        [dh_2d.view(B_m, T_m, D_m), torch.ones_like(final_h_loss)],
    )
    del h_student, hs_2d, dh_2d, mid_loss, final_h_loss
    return loss


def heal_bolt3_model(
    ckpt_path: str,
    *,
    donor_id: str | None = None,
    work: str = ".",
    out_prefix: str = "bolt3_healed",
    tokens_budget: int = 16_000_000,
    data_tokens: int = 100_000_000,
    stage1_tokens: int = 0,
    bs: int = 8,
    lr: float = 8e-5,
    warmup: int = 150,
    kd: float = 1.0,
    eval_every: int = 250,
    save_every: int = 50,
    seed: int = 7777,
    hf_repo: str | None = None,
    highspeed: bool = False,
    resume: bool = False,
    freeze_mlp: bool = True,
    train_proj: bool = False,
    stage1_onpolicy: bool = False,
    profile: str | None = None,
    ckpt_stride: int | None = None,
    micro_bs_override: int | None = None,
):
    _purge_zombie_vram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs(work, exist_ok=True)
    out_ckpt = os.path.join(work, f"{out_prefix}_latest.pth")
    opt_ckpt = os.path.join(work, f"{out_prefix}_optim.pth")
    active_ckpt = out_ckpt if (resume and os.path.isfile(out_ckpt)) else ckpt_path

    stu, stu_cfg, ck_donor, ck_step, ck = load_bolt3_checkpoint(active_ckpt, device=None, eval_mode=False)
    start_step = int(ck_step) if (resume and isinstance(ck_step, int)) else 0
    donor_repo = donor_id or ck_donor
    if not donor_repo:
        raise ValueError("Pass --donor <repo_id> or use a checkpoint created by `python -m bolt3 convert`")

    opt_groups, trainable = _setup_trainable(stu, stu_cfg, device, freeze_mlp=freeze_mlp, train_proj=train_proj)

    n_params = sum(p.numel() for p in stu.parameters())
    T = int(stu_cfg["max_T"])
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1024**3) if device.type == "cuda" else 16.0

    # Auto-tune activation checkpointing for 48 GB / 96 GB RTX 6000 Pro vs smaller GPUs
    if highspeed and vram_gb >= 80.0 and freeze_mlp and T <= 2048:
        micro_bs = 2
        use_ckpt = False
        eff_stride = 1
        mode_str = f"RTX-96GB HIGHSPEED (use_ckpt=False, micro_bs={micro_bs}, T={T})"
    elif highspeed and vram_gb >= 44.0 and freeze_mlp and T <= 2048:
        micro_bs = 2
        use_ckpt = True
        eff_stride = 2 if ckpt_stride is None else int(ckpt_stride)
        mode_str = f"RTX-48GB FAST-CKPT (use_ckpt=True, stride={eff_stride}, micro_bs={micro_bs}, T={T})"
    elif highspeed and vram_gb >= 32.0 and n_params < 3_000_000_000:
        micro_bs = 4 if T <= 1024 else 2
        use_ckpt = False
        eff_stride = 1
        mode_str = f"HIGHSPEED (use_ckpt=False, micro_bs={micro_bs}, T={T})"
    else:
        micro_bs = 4 if (vram_gb >= 80.0 and not freeze_mlp and T <= 1024) else 2
        use_ckpt = True
        eff_stride = 1 if ckpt_stride is None else int(ckpt_stride)
        mode_str = f"LOW-VRAM (use_ckpt=True, stride={eff_stride}, micro_bs={micro_bs}, T={T})"

    if ckpt_stride is not None:
        eff_stride = int(ckpt_stride)
    if micro_bs_override is not None:
        micro_bs = int(micro_bs_override)

    donor, is_hf_donor, w_lm_t, donor_dir = _load_tied_teacher(donor_repo, stu_cfg, stu, device)
    train_tokens, eval_tokens = _build_reasoning_and_eval_caches(
        work, donor_dir, donor_repo, max(data_tokens, int(tokens_budget * 1.05)), max_T=T, profile=profile
    )

    if start_step == 0 and stage1_tokens > 0:
        _stage1_attention_transfer(
            stu,
            donor,
            is_hf_donor,
            train_tokens,
            device,
            stage1_tokens=stage1_tokens,
            bs=(4 if vram_gb >= 45.0 else max(2, micro_bs)),
            on_policy=stage1_onpolicy,
        )
        for p_ in trainable:
            p_.data = p_.data.to(device, dtype=torch.float32)
            p_.requires_grad_(True)
        _purge_zombie_vram()
        _save_student(stu, stu_cfg, ck, os.path.join(work, f"{out_prefix}_stage1.pth"), 0)

    start_ppl, donor_ppl, start_kl, start_top1 = _eval_reasoning_metrics(stu, donor, is_hf_donor, eval_tokens, device, bs=2)
    total_steps = max(start_step + 1, math.ceil(tokens_budget / (bs * T)))

    print("=" * 84)
    print(f"BOLT-3 DISTILLATION HEALER: {donor_repo} (kd={kd}, ce={1.0-kd:.2f})")
    print("=" * 84)
    print(f"  Memory Mode   : {mode_str} ({n_params/1e9:.2f}B Student)")
    print(
        f"  Held-Out Eval : KL={start_kl:.4f} nats | Top1={start_top1:.2f}% | "
        f"Student PPL={start_ppl:.2f} / Donor PPL={donor_ppl:.2f} ({start_ppl/donor_ppl:.3f}x)"
    )
    print(f"  Stage-2 Budget: {tokens_budget/1e6:.1f}M tokens ({total_steps} steps, resume={start_step})")

    opt = torch.optim.AdamW(opt_groups, lr=lr, betas=(0.9, 0.95), fused=(device.type == "cuda"))
    if resume and os.path.isfile(opt_ckpt):
        try:
            opt.load_state_dict(torch.load(opt_ckpt, map_location=device, weights_only=False))
            print("  [Resume] Restored optimizer moments from sidecar.")
        except Exception as e:
            print(f"  [Resume] Optimizer sidecar skipped: {e}")

    stable_end = int(total_steps * 0.65)

    def set_step_lr(s: int):
        if s < warmup:
            m = (s + 1) / max(1, warmup)
        elif s < stable_end:
            m = 1.0
        else:
            prog = (s - stable_end) / max(1, total_steps - stable_end)
            m = 0.05 + 0.475 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        for g in opt.param_groups:
            g["lr"] = lr * m * g.get("lr_scale", 1.0)

    n_win = max(1, (len(train_tokens) - 1) // T)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(start_step):
        torch.randint(0, n_win, (bs,), generator=gen)
    ar = torch.arange(T)

    _purge_zombie_vram()
    stu.train()
    set_step_lr(start_step)
    t_window = time.perf_counter()
    step_window = start_step

    def _upload_hf(p: str, r: str, sync: bool = False):
        def _up():
            try:
                from huggingface_hub import HfApi

                token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                api = HfApi(token=token)
                api.create_repo(repo_id=r, repo_type="model", exist_ok=True)
                api.upload_file(path_or_fileobj=p, path_in_repo=os.path.basename(p), repo_id=r, repo_type="model")
                print(f"[bolt3-hf] Uploaded {os.path.basename(p)} -> https://huggingface.co/{r}", flush=True)
            except Exception as e:
                print(f"[bolt3-hf] Upload skipped: {e}", flush=True)

        if sync:
            _up()
        else:
            threading.Thread(target=_up, daemon=True).start()

    from transformers import AutoTokenizer

    binder_tok = AutoTokenizer.from_pretrained(donor_dir)
    last_step_done = start_step
    try:
        for step in range(start_step, total_steps):
            idx = torch.randint(0, n_win, (bs,), generator=gen)
            base = idx[:, None] * T + ar[None, :]
            x, y = train_tokens[base].long(), train_tokens[base + 1].long()
            if step % 4 == 0 and T >= 128:
                rng_n = random.Random(700000 + step)
                for r_i in range(bs):
                    s1, s2 = rng_n.sample(("Orion", "Hemlock", "Vanguard", "Chronos", "Lyra", "Titan"), 2)
                    c1, c2 = rng_n.randint(10000, 99999), rng_n.randint(10000, 99999)
                    nd1 = binder_tok.encode(f" [MEMO: Secret code for Project {s1} is {c1}.] ", add_special_tokens=False)
                    nd2 = binder_tok.encode(f" [MEMO: Secret code for Project {s2} is {c2}.] ", add_special_tokens=False)
                    qs, qc = (s1, c1) if r_i % 2 == 0 else (s2, c2)
                    qa = binder_tok.encode(
                        f"\nQuestion: What is the secret code for Project {qs}? Answer: {qc}.",
                        add_special_tokens=False,
                    )
                    row_seq = x[r_i].tolist() + [int(y[r_i, -1].item())]
                    p1 = rng_n.randint(16, max(17, T // 6))
                    p2 = rng_n.randint(max(18, T // 4), max(19, T // 2))
                    if p1 + len(nd1) < p2 and p2 + len(nd2) < len(row_seq) - len(qa) - 4:
                        row_seq[p1 : p1 + len(nd1)] = nd1
                        row_seq[p2 : p2 + len(nd2)] = nd2
                        row_seq[-len(qa) :] = qa
                        t_r = torch.tensor(row_seq[: T + 1], dtype=torch.long)
                        x[r_i], y[r_i] = t_r[:-1], t_r[1:]

            opt.zero_grad(set_to_none=True)
            loss_gpu = torch.zeros((), device=device, dtype=torch.float32)
            for lo in range(0, bs, micro_bs):
                hi = min(lo + micro_bs, bs)
                xm, ym = x[lo:hi].to(device, non_blocking=True), y[lo:hi].to(device, non_blocking=True)
                loss_gpu += _kd_microbatch_backward(
                    stu,
                    donor,
                    is_hf_donor,
                    w_lm_t,
                    xm,
                    ym,
                    (hi - lo) / bs,
                    kd,
                    use_ckpt,
                    ckpt_stride=eff_stride,
                )

            gn_tensor = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if torch.isfinite(gn_tensor):
                opt.step()
            step_done = step + 1
            last_step_done = step_done
            set_step_lr(step_done)

            if step_done in (start_step + 1, start_step + 5) or (step_done % 25 == 0 and step_done % eval_every != 0):
                loss_val, gn = float(loss_gpu.item()), float(gn_tensor.item())
                now = time.perf_counter()
                dt = (now - t_window) / max(1, step_done - step_window)
                t_window, step_window = now, step_done
                peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
                print(
                    f"  [{out_prefix}] step {step_done:5d}/{total_steps}  kd_ce {loss_val:.4f}  gnorm {gn:.2f}  "
                    f"toks {step_done*bs*T/1e6:.2f}M  {dt:.2f}s/it  VRAM {peak_gb:.2f} GiB",
                    flush=True,
                )

            if step_done % eval_every == 0 or step_done == total_steps:
                s_ppl, d_ppl, kl_v, top1_v = _eval_reasoning_metrics(stu, donor, is_hf_donor, eval_tokens, device, bs=2)
                stu.train()
                now = time.perf_counter()
                dt = (now - t_window) / max(1, step_done - step_window)
                t_window, step_window = now, step_done
                peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
                print(
                    f"  >>> [{out_prefix} EVAL] step {step_done:5d}/{total_steps}  KL {kl_v:.4f} nats  Top1 {top1_v:.2f}%  "
                    f"val_ppl {s_ppl:.2f}/{d_ppl:.2f} ({s_ppl/d_ppl:.3f}x donor)  {dt:.2f}s/it  VRAM {peak_gb:.2f} GiB",
                    flush=True,
                )

            if step_done % save_every == 0 or step_done == total_steps:
                _save_student(
                    stu,
                    stu_cfg,
                    ck,
                    out_ckpt,
                    step_done,
                    opt=opt if step_done < total_steps else None,
                    opt_ckpt=opt_ckpt,
                )
                if step_done >= total_steps and os.path.isfile(opt_ckpt):
                    os.remove(opt_ckpt)
                if hf_repo and (step_done % 250 == 0 or step_done == total_steps):
                    _upload_hf(out_ckpt, hf_repo, sync=(step_done == total_steps))
    except KeyboardInterrupt:
        print(f"\n  [{out_prefix}] Interrupted at step {last_step_done} — saving checkpoint...", flush=True)
        _save_student(stu, stu_cfg, ck, out_ckpt, last_step_done, opt=opt, opt_ckpt=opt_ckpt)
        if hf_repo:
            _upload_hf(out_ckpt, hf_repo, sync=True)
        raise

    return out_ckpt


def _onpolicy_prompt_batches(tok, prof, gen_bs: int, seed: int, max_prompt_tokens: int = 512) -> list[list[list[int]]]:
    from datasets import load_dataset

    from .benchmark import gsm8k_prompt
    from .profiles import format_user_prompt

    texts = []
    try:
        texts += [gsm8k_prompt(tok, r["question"]) for r in load_dataset("openai/gsm8k", "main", split="train")]
    except Exception as e:
        print(f"  [Stage-3] GSM8K train prompts skipped: {e}", flush=True)
    try:
        if prof.stage3_extra == "stratos":
            for row in load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train"):
                msgs = row.get("conversations") or row.get("messages") or []
                user = next(
                    (
                        str(m.get("value", m.get("content", "")))
                        for m in msgs
                        if str(m.get("from", m.get("role", ""))).lower() in ("user", "human")
                    ),
                    "",
                )
                if user.strip():
                    texts.append(format_user_prompt(tok, prof, user))
        elif prof.stage3_extra == "ultrachat":
            for i, row in enumerate(load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)):
                if str(row.get("prompt", "")).strip():
                    texts.append(format_user_prompt(tok, prof, row["prompt"]))
                if i >= 20_000:
                    break
    except Exception as e:
        print(f"  [Stage-3] {prof.stage3_extra} prompts skipped: {e}", flush=True)
    if not texts:
        raise RuntimeError("No Stage-3 prompts could be loaded")

    by_len: dict[int, list[list[int]]] = {}
    for ids in tok(texts, add_special_tokens=False)["input_ids"]:
        if len(ids) <= max_prompt_tokens:
            by_len.setdefault(len(ids), []).append(ids)
    rng = random.Random(seed)
    min_rows = max(1, gen_bs // 4)
    batches = []
    for n_tok in sorted(by_len):
        group = by_len[n_tok]
        rng.shuffle(group)
        batches += [
            group[i : i + gen_bs] for i in range(0, len(group), gen_bs) if len(group[i : i + gen_bs]) >= min_rows
        ]
    rng.shuffle(batches)
    print(f"  [Stage-3] {sum(len(b) for b in batches)} prompts in {len(batches)} equal-length batches", flush=True)
    return batches


def onpolicy_distill(
    ckpt_path: str,
    *,
    donor_id: str | None = None,
    work: str = ".",
    out_prefix: str = "bolt3_onpolicy",
    rounds: int = 10,
    prompts_per_round: int = 256,
    gen_bs: int = 16,
    max_new: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    lr: float = 5e-5,
    warmup: int = 50,
    seqs_per_step: int = 8,
    corpus_every: int = 4,
    eval_prompts: int = 8,
    seed: int = 4242,
    resume: bool = False,
    freeze_mlp: bool = True,
    train_proj: bool = False,
    stop_capped_frac: float = 0.5,
    chain_source: str = "student",
    profile: str | None = None,
):
    if chain_source not in ("student", "teacher"):
        raise ValueError(f"chain_source must be 'student' or 'teacher', got {chain_source!r}")

    from transformers import AutoTokenizer

    from .benchmark import _load_rows, gsm8k_prompt
    from .diagnose import bolt3_logits_fn, student_drift
    from .profiles import resolve_profile
    from .server import get_all_eos_ids

    _purge_zombie_vram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs(work, exist_ok=True)
    out_ckpt = os.path.join(work, f"{out_prefix}_latest.pth")
    opt_ckpt = os.path.join(work, f"{out_prefix}_optim.pth")
    resuming = resume and os.path.isfile(out_ckpt)

    stu, stu_cfg, ck_donor, _, ck = load_bolt3_checkpoint(out_ckpt if resuming else ckpt_path, device=None, eval_mode=False)
    start_round = int(ck.get("onpolicy_round", 0)) if resuming else 0
    opt_steps = int(ck.get("onpolicy_opt_steps", 0)) if resuming else 0
    cursor = int(ck.get("onpolicy_cursor", 0)) if resuming else 0
    donor_repo = donor_id or ck_donor
    if not donor_repo:
        raise ValueError("Pass --donor <repo_id> or use a checkpoint created by `python -m bolt3 convert`")

    groups, trainable = _setup_trainable(stu, stu_cfg, device, freeze_mlp=freeze_mlp, train_proj=train_proj)
    donor, is_hf_donor, w_lm_t, donor_dir = _load_tied_teacher(donor_repo, stu_cfg, stu, device)
    tok = AutoTokenizer.from_pretrained(donor_dir)
    prof = resolve_profile(tok, profile, donor_dir)
    max_new = prof.max_new if max_new is None else max_new
    temperature = prof.temperature if temperature is None else temperature
    top_p = prof.top_p if top_p is None else top_p
    eos_ids = get_all_eos_ids(tok)
    T = int(stu_cfg["max_T"])
    train_tokens, _ = _build_reasoning_and_eval_caches(work, donor_dir, donor_repo, 20_000_000, max_T=T, profile=prof.name)
    n_win = max(1, (len(train_tokens) - 1) // T)
    ar = torch.arange(T)
    corpus_gen = torch.Generator().manual_seed(seed + 1)
    for _ in range(opt_steps):
        torch.randint(0, n_win, (seqs_per_step,), generator=corpus_gen)

    batches = _onpolicy_prompt_batches(tok, prof, gen_bs, seed)
    eval_ps = [gsm8k_prompt(tok, r["question"]) for r in _load_rows("openai/gsm8k", "main", "test", 1000 + eval_prompts)[1000:]]
    s_fn = bolt3_logits_fn(stu)
    t_fn = bolt3_logits_fn(donor)

    opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=(device.type == "cuda"))
    if resuming and os.path.isfile(opt_ckpt):
        try:
            opt.load_state_dict(torch.load(opt_ckpt, map_location=device, weights_only=False))
            print("  [Resume] Restored optimizer moments from sidecar.")
        except Exception as e:
            print(f"  [Resume] Optimizer sidecar skipped: {e}")

    def set_lr(s: int):
        m = min(1.0, (s + 1) / max(1, warmup))
        for g in opt.param_groups:
            g["lr"] = lr * m * g.get("lr_scale", 1.0)

    print("=" * 84)
    print(
        f"BOLT-3 STAGE-3 CHAIN KD ({chain_source}-written chains, profile {prof.name}): {donor_repo} | "
        f"rounds {start_round}->{rounds} | {prompts_per_round} chains/round"
    )
    print("=" * 84)
    set_lr(opt_steps)

    t_dec = Bolt3Decoder(donor) if chain_source == "teacher" else None

    for rnd in range(start_round, rounds):
        t0 = time.perf_counter()
        stu.eval()
        dec = Bolt3Decoder(stu) if chain_source == "student" else None
        pool, lens = [], []
        while len(pool) < prompts_per_round:
            b = batches[cursor % len(batches)]
            cursor += 1
            torch.manual_seed(seed * 100_003 + cursor)
            x = torch.tensor(b, dtype=torch.long, device=device)
            active_dec = dec if dec is not None else t_dec
            toks, n = active_dec.generate_batch(x, max_new, temperature=temperature, top_p=top_p, eos_ids=eos_ids)
            toks, n = toks.cpu(), n.cpu()
            for r in range(len(b)):
                k = int(n[r])
                pool.append((torch.tensor(b[r] + toks[r, :k].tolist(), dtype=torch.long), len(b[r])))
                lens.append(k)
        del dec
        t_gen = time.perf_counter() - t0

        stu.train()
        train_pool = [c for c, k in zip(pool, lens) if k < max_new]
        if len(train_pool) < seqs_per_step:
            print(f"  [Stage-3] only {len(train_pool)} finished chains (< {seqs_per_step}); skipping training this round", flush=True)
            train_pool = []
        random.Random(seed + rnd).shuffle(train_pool)
        loss_sum, n_steps = 0.0, 0
        for s0 in range(0, len(train_pool), seqs_per_step):
            chunk = train_pool[s0 : s0 + seqs_per_step]
            n_corpus = max(1, len(chunk) // max(1, corpus_every - 1)) if corpus_every > 1 else 0
            w = 1.0 / (len(chunk) + n_corpus)
            opt.zero_grad(set_to_none=True)
            loss_gpu = torch.zeros((), device=device, dtype=torch.float32)
            for seq, P in chunk:
                xm = seq[:-1].view(1, -1).to(device)
                ym = seq[1:].view(1, -1).to(device)
                pos = torch.arange(xm.shape[1], device=device)
                mask = (pos >= P - 1).view(1, -1)
                loss_gpu += _kd_microbatch_backward(stu, donor, is_hf_donor, w_lm_t, xm, ym, w, 1.0, True, loss_mask=mask)
            for i0 in torch.randint(0, n_win, (seqs_per_step,), generator=corpus_gen)[:n_corpus].tolist():
                base = i0 * T + ar
                xm = train_tokens[base].long().view(1, -1).to(device)
                ym = train_tokens[base + 1].long().view(1, -1).to(device)
                loss_gpu += _kd_microbatch_backward(stu, donor, is_hf_donor, w_lm_t, xm, ym, w, 1.0, True)
            gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if torch.isfinite(gn):
                opt.step()
            opt_steps += 1
            set_lr(opt_steps)
            loss_sum += float(loss_gpu.item())
            n_steps += 1
        t_train = time.perf_counter() - t0 - t_gen

        drift = student_drift(stu, s_fn, t_fn, tok, device, eval_ps, max_new=min(max_new, 2048), temperature=temperature, top_p=top_p)
        capped = sum(int(k >= max_new) for k in lens)
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
        kl_line = "  ".join(f"{r['pos']}:{r['mean']:.3f}" for r in drift["kl"])
        print(
            f"  >>> [{out_prefix} R{rnd+1}/{rounds}] kd {loss_sum/max(1,n_steps):.4f} | gen median {int(statistics.median(lens))} tok, "
            f"capped {capped}/{len(lens)}, trained {len(train_pool)} | probe median {drift['median_len']}, capped {drift['capped']} | "
            f"gen {t_gen/60:.1f}m train {t_train/60:.1f}m | VRAM {peak_gb:.1f} GiB",
            flush=True,
        )
        print(f"      drift KL by pos: {kl_line}", flush=True)

        _save_student(
            stu,
            stu_cfg,
            ck,
            out_ckpt,
            int(ck.get("step", 0)) if isinstance(ck.get("step"), int) else 0,
            opt=opt,
            opt_ckpt=opt_ckpt,
            onpolicy_round=rnd + 1,
            onpolicy_opt_steps=opt_steps,
            onpolicy_cursor=cursor,
        )
        _save_student(
            stu,
            stu_cfg,
            ck,
            os.path.join(work, f"{out_prefix}_r{rnd+1}.pth"),
            int(ck.get("step", 0)) if isinstance(ck.get("step"), int) else 0,
            onpolicy_round=rnd + 1,
            onpolicy_opt_steps=opt_steps,
            onpolicy_cursor=cursor,
        )

        if drift["n"] and drift["capped_n"] / drift["n"] > stop_capped_frac:
            print(
                f"  [Stage-3] stopping: {drift['capped']} probe chains hit the cap (> {stop_capped_frac:.0%}).",
                flush=True,
            )
            return {"stopped_early": True}
    return {"stopped_early": False}


def scan_checkpoint(ckpt_path: str) -> dict:
    """Static CPU capacity & parameter scan for a Bolt-3 checkpoint."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    cfg = ck.get("config", {})
    num_layers = int(cfg.get("num_layers", 36))
    cold_set = {int(i) for i in (cfg.get("bolt3_cold_layers") or ())}
    prefix_attn = "attn" if any(f"blocks.0.attn." in k for k in sd) else "vsa"

    print("=" * 84)
    print(f"BOLT-3 PARAMETER SCAN: {ckpt_path} (step {ck.get('step', '?')}, cold_layers={sorted(cold_set)})")
    print("=" * 84)
    print(
        f"  {'L':>3} | {'gam_min':>7} {'gam_max':>7} | {'kappa_med':>9} | "
        f"{'beta_b':>7} {'|beta_w|':>8} | {'eta_b':>7} {'|eta_w|':>8} | {'res_rms':>7}"
    )
    print("  " + "-" * 80)

    out = {}
    for li in range(num_layers):
        lg_key = f"blocks.{li}.{prefix_attn}.log_gamma"
        alog_key = f"blocks.{li}.{prefix_attn}.A_log"
        if lg_key not in sd:
            if alog_key in sd:
                a_exp = sd[alog_key].float().exp()
                dtb = sd[f"blocks.{li}.{prefix_attn}.dt_bias"].float()
                print(
                    f"  L{li:<2d} | [gdn] decay_a={float(a_exp.mean()):.3f}..{float(a_exp.max()):.3f} | "
                    f"dt_bias={float(dtb.mean()):+.3f}"
                )
                out[str(li)] = {"type": "linear_attention", "a_exp_mean": float(a_exp.mean())}
            continue
        lg = sd[lg_key].float()
        gam = (0.999 * torch.sigmoid(lg)).clamp(1e-3, 0.999)
        kap = sd[f"blocks.{li}.{prefix_attn}.kappa"].float()
        bw = sd[f"blocks.{li}.{prefix_attn}.beta_proj.weight"].float()
        bb = sd[f"blocks.{li}.{prefix_attn}.beta_proj.bias"].float()
        ew = sd[f"blocks.{li}.{prefix_attn}.eta_proj.weight"].float()
        eb = sd[f"blocks.{li}.{prefix_attn}.eta_proj.bias"].float()
        rn = sd[f"blocks.{li}.{prefix_attn}.res_norm.weight"].float()
        rn_rms = float(rn.pow(2).mean().sqrt())
        cold_tag = " [cold]" if li in cold_set else ""
        print(
            f"  L{li:<2d} | {float(gam.min()):7.4f} {float(gam.max()):7.4f} | {float(kap.median()):9.4f} | "
            f"{float(bb.mean()):7.3f} {float(bw.norm()):8.4f} | {float(eb.mean()):7.3f} {float(ew.norm()):8.4f} | {rn_rms:7.4f}{cold_tag}"
        )
        out[str(li)] = dict(
            gamma_min=float(gam.min()),
            gamma_max=float(gam.max()),
            kappa_median=float(kap.median()),
            beta_bias_mean=float(bb.mean()),
            beta_weight_norm=float(bw.norm()),
            eta_bias_mean=float(eb.mean()),
            eta_weight_norm=float(ew.norm()),
            res_norm_rms=rn_rms,
        )
    return out
