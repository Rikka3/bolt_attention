# AGENTS.md

Guidance for coding agents working in the **Bolt-3 Attention** (`bolt3`) repository.

## Overview

This repository implements **Bolt-3 Dual-Field Attention** (`bolt3`): a pure-PyTorch, constant-memory (`O(1)` inference state) attention architecture for retrofitting pretrained HuggingFace Transformer LLMs (such as `Qwen/Qwen3-4B`, Llama, Mistral, Qwen2/3) into **100% Full Bolt-3** models with **zero pure Softmax layers**.

There is no `pyproject.toml`, `setup.py`, `requirements.txt`, or CI pipeline. Commands run directly from the repository root via `python -m bolt3 <subcommand>`. Local Molab (RTX 6000 Pro) cell scripts live in `cells/` (gitignored; never commit or push `cells/` to GitHub).

### Runtime Dependencies (Observed from Imports)
- **Core:** `torch` (CUDA recommended for training/healing; CPU works for tests, `scan`, and quantized `serve`), `transformers`, `safetensors`, `huggingface_hub`, `datasets`, `pyarrow`, `tqdm`.
- **Optional / Evaluation:** `inspect_ai` (optional official logging in `bolt3/benchmark.py`), `matplotlib` & `numpy` (benchmark figure generation in `bolt3/benchmark.py`).
- **Environment variables:** `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` for gated donor downloads and `--hf-repo` uploads; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` recommended on Molab.

---

## Architecture

### Core Model & Attention (`bolt3/nn.py`, `bolt3/attention.py`, `bolt3/model.py`)

- **`Bolt3LM` / `Bolt3Block`** (`bolt3/model.py`):
  - Uses HuggingFace-style parameter names (`embed_tokens`, `blocks`, `norm`, `lm_head`, `norm1`, `norm2`, `attn`, `mlp` with `gate_proj`/`up_proj`/`down_proj`).
  - Default student layout sets `layer_types = ["bolt3"] * num_layers` (all layers are `Bolt3Attn`, zero pure Softmax layers).
  - Teacher replicas used during distillation construct `Bolt3LM(..., layer_types=["softmax"] * num_layers)` with `SoftmaxAttn` (`bolt3/nn.py`) so teacher and student share `embed_tokens` and `lm_head` weights without duplicate VRAM overhead.
  - `load_bolt3_checkpoint` automatically remaps legacy `blocks.{li}.vsa.*` checkpoints to `blocks.{li}.attn.*`, converts `"vsa"` layer types to `"bolt3"`, and casts `FP32_KEYS` parameters to `float32`.

- **`Bolt3Attn` & `Bolt3Decoder`** (`bolt3/attention.py`):
  - Processes sequences in chunks of `C=128` across **three fixed-size memory fields** per KV head:
    1. **Attention Sinks + Local Window:** First `n_sinks=4` tokens plus previous and current chunks (up to `2C=256` tokens) with causal Softmax attention.
    2. **Diversity Softmax Attractors (`S=128`):** Stores raw keys (`ak`), values (`av`), running `mass` (contributing `kappa * log(mass)` bias), mean RoPE position (`pm`), and `util`. Branchless merge at `sim_k=0.94`, `sim_v=0.88`; diversity eviction with `gamma_u=0.985`.
    3. **Gershgorin Eviction-to-Delta Residual Memory (`M_res`):** Fixed `[HKV, HD, HD]` `float32` state updated by unmerged/evicted tokens via `beta_proj`, Gershgorin step normalization, and `gamma_delta()` (`0.999 * sigmoid(log_gamma)`), gated into output via `eta_proj` and `res_norm` (initialized at `0.05`).
  - `Bolt3Decoder` keeps chunked `forward()` and recurrent `step()` mathematically aligned: `step()` splits arbitrary multi-token inputs on `C - cur_len` boundaries and accumulates `util_acc` across tokens until chunk completion (`cur_kp.shape[2] >= C`) before updating `util`. It also caches RoPE-rotated `ak_eff`/`att_bias` between consolidations, reuses preallocated `C`-slot chunk buffers, and supports left-padded batching via `attn_mask` in `prefill()`/`generate_batch()`.
  - Supports algebraic `memorize` and `forget` over `M_res` states and `StaticKV` if stepping a Softmax replica.

### Conversion, Distillation & Molab Pipeline

- **`bolt3/inspector.py`**: `fetch_any_donor`, `load_all_safetensors` (strips visual/multimodal prefixes), `inspect_donor_config`, and `map_hf_to_bolt3` (handles `qk_norm`, `qkv_bias`, `packed_q_gate`, `tie_word_embeddings`, and `head_dim != dim // num_heads` such as `Qwen/Qwen3-4B`).
- **`bolt3/profiles.py`**: `Profile` registry (`reasoning`, `chat`, `base`) auto-detected from tokenizer `chat_template` and overridden by `generation_config.json`.
- **`bolt3/converter.py`**:
  - `convert_donor_model`: initializes 100% `Bolt3Attn` student from HF weights and writes atomic `.pth` checkpoints.
  - `heal_bolt3_model`: optional Stage-1 (`_stage1_attention_transfer`, teacher-forced or `--stage1-onpolicy`) + Stage-2 (`_kd_microbatch_backward` using token-sliced `float32` KL+CE, intermediate/final hidden-state cosine alignment at `kd_anchor_every=4`, and synthetic `[MEMO: ...]` sequences every 4th step). Auto-tunes activation checkpointing for 48 GB (`ckpt_stride=2`) and 96 GB (`use_ckpt=False`) RTX 6000 Pro GPUs when `--highspeed --freeze-mlp` is set.
  - `onpolicy_distill`: Stage-3 student- or teacher-written chain distillation (`Bolt3Decoder.generate_batch`) filtered to non-capped chains, mixed with corpus windows (`--corpus-every`).
  - `scan_checkpoint`: CPU inspection of per-layer `gamma_delta`, `kappa`, `beta_proj`, `eta_proj`, and `res_norm`.
- **`bolt3/memory_bank.py`**: `Bolt3MemoryBank` stores/routes disk-backed `.vsa` cartridges using embedding cosine similarity and verifies `Bolt3Decoder.forget` residuals.

### Module & File Map

| Path | Role |
|---|---|
| `bolt3/__init__.py`, `__main__.py` | Public API exports and `python -m bolt3` entrypoint |
| `bolt3/cli.py` | CLI parser & subcommand router (`convert`, `heal`, `onpolicy`, `diagnose`, `hubcheck`, `benchmark`, `scan`, `serve`, `test`) |
| `bolt3/nn.py` | `RMSNorm`, `RoPE`, `apply_rope_pos`, `SwiGLUMLP`, `SoftmaxAttn` (teacher replica), `repeat_kv` |
| `bolt3/attention.py` | `Bolt3Attn`, `Bolt3Decoder`, `StaticKV` |
| `bolt3/model.py` | `Bolt3Block`, `Bolt3LM`, `FP32_KEYS`, `load_bolt3_checkpoint` |
| `bolt3/inspector.py` | HF donor download, safetensor mapping, and config inspection |
| `bolt3/profiles.py` | Donor sampling & healing mix `Profile` detection |
| `bolt3/converter.py` | `convert_donor_model`, data cache builder, Stage-1/2/3 distillation, `scan_checkpoint` |
| `bolt3/memory_bank.py` | Disk-backed `.vsa` neural state cartridges (`Bolt3MemoryBank`) |
| `bolt3/server.py` | OpenAI-compatible `/v1/models` & `/v1/chat/completions` HTTP server + CPU INT8 quantization |
| `bolt3/diagnose.py` | Chunked-vs-token decode parity, per-layer relative error, and KL drift-by-position |
| `bolt3/hubcheck.py` | End-to-end release suite (`load`, `chat`, `decode_parity`, `batch`, `passkey`, `long_ppl`, `memory_bank`, `server`) |
| `bolt3/benchmark.py` | 7-task evaluation suite + 2k–64k throughput/VRAM scaling + cartridge recall + 4-panel chart |
| `bolt3/tests.py` | CPU/CUDA unit test suite (`run_all`) |
| `cells/*.py` (local-only) | Gitignored Molab (RTX 6000 Pro) Qwen3-4B scripts — never commit or push to GitHub |

---

## Build, Test, and CLI Commands

Run all commands from the repository root.

```bash
# Unit & parity test suite (works on CPU or CUDA)
python -m bolt3 test

# Convert Qwen/Qwen3-4B to 100% Full Bolt-3 (no pure softmax)
python -m bolt3 convert --donor Qwen/Qwen3-4B --out checkpoints/bolt3_qwen3_4b_init.pth

# Heal converted checkpoint (48-96 GB RTX 6000 Pro fast mode)
python -m bolt3 heal --ckpt checkpoints/bolt3_qwen3_4b_init.pth --donor Qwen/Qwen3-4B \
  --work checkpoints --out-prefix bolt3_qwen3_4b --tokens 16000000 --bs 8 --lr 8e-5 --kd 1.0 \
  --freeze-mlp --highspeed --resume

# Stage-3 chain distillation
python -m bolt3 onpolicy --ckpt checkpoints/bolt3_qwen3_4b_latest.pth --donor Qwen/Qwen3-4B \
  --work checkpoints --out-prefix bolt3_qwen3_4b_onpolicy --chain-source student --rounds 6 --resume

# Static scan, decode parity, hubcheck, benchmark, and OpenAI server
python -m bolt3 scan --ckpt checkpoints/bolt3_qwen3_4b_latest.pth
python -m bolt3 diagnose --ckpt checkpoints/bolt3_qwen3_4b_latest.pth --donor Qwen/Qwen3-4B --parity-only
python -m bolt3 hubcheck --ckpt checkpoints/bolt3_qwen3_4b_latest.pth --donor Qwen/Qwen3-4B
python -m bolt3 benchmark --ckpt checkpoints/bolt3_qwen3_4b_latest.pth --donor Qwen/Qwen3-4B --limit 200 --gsm8k-limit 200 [--batch-size 32] [--tasks MMLU,GSM8K]
python -m bolt3 serve --ckpt checkpoints/bolt3_qwen3_4b_latest.pth --donor Qwen/Qwen3-4B --port 8000
```

---

## Coding Conventions & Critical Invariants

1. **Precision Boundaries (`FP32_KEYS`):**
   - Keep backbone weights (`embed_tokens`, `q/k/v/o_proj`, `mlp`, norms) in `bfloat16` on CUDA, while keeping recurrent state (`M_res`), attractor metadata (`mass`, `util`, `pm`), softmax score reductions, KL/CE head slices, and `FP32_KEYS` (`log_gamma`, `kappa`, `beta_proj`, `eta_proj`, `res_norm`) in `float32`.
   - Enable TF32 for training/healing/hubcheck; disable TF32 (`allow_tf32 = False`) in strict parity tests (`bolt3/tests.py`).

2. **Forward / Decoder Parity & Branchless Consolidation:**
   - Any edit to `Bolt3Attn.forward` or `Bolt3Attn._consolidate_functional` requires matching updates in `Bolt3Decoder.step_block`. Verify with `python -m bolt3 test` and `python -m bolt3 diagnose --parity-only`.
   - Keep `Bolt3Attn._consolidate_functional` free of `.item()` / data-dependent Python branches to avoid CPU-GPU syncs, and preserve the `0.0 * (...)` dummy gradient term in `Bolt3Attn.forward` when `m_res is None` on the final chunk so short sequences (`T <= 2*C`) never drop parameter `.grad`s.
   - Cross-chunk consolidation detaches `ak`, `av`, `pm`, and `m_res` (`TBPTT`) across chunk boundaries to bound training VRAM at `O(1)` depth per chunk while preserving single-chunk gradient flow to `beta_proj` and `log_gamma`.

3. **Memory Hygiene, Atomic Saves & Cells:**
   - Lazy-import heavy packages (`transformers`, `datasets`, `huggingface_hub`, `safetensors`, `matplotlib`, `pyarrow`, `inspect_ai`) inside functions.
   - Chunk `[B*T, V]` logit/KL computations in `_kd_microbatch_backward` and `_nll_from_hidden`, and call `_purge_zombie_vram()` when swapping teacher/student models.
   - Write checkpoints and JSON reports atomically via `<path>.tmp` + `os.replace`.
   - Keep local `cells/` scripts pure Python (resolving `REPO_ROOT` without requiring `__file__` so they work both via `python cells/...py` and pasted directly into Molab cells) and tag the newest cell with ` (latest)` at the end of its basename. **Never commit or upload `cells/` to GitHub (`cells/` is gitignored)** — simply tell the user which cell to run.
