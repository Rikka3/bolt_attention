# Bolt-3 Attention (`bolt3`)

**Dual-Field Linear/Recurrent Attention** for converting pretrained HuggingFace Transformer LLMs (such as `Qwen/Qwen3-4B`) into **100% Full Bolt-3** models with **zero pure Softmax layers**, plus optional sparse **Cold-Armed** mid-deep layers (`Option-B`) for capped `O(1)` memory at long context without sacrificing multi-step reasoning.

- **Pretrained Checkpoint (Option-B `Qwen/Qwen3-4B`):** [`rikkathree/bolt-qwen3-4b`](https://huggingface.co/rikkathree/bolt-qwen3-4b) (`bolt3_b_qwen3_4b_latest.pth`)

---

## Architecture: Bolt-3 Dual-Field Attention

Every layer uses **`Bolt3Attn`** (`bolt3/attention.py`), processing sequences in chunks of size `C` (default `128`) over **three fixed-size memory fields** per KV head so pure Bolt-3 state remains strictly `O(1)` at any context length:

1. **Attention Sinks + Local Window**
   - Keeps the first `n_sinks = 4` tokens plus the previous and current chunks (up to `2C = 256` tokens) with exact causal Softmax attention.
2. **Diversity Softmax Attractors (`S = 128` slots)**
   - As older chunks leave the local window, tokens are consolidated into `S` attractor slots storing raw keys (`ak`), values (`av`), running `mass` (adds `kappa * log(mass)` attention bias), mean RoPE position (`pm`), and `util`.
   - **Branchless Merge:** Tokens matching an existing slot (`cos_k >= 0.94` and `cos_v >= 0.88`) merge into it and update its `mass` and mean position.
   - **Diversity Eviction:** Unmerged tokens scored by value prediction error (`rel_err`) and key novelty (`1 - best_sim_k`) replace low-utility / redundant slots (`gamma_u = 0.985`).
3. **Gershgorin Eviction-to-Delta Residual Memory (`M_res`)**
   - Remaining unmerged/evicted tokens write into a fixed `[HKV, HD, HD]` `float32` associative state `M_res` using Gershgorin-normalized delta updates (`beta_proj`, `gamma_delta`) and gated readout (`eta_proj`, `res_norm` initialized at `0.05` for near-identity surgery).
4. **Optional Option-B Sparse Cold-Armed Layers (`--cold-layers`, `--max-cold-blocks`)**
   - Sparse mid-deep layers (e.g. `[14, 19, 24, 29]` with `max_cold_blocks = 256`) retain FIFO exact KV blocks up to `32,768` tokens before overflowing oldest blocks (via RoPE inversion) into attractors and `M_res`, capping memory growth asymptotically (`O(1)` past `32k`).
   - Single-token steady decode packs `[sink | ak_eff | half0 | half1 | cold]` into unified CUDA-Graph buffers (`_Bolt3GraphRunner`), updating state in-place across chunk boundaries and power-of-2 cold buckets.

---

## Qwen/Qwen3-4B Results (`bolt3_b_qwen3_4b_latest.pth`, Option-B)

Measured on RTX PRO 6000 under paired prompts, seeds, and sampling settings:

| Metric | HF Donor (`Qwen/Qwen3-4B`) | **Bolt-3 (Option-B)** | Notes |
|---|---:|---:|---|
| **MMLU** (`n=200`) | `69.50%` | **`68.50%`** | McNemar `p=0.790` |
| **ARC-Challenge** (`n=200`) | `54.00%` | **`56.00%`** | McNemar `p=0.455` |
| **HellaSwag** (`n=200`) | `52.00%` | **`55.50%`** | McNemar `p=0.118` |
| **GSM8K** (`n=200`, `cap=2048`) | `88.00%` | **`90.00%`** | McNemar `p=0.523` |
| **HumanEval** (`n=164`) | `80.49%` | **`76.22%`** | McNemar `p=0.230` |
| **MBPP** (`n=200`) | `55.50%` | **`53.50%`** | McNemar `p=0.481` |
| **IFEval** (`n=200`) | `86.50%` | **`87.50%`** | McNemar `p=0.774` |
| **Decode Speed @64k** | `29.6 tok/s` | **`44.0 tok/s`** | **`1.49x` faster** (`128`-tok chunks incl. consolidation) |
| **Live State RAM @64k** | `9216.00 MB` | **`1186.11 MB`** | **`-87.1%` RAM** (incl. CUDA-Graph buffers) |
| **1-Needle NIAH (`2k–16k`)** | — | **`100.0%`** | Non-repeating WikiText-103 (`4/4` per context) |
| **Cartridge Recall (`.vsa`)** | — | **`100.0% (4/4)`** | Avg route `37.44 ms`, erase residual `0.0` |

---

## CLI Quick Reference

```bash
# 1. Run Bolt-3 7-test parity, cold-armed, gradient, and erasure suite (CPU or CUDA)
python -m bolt3 test

# 2. Convert Qwen/Qwen3-4B (pure O(1), or add --cold-layers auto4 for Option-B)
python -m bolt3 convert --donor Qwen/Qwen3-4B --out checkpoints/bolt3_qwen3_4b_init.pth [--cold-layers auto4]

# 3. Heal via Knowledge Distillation
python -m bolt3 heal \
  --ckpt checkpoints/bolt3_qwen3_4b_init.pth \
  --donor Qwen/Qwen3-4B \
  --work checkpoints \
  --out-prefix bolt3_qwen3_4b \
  --tokens 16000000 --bs 8 --lr 8e-5 --kd 1.0 \
  --freeze-mlp --highspeed --resume

# 4. Stage-3 Chain Distillation
python -m bolt3 onpolicy \
  --ckpt checkpoints/bolt3_qwen3_4b_latest.pth \
  --donor Qwen/Qwen3-4B \
  --work checkpoints --out-prefix bolt3_qwen3_4b_onpolicy --resume

# 5. Scan / Diagnose / HubCheck / Benchmark / Serve
python -m bolt3 scan --ckpt checkpoints/bolt3_qwen3_4b_latest.pth
python -m bolt3 diagnose --ckpt checkpoints/bolt3_qwen3_4b_latest.pth --donor Qwen/Qwen3-4B --parity-only
python -m bolt3 hubcheck --ckpt checkpoints/bolt3_qwen3_4b_latest.pth --donor Qwen/Qwen3-4B
python -m bolt3 benchmark --ckpt checkpoints/bolt3_qwen3_4b_latest.pth --donor Qwen/Qwen3-4B --limit 200 --gsm8k-limit 200
python -m bolt3 serve --ckpt checkpoints/bolt3_qwen3_4b_latest.pth --donor Qwen/Qwen3-4B --port 8000
```

---

## Repository Layout

- `bolt3/attention.py` — `Bolt3Attn`, `Bolt3Decoder`, `_Bolt3GraphRunner` (unified CUDA-Graph steady decode + cold-armed buckets), and `memorize`/`forget`.
- `bolt3/nn.py` — Shared `RMSNorm`, `RoPE`, `SwiGLUMLP`, `SoftmaxAttn` (donor teacher replica), and `repeat_kv`.
- `bolt3/model.py` — `Bolt3Block`, `Bolt3LM`, `FP32_KEYS`, and `load_bolt3_checkpoint` (with legacy `.vsa.` key remap).
- `bolt3/inspector.py` — HuggingFace snapshot download, safetensor inspection, and weight mapping.
- `bolt3/profiles.py` — Donor sampling & data-mix profile detection (`reasoning`, `chat`, `base`).
- `bolt3/converter.py` — `convert_donor_model`, Stage-1 attention transfer, Stage-2 KD healing, Stage-3 `onpolicy_distill`, and `scan_checkpoint`.
- `bolt3/memory_bank.py` — Disk-backed `.vsa` neural state cartridges (`Bolt3MemoryBank`).
- `bolt3/diagnose.py`, `hubcheck.py`, `benchmark.py`, `server.py`, `tests.py`, `cli.py` — Diagnostics, 7-task + scaling evaluation, OpenAI-compatible serving, and 7-test CPU/CUDA unit suite.
