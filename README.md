# Bolt-3 Attention (`bolt3`)

**Dual-Field Linear/Recurrent Attention** for converting pretrained HuggingFace Transformer LLMs (both dense Softmax models like `Qwen/Qwen3-4B` and hybrid models like `Qwen/Qwen3.5-4B`) into **bounded-state** models with **zero unbounded pure Softmax layers**, plus optional sparse **Cold-Armed** mid-deep layers (`Option-B`) for capped `O(1)` memory at long context without sacrificing multi-step reasoning or multi-needle retrieval.

- **Pretrained Checkpoint (`Qwen/Qwen3.5-4B` Hybrid Option-B `auto3`):** [`rikkathree/bolt-qwen3.5-4b`](https://huggingface.co/rikkathree/bolt-qwen3.5-4b) (`bolt3_b_qwen3_5_4b_latest.pth`)
- **Pretrained Checkpoint (`Qwen/Qwen3-4B` Full Bolt-3 Option-B `auto4`):** [`rikkathree/bolt-qwen3-4b`](https://huggingface.co/rikkathree/bolt-qwen3-4b) (`bolt3_b_qwen3_4b_latest.pth`)
- **Interactive Architecture Walkthrough:** [GitHub Pages](https://rikka3.github.io/bolt_attention/)

---

## Architecture: Bolt-3 Dual-Field Attention (+ Hybrid Donor Preservation)

Converted attention layers use **`Bolt3Attn`** (`bolt3/attention.py`), processing sequences in chunks of size `C` (default `128`) over **three fixed-size memory fields** per KV head so pure Bolt-3 state remains strictly `O(1)` at any context length:

1. **Attention Sinks + Local Window**
   - Keeps the first `n_sinks = 4` tokens plus the previous and current chunks (up to `2C = 256` tokens) with exact causal Softmax attention.
2. **Diversity Softmax Attractors (`S = 128` slots)**
   - As older chunks leave the local window, tokens are consolidated into `S` attractor slots storing raw keys (`ak`), values (`av`), running `mass` (adds `kappa * log(mass)` attention bias), mean RoPE position (`pm`), and `util`.
   - **Branchless Merge:** Tokens matching an existing slot (`cos_k >= 0.94` and `cos_v >= 0.88`) merge into it and update its `mass` and mean position.
   - **Diversity Eviction:** Unmerged tokens scored by value prediction error (`rel_err`) and key novelty (`1 - best_sim_k`) replace low-utility / redundant slots (`gamma_u = 0.985`).
3. **Gershgorin Eviction-to-Delta Residual Memory (`M_res`)**
   - Remaining unmerged/evicted tokens write into a fixed `[HKV, HD, HD]` `float32` associative state `M_res` using Gershgorin-normalized delta updates (`beta_proj`, `gamma_delta`) and gated readout (`eta_proj`, `res_norm` initialized at `0.05` for near-identity surgery).
4. **Optional Option-B Sparse Cold-Armed Layers (`--cold-layers`, `--max-cold-blocks`)**
   - Sparse mid-deep layers (`auto3` `[11, 19, 27]` on 32-layer Qwen3.5 or `auto4` `[14, 19, 24, 29]` on 36-layer Qwen3 with `max_cold_blocks = 256`) retain FIFO exact KV blocks up to `32,768` tokens before overflowing oldest blocks (via RoPE inversion) into attractors and `M_res`, capping live state flat past `32k`.
   - Single-token steady decode packs `[sink | ak_eff | half0 | half1 | cold]` into unified CUDA-Graph buffers (`_Bolt3GraphRunner`), updating both `Bolt3Attn` and native `GatedDeltaNet` (`conv_state`, `recurrent_state`) in-place across chunk boundaries and power-of-2 cold buckets.
5. **Hybrid Donor Preservation (`Qwen/Qwen3.5-4B`)**
   - Preserves native `GatedDeltaNet` (`linear_attention`) layers, zero-centered `RMSNorm` (`weight_offset=1.0`), `RMSNormGated`, partial RoPE (`partial_rotary_factor=0.25`), and `packed_q_gate` (`sigmoid(gate)` output gating) while replacing every remaining quadratic Softmax layer with `Bolt3Attn`.

---

## `Qwen/Qwen3.5-4B` Results (`bolt3_b_qwen3_5_4b_latest.pth`, `24 GDN + 5 Pure O(1) + 3 Cold [11,19,27]`)

Measured on RTX PRO 6000 under paired prompts, seeds, `eval_bs=64`, `top_k=0`, `repetition_penalty=1.0` *(HF Donor ran eager PyTorch GDN fallback without `causal_conv1d`/`flash-linear-attention`; Bolt-3 runs pure-PyTorch with CUDA Graphs)*:

| Metric | HF Donor (`Qwen/Qwen3.5-4B`) | **Bolt-3.5 (Option-B `auto3`)** | Notes |
|---|---:|---:|---|
| **MMLU** (`n=200`) | `67.50%` | **`70.00%`** | McNemar `p=0.442` |
| **ARC-Challenge** (`n=200`) | `51.50%` | **`53.50%`** | McNemar `p=0.219` |
| **HellaSwag** (`n=200`) | `62.00%` | **`60.50%`** | McNemar `p=0.453` |
| **GSM8K** (`n=200`, `cap=2048`) | `47.50%` | **`66.00%`** | `p<0.001` (Donor hit 2048-tok cap `104/200` vs Bolt-3 `67/200`) |
| **HumanEval** (`n=164`) | `79.88%` | **`78.66%`** | McNemar `p=0.774` |
| **MBPP** (`n=200`) | `56.00%` | **`54.00%`** | McNemar `p=0.344` |
| **IFEval** (`n=200`) | **`87.50%`** | **`87.50%`** | McNemar `p=1.000` |
| **4-Needle NIAH (`4k / 16k / 32k`)** | `4/4` / `4/4` / `4/4` | **`4/4` / `4/4` / `4/4` (`100%`)** | Exact match with Donor across all 3 lengths |
| **1-Needle Passkey (`4k–32k`)** | `100.0%` (`+51,769 MB` @32k) | **`100.0%`** (`+9,537 MB` @32k) | **`5.4x` lower peak batch VRAM** |
| **Decode Speed @64k** | `39.8 tok/s` | **`50.2 tok/s`** | **`1.26x` faster** (`1.87x` @ 2k: `75.1 vs 40.1 tok/s`) |
| **Live State RAM @64k** | `2,097.12 MB` | **`858.26 MB`** | **`-59.1%` RAM** (flat ceiling past `32k`) |
| **Cartridge Recall (`.vsa`)** | — | **`100.0% (4/4)`** | Avg route `18.28 ms`, erase residual `0.0` |

---

## `Qwen/Qwen3-4B` Results (`bolt3_b_qwen3_4b_latest.pth`, `32 Pure O(1) + 4 Cold [14,19,24,29]`)

Measured on RTX PRO 6000 under identical paired prompts, seeds, and sampling settings (`36/36` Softmax layers replaced):

| Metric | HF Donor (`Qwen/Qwen3-4B`) | **Bolt-3 (Option-B `auto4`)** | Notes |
|---|---:|---:|---|
| **MMLU** (`n=200`) | `69.50%` | **`68.50%`** | McNemar `p=0.790` |
| **ARC-Challenge** (`n=200`) | `54.00%` | **`56.00%`** | McNemar `p=0.455` |
| **HellaSwag** (`n=200`) | `52.00%` | **`55.50%`** | McNemar `p=0.118` |
| **GSM8K** (`n=200`, `cap=2048`) | `88.00%` | **`90.00%`** | McNemar `p=0.523` |
| **HumanEval** (`n=164`) | `80.49%` | **`76.22%`** | McNemar `p=0.230` |
| **MBPP** (`n=200`) | `55.50%` | **`53.50%`** | McNemar `p=0.481` |
| **IFEval** (`n=200`) | `86.50%` | **`87.50%`** | McNemar `p=0.774` |
| **Decode Speed @64k** | `29.6 tok/s` | **`44.0 tok/s`** | **`1.49x` faster** (`128`-tok chunks incl. consolidation) |
| **Live State RAM @64k** | `9,216.00 MB` | **`1,186.11 MB`** | **`-87.1%` RAM** (incl. CUDA-Graph buffers) |
| **1-Needle NIAH (`2k–16k`)** | — | **`100.0%`** | Non-repeating WikiText-103 (`4/4` per context) |
| **Cartridge Recall (`.vsa`)** | — | **`100.0% (4/4)`** | Avg route `37.44 ms`, erase residual `0.0` |

---

## CLI Quick Reference

```bash
# 1. Run Bolt-3 8-test parity, hybrid GatedDeltaNet, cold-armed, gradient, and erasure suite (CPU or CUDA)
python -m bolt3 test

# 2. Convert Qwen/Qwen3.5-4B (hybrid 24 GDN + 8 Bolt3Attn; --cold-layers auto3 picks [11,19,27])
python -m bolt3 convert --donor Qwen/Qwen3.5-4B --out checkpoints/bolt3_b_qwen3_5_4b_init.pth --cold-layers auto3 --no-freeze-proj

# 3. Stage-1 + Stage-2 Knowledge Distillation Healing
python -m bolt3 heal \
  --ckpt checkpoints/bolt3_b_qwen3_5_4b_init.pth \
  --donor Qwen/Qwen3.5-4B \
  --work checkpoints \
  --out-prefix bolt3_b_qwen3_5_4b \
  --stage1-tokens 4000000 --stage1-onpolicy \
  --tokens 8200000 --bs 8 --lr 6e-5 --kd 1.0 \
  --train-proj --highspeed --ckpt-stride 2 --resume

# 4. Scan / Diagnose / HubCheck / Benchmark / Serve
python -m bolt3 scan --ckpt checkpoints/bolt3_b_qwen3_5_4b_latest.pth
python -m bolt3 diagnose --ckpt checkpoints/bolt3_b_qwen3_5_4b_latest.pth --donor Qwen/Qwen3.5-4B --parity-only
python -m bolt3 hubcheck --ckpt checkpoints/bolt3_b_qwen3_5_4b_latest.pth --donor Qwen/Qwen3.5-4B
python -m bolt3 benchmark --ckpt checkpoints/bolt3_b_qwen3_5_4b_latest.pth --donor Qwen/Qwen3.5-4B --limit 200 --gsm8k-limit 200
python -m bolt3 serve --ckpt checkpoints/bolt3_b_qwen3_5_4b_latest.pth --donor Qwen/Qwen3.5-4B --port 8000
```

---

## Repository Layout

- `bolt3/attention.py` — `Bolt3Attn`, `Bolt3Decoder`, `_Bolt3GraphRunner` (unified CUDA-Graph steady decode across `Bolt3Attn` + `GatedDeltaNet` + cold-armed buckets), and `memorize`/`forget`.
- `bolt3/nn.py` — Shared `RMSNorm` (`weight_offset`), `RMSNormGated`, `GatedDeltaNet`, `RoPE` (`partial_rotary_factor`), `SwiGLUMLP`, `SoftmaxAttn` (teacher replica), and `repeat_kv`.
- `bolt3/model.py` — `Bolt3Block`, `Bolt3LM`, `FP32_KEYS`, and `load_bolt3_checkpoint` (with legacy `.vsa.` key remap).
- `bolt3/inspector.py` — HuggingFace snapshot download, safetensor inspection (incl. `linear_attn.*`, `mtp.*` skip, `packed_q_gate`), and weight mapping.
- `bolt3/profiles.py` — Donor sampling & data-mix profile detection (`reasoning`, `chat`, `base`).
- `bolt3/converter.py` — `convert_donor_model`, Stage-1 attention transfer, Stage-2 KD healing, Stage-3 `onpolicy_distill`, and `scan_checkpoint`.
- `bolt3/memory_bank.py` — Disk-backed `.vsa` neural state cartridges (`Bolt3MemoryBank`).
- `bolt3/diagnose.py`, `hubcheck.py`, `benchmark.py`, `server.py`, `tests.py`, `cli.py` — Diagnostics, 7-task + scaling evaluation, OpenAI-compatible serving, and 8-test CPU/CUDA unit suite.
