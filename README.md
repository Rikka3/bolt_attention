# Bolt-3 Attention (`bolt3`)

**Dual-Field Linear/Recurrent Attention** for converting pretrained HuggingFace Transformer LLMs (both dense Softmax models like `Qwen/Qwen3-4B` and hybrid models like `Qwen/Qwen3.5-4B`) into **bounded-state** models with **zero unbounded pure Softmax layers**, plus optional sparse **Cold-Armed** mid-deep layers (`Option-B`) for capped `O(1)` memory at long context without sacrificing multi-step reasoning or multi-needle retrieval.

- **Pretrained Checkpoint (`Qwen/Qwen3.5-4B` Hybrid Option-B `auto3`):** [`rikkathree/bolt-qwen3.5-4b`](https://huggingface.co/rikkathree/bolt-qwen3.5-4b) (`bolt3.1-qwen3.5-4b.pth`)
- **Pretrained Checkpoint (`Qwen/Qwen3-4B` Full Bolt-3 Option-B `auto4`):** [`rikkathree/bolt-qwen3-4b`](https://huggingface.co/rikkathree/bolt-qwen3-4b) (`bolt3_b_qwen3_4b_latest.pth`)
- **Interactive Architecture Walkthrough:** [GitHub Pages](https://rikka3.github.io/bolt_attention/)

---

## Architecture: Bolt-3.1 Dual-Field Attention (+ Hybrid Donor Preservation)

Converted attention layers use **`Bolt3Attn`** (`bolt3/attention.py`), processing sequences in chunks of size `C` (default `128`) over **three fixed-size memory fields** per KV head so pure Bolt-3 state remains strictly `O(1)` at any context length:

1. **Attention Sinks + `4C` Unified Local Ring (`loc_chunks = 4`)**
   - Keeps the first `n_sinks = 4` tokens plus a circular `4C = 512` local token ring with exact causal Softmax attention.
2. **Tri-Partitioned Diversity Softmax Attractors (`S = 128` slots)**
   - Older chunks leaving the local ring consolidate cleanly into `S = S_hh(48) + S_land(48) + S_proto(32)` un-smeared attractor slots storing raw keys (`ak`), values (`av`), running `mass` (adds `kappa * log(mass)` attention bias), mean RoPE position (`pm`), and `util`.
   - **Branchless Merge & Multi-Band Eviction:** Matching keys (`cos_k >= 0.94`, `cos_v >= 0.88`) update slots cleanly; remaining high-surprise tokens target partition-appropriate slots (`gamma_u = 0.985`), cutting `16k`/`64k` WikiText-103 KL by `31–36%` over chunked local windows alone.
3. **Gershgorin `M_res` + Evicted-Mass Supervision (`L_mem`)**
   - Unmerged/evicted tokens write into a fixed `[HKV, HD, HD]` `float32` associative state `M_res` via Gershgorin-normalized delta updates (`beta_proj`, `gamma_delta`) and soft-clipped readout (`eta_proj`, `res_norm`).
   - Step-2 healing adds `--evict-loss` (`L_mem` on post-local tokens whose teacher attention mass falls outside the local window) with frozen `Q/K/V/O` projections (`--freeze-proj`), training `beta`/`eta` across all Bolt-3 layers.
4. **Optional Option-B Sparse Cold-Armed Layers (`--cold-layers`, `--max-cold-blocks`)**
   - Sparse mid-deep layers (`auto3` `[11, 19, 27]` on 32-layer Qwen3.5 or `auto4` `[14, 19, 24, 29]` on 36-layer Qwen3 with `max_cold_blocks = 256`) retain FIFO exact KV blocks up to `32,768` tokens (`cold_hit` deduplicates already-cold tokens from attractors) before overflowing oldest blocks via RoPE inversion into attractors and `M_res`, locking live state flat past `32k`.
   - Single-token steady decode packs `[sink | ak_eff | loc_ring | cold]` into unified CUDA-Graph buffers (`_Bolt3GraphRunner`), updating both `Bolt3Attn` and native `GatedDeltaNet` (`conv_state`, `recurrent_state`) in-place across chunk boundaries and power-of-2 cold buckets.
5. **Hybrid Donor Preservation (`Qwen/Qwen3.5-4B`)**
   - Preserves native `GatedDeltaNet` (`linear_attention`) layers, zero-centered `RMSNorm` (`weight_offset=1.0`), `RMSNormGated`, partial RoPE (`partial_rotary_factor=0.25`), and `packed_q_gate` (`sigmoid(gate)` output gating) while replacing every remaining quadratic Softmax layer with `Bolt3Attn`.

---

## `Qwen/Qwen3.5-4B` Results (`bolt3.1-qwen3.5-4b.pth`, `24 GDN + 5 Pure O(1) + 3 Cold [11,19,27]`)

Measured on RTX PRO 6000 under paired prompts, seeds, `eval_bs=64`, `top_k=0`, `repetition_penalty=1.0` *(HF Donor ran eager PyTorch GDN fallback without `causal_conv1d`/`flash-linear-attention`; Bolt-3.1 runs pure-PyTorch with CUDA Graphs)*:

| Metric | HF Donor (`Qwen/Qwen3.5-4B`) | **Bolt-3.1 (Option-B `auto3`)** | Notes |
|---|---:|---:|---|
| **MMLU** (`n=200`) | `67.50%` (`135/200`) | **`67.00%`** (`134/200`) | McNemar `p=1.000` (`0/1`) |
| **ARC-Challenge** (`n=200`) | `51.50%` (`103/200`) | **`51.50%`** (`103/200`) | McNemar `p=1.000` (`0/0` exact tie) |
| **HellaSwag** (`n=200`) | `62.00%` (`124/200`) | **`61.50%`** (`123/200`) | McNemar `p=1.000` (`0/1`) |
| **GSM8K** (`n=200`, `cap=2048`) | `49.50%` (`99/200`, `trunc=100`) | **`46.50%`** (`93/200`, `trunc=107`) | McNemar `p=0.461` (`med_len=2048` cap) |
| **HumanEval** (`n=164`) | `80.49%` (`132/164`) | **`80.49%`** (`132/164`) | McNemar `p=1.000` (`2/2` exact tie) |
| **MBPP** (`n=200`) | `57.00%` (`114/200`) | **`55.00%`** (`110/200`) | McNemar `p=0.125` (`0/4`) |
| **IFEval** (`n=200`) | `87.50%` (`175/200`) | **`88.00%`** (`176/200`) | McNemar `p=1.000` (`3/2`) |
| **TruthfulQA `mc1 / mc2`** (`n=817`) | `72.22% / 34.39%` | **`71.73% / 34.15%`** | UK AISI Inspect (`p=0.125 / 0.791`, `0` fmt err) |
| **Paired 4-Needle NIAH (`4k/16k/32k/64k`)** | `4/4 / 4/4 / 4/4 / 4/4` | **`4/4 / 4/4 / 4/4 / 2/4`** | `100%` through `32k`; multi-seed `64k_32k`: **`75.0%` (`18/24`)**, `128k_32k`: **`62.5%` (`15/24`)** |
| **1-Needle Passkey (`4k / 16k / 32k`)** | `100% / 100% / 100%` | **`83.3% / 100% / 100%`** | `5.4x` lower peak batch VRAM (`+9,510` vs `+51,768 MB` @32k) |
| **Decode Speed @64k** | `38.9 tok/s` | **`49.3 tok/s`** | **`1.27x` faster** (`1.92x` @ 2k: `73.2 vs 38.2 tok/s`) |
| **Live State RAM @64k** | `2,097.12 MB` | **`873.38 MB`** | **`-58.4%` RAM** (flat ceiling past `32k`) |
| **Cartridge Recall (`.vsa`)** | — | **`100.0% (4/4)`** | Avg route `33.64 ms`, erase residual `0.0` (`hubcheck`: `3/3`) |

> **Equal-Byte SWA & Ablation Controls (WikiText-103 KL `[mean, tail]`):** `OptB_Full` scores **`0.0660`** (`16k`) / **`0.1272`** (`64k`), beating `OptB_TrueNoAtt` (`0.1026` / `0.1853`) and `SWA_W640+Cold` (`0.0798` / `0.1464`), while slightly trailing full-byte `SWA_W898+Cold` (`0.0624` / `0.1137`) when `M_res`'s FP32 bytes are given to raw recent SWA tokens. Non-repetitive WikiText-haystack evicted NIAH (`16k_wt_4k`) remains `0/8` outside the cold cap (`4/4` inside).

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
# 1. Run Bolt-3 parity, hybrid GatedDeltaNet, cold-armed, gradient, and erasure tests (CPU or CUDA)
python -m bolt3 test

# 2. Convert Qwen/Qwen3.5-4B (hybrid 24 GDN + 8 Bolt3Attn; --cold-layers auto3 picks [11,19,27])
python -m bolt3 convert --donor Qwen/Qwen3.5-4B --out checkpoints/bolt3.1-qwen3.5-4b-init.pth --cold-layers auto3

# 3. Stage-1 + Stage-2 Knowledge Distillation with Evicted-Mass Supervision (L_mem)
python -m bolt3 heal \
  --ckpt checkpoints/bolt3.1-qwen3.5-4b-init.pth \
  --donor Qwen/Qwen3.5-4B \
  --work checkpoints \
  --out-prefix bolt3.1-qwen3.5-4b \
  --tokens 8200000 --bs 8 --lr 6e-5 --warmup 100 --kd 1.0 \
  --freeze-mlp --freeze-proj --highspeed --evict-loss 0.5 --mres-tbptt 0 --resume

# 4. Scan / Diagnose / HubCheck / Benchmark / Serve
python -m bolt3 scan --ckpt checkpoints/bolt3.1-qwen3.5-4b.pth
python -m bolt3 diagnose --ckpt checkpoints/bolt3.1-qwen3.5-4b.pth --donor Qwen/Qwen3.5-4B --parity-only
python -m bolt3 hubcheck --ckpt checkpoints/bolt3.1-qwen3.5-4b.pth --donor Qwen/Qwen3.5-4B
python -m bolt3 benchmark --ckpt checkpoints/bolt3.1-qwen3.5-4b.pth --donor Qwen/Qwen3.5-4B --limit 200 --gsm8k-limit 200
python -m bolt3 serve --ckpt checkpoints/bolt3.1-qwen3.5-4b.pth --donor Qwen/Qwen3.5-4B --port 8000
```

---

## Builds On

Bolt-3 / Bolt-3.1 composes established streaming, KV-merging, delta-rule, and pretrained attention-retrofit mechanisms:
- **Streaming & Attention Sinks:** StreamingLLM
- **KV Merging & Eviction:** ToMe, ZSMerge / SELKV
- **Linear / Delta-Rule Recurrent Memory:** DeltaNet, GatedDeltaNet, Infini-attention, LoLA, Based
- **Evicted-Mass Supervision & Overflow Hierarchy:** Tensor Cache, IndexMem
- **Attention-to-Recurrent Retrofits:** LoLCATs, Jet-Nemotron, Opal

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
