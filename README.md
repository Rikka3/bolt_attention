# Bolt-3 Attention (`bolt3`)

**Constant-memory Dual-Field Attention** for converting pretrained Transformer LLMs (such as `Qwen/Qwen3-4B`) into **100% Full Bolt-3** fixed-state models with **zero pure Softmax layers**.

---

## Architecture: Bolt-3 Dual-Field Attention

Every layer in a converted model uses **`Bolt3Attn`** (`bolt3/attention.py`), processing sequences in chunks of size `C` (default `128`) over **three fixed-size memory fields** per KV head so inference RAM remains strictly `O(1)` at any context length:

1. **Attention Sinks + Local Window**
   - Keeps the first `n_sinks = 4` tokens plus the previous and current chunks (up to `2C = 256` tokens) with exact causal Softmax attention.
2. **Diversity Softmax Attractors (`S = 128` slots)**
   - As older chunks leave the local window, tokens are consolidated into `S` attractor slots storing raw keys (`ak`), values (`av`), running `mass` (adds `kappa * log(mass)` attention bias), mean RoPE position (`pm`), and `util`.
   - **Branchless Merge:** Tokens matching an existing slot (`cos_k >= 0.94` and `cos_v >= 0.88`) merge into it and update its `mass` and mean position.
   - **Diversity Eviction:** Unmerged tokens scored by value prediction error (`rel_err`) and key novelty (`1 - best_sim_k`) replace low-utility / redundant slots (`gamma_u = 0.985`).
3. **Gershgorin Eviction-to-Delta Residual Memory (`M_res`)**
   - Remaining unmerged/evicted tokens write into a fixed `[HKV, HD, HD]` `float32` associative state `M_res` using Gershgorin-normalized delta updates (`beta_proj`, `gamma_delta`) and gated readout (`eta_proj`, `res_norm` initialized at `0.05` for near-identity surgery).

---

## Molab (RTX 6000 Pro) Cells for `Qwen/Qwen3-4B`

Runnable notebook-ready cells are provided in `cells/` (ordered sequentially; the newest step is tagged `(latest)`):

1. [`cells/01_setup_molab_rtx6000.py`](cells/01_setup_molab_rtx6000.py) — Environment check, dependencies, and `python -m bolt3 test`.
2. [`cells/02_convert_qwen3_4b.py`](cells/02_convert_qwen3_4b.py) — Convert `Qwen/Qwen3-4B` to Full Bolt-3 (36/36 `Bolt3Attn` layers, 0 pure Softmax), `scan`, and parity check.
3. [`cells/03_heal_qwen3_4b.py`](cells/03_heal_qwen3_4b.py) — Knowledge distillation healing (`--freeze-mlp --highspeed --resume`), auto-tuned for 48 GB / 96 GB RTX 6000 Pro.
4. [`cells/04_onpolicy_qwen3_4b.py`](cells/04_onpolicy_qwen3_4b.py) — Stage-3 chain distillation (`onpolicy`) to remove long-chain reasoning drift.
5. [`cells/05_diagnose_and_hubcheck.py`](cells/05_diagnose_and_hubcheck.py) — Full decode parity (`T=3072`), KL drift probe, 4k–32k passkey recall, 32k perplexity, and `.vsa` cartridge check.
6. [`cells/06_benchmark_qwen3_4b.py`](cells/06_benchmark_qwen3_4b.py) — 7-task evaluation suite (MMLU, ARC-C, HellaSwag, GSM8K, HumanEval, MBPP, IFEval) + 2k–64k throughput/VRAM chart.
7. [`cells/07_serve_and_chat (latest).py`](cells/07_serve_and_chat%20(latest).py) — Interactive chat, `.vsa` memory bank cartridge test, and OpenAI-compatible HTTP server.

---

## CLI Quick Reference

```bash
# 1. Run Bolt-3 parity, gradient, boundary, and erasure unit tests
python -m bolt3 test

# 2. Convert Qwen/Qwen3-4B to 100% Full Bolt-3 (0 pure softmax layers)
python -m bolt3 convert --donor Qwen/Qwen3-4B --out checkpoints/bolt3_qwen3_4b_init.pth

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

# 5. Diagnose / HubCheck / Benchmark / Serve
python -m bolt3 scan --ckpt checkpoints/bolt3_qwen3_4b_latest.pth
python -m bolt3 diagnose --ckpt checkpoints/bolt3_qwen3_4b_latest.pth --donor Qwen/Qwen3-4B --parity-only
python -m bolt3 hubcheck --ckpt checkpoints/bolt3_qwen3_4b_latest.pth --donor Qwen/Qwen3-4B
python -m bolt3 benchmark --ckpt checkpoints/bolt3_qwen3_4b_latest.pth --donor Qwen/Qwen3-4B
python -m bolt3 serve --ckpt checkpoints/bolt3_qwen3_4b_latest.pth --donor Qwen/Qwen3-4B --port 8000
```

---

## Repository Layout

- `bolt3/attention.py` — `Bolt3Attn` (chunked training forward) and `Bolt3Decoder` (constant-memory recurrent inference & `memorize`/`forget`).
- `bolt3/nn.py` — Shared `RMSNorm`, `RoPE`, `SwiGLUMLP`, `SoftmaxAttn` (donor teacher replica), and `repeat_kv`.
- `bolt3/model.py` — `Bolt3Block`, `Bolt3LM`, `FP32_KEYS`, and `load_bolt3_checkpoint` (with legacy `.vsa.` key remap).
- `bolt3/inspector.py` — HuggingFace snapshot download, safetensor inspection, and weight mapping.
- `bolt3/profiles.py` — Donor sampling & data-mix profile detection (`reasoning`, `chat`, `base`).
- `bolt3/converter.py` — `convert_donor_model`, Stage-1 attention transfer, Stage-2 KD healing, Stage-3 `onpolicy_distill`, and `scan_checkpoint`.
- `bolt3/memory_bank.py` — Disk-backed `.vsa` neural state cartridges (`Bolt3MemoryBank`).
- `bolt3/diagnose.py`, `hubcheck.py`, `benchmark.py`, `server.py`, `tests.py`, `cli.py` — Verification, benchmarking, OpenAI-compatible serving, and CLI entrypoint.
- `cells/` — Molab / Colab runnable cells for converting, healing, evaluating, and serving `Qwen/Qwen3-4B`.
