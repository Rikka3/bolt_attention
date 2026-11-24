"""Cell 07 — Interactive Bolt-3 Chat, .vsa Cartridge Demo, & OpenAI Server (latest)."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
if not (REPO_ROOT / "bolt3").is_dir() and (REPO_ROOT.parent / "bolt3").is_dir():
    REPO_ROOT = REPO_ROOT.parent
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bolt3.attention import Bolt3Decoder
from bolt3.memory_bank import Bolt3MemoryBank
from bolt3.server import generate_completion, load_model_for_serving

DONOR_REPO = "Qwen/Qwen3-4B"
CKPT_DIR = REPO_ROOT / "checkpoints"
ONPOLICY_CKPT = CKPT_DIR / "bolt3_qwen3_4b_onpolicy_latest.pth"
HEALED_CKPT = CKPT_DIR / "bolt3_qwen3_4b_latest.pth"
INIT_CKPT = CKPT_DIR / "bolt3_qwen3_4b_init.pth"
CKPT = next((p for p in (ONPOLICY_CKPT, HEALED_CKPT, INIT_CKPT) if p.is_file()), HEALED_CKPT)

START_HTTP_SERVER = False  # Set True to start OpenAI-compatible server on port 8000
HOST = "0.0.0.0"
PORT = 8000

print(f"[chat] Loading {CKPT}...")
model, tok, device = load_model_for_serving(str(CKPT), donor_tokenizer=DONOR_REPO, quantize_cpu=False)
dec = Bolt3Decoder(model)

flat_b, per_tok_b = Bolt3Decoder.state_bytes(model)
print(f"[state] Fixed Bolt-3 state: {flat_b / 1024**2:.2f} MB | KV growth per token: {per_tok_b} bytes")

reply = generate_completion(
    model,
    tok,
    device,
    [{"role": "user", "content": "Explain in 2 concise sentences why constant-memory attention helps long-context inference."}],
    max_tokens=512,
    temperature=0.6,
    decoder=dec,
)
print("\n[reply]\n" + reply)

with tempfile.TemporaryDirectory() as tmp_bank:
    bank = Bolt3MemoryBank(model, tok, bank_dir=tmp_bank)
    bank.memorize_to_disk("vault", "The emergency override passphrase for Vault 9 is SOLAR-QUARTZ-42.")
    res = bank.ask("What is the emergency override passphrase for Vault 9?", max_new_tokens=32)
    print("\n[memory_bank]", res)

if START_HTTP_SERVER:
    subprocess.run(
        [sys.executable, "-m", "bolt3", "serve", "--ckpt", str(CKPT), "--donor", DONOR_REPO, "--host", HOST, "--port", str(PORT)],
        check=True,
    )
