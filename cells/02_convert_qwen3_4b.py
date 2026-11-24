"""Cell 02 — Convert Qwen/Qwen3-4B to 100% Full Bolt-3 (0 pure softmax layers)."""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
if not (REPO_ROOT / "bolt3").is_dir() and (REPO_ROOT.parent / "bolt3").is_dir():
    REPO_ROOT = REPO_ROOT.parent
os.chdir(REPO_ROOT)

DONOR_REPO = "Qwen/Qwen3-4B"
CKPT_DIR = REPO_ROOT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
INIT_CKPT = CKPT_DIR / "bolt3_qwen3_4b_init.pth"

MAX_T = 2048
BOLT3_S = 128
BOLT3_C = 128

cmd = [
    sys.executable, "-m", "bolt3", "convert",
    "--donor", DONOR_REPO,
    "--out", str(INIT_CKPT),
    "--max-T", str(MAX_T),
    "--S", str(BOLT3_S),
    "--C", str(BOLT3_C),
]
print("[run]", " ".join(cmd))
subprocess.run(cmd, check=True)

print("\n[scan] Inspecting initialized Bolt-3 parameters...")
subprocess.run([sys.executable, "-m", "bolt3", "scan", "--ckpt", str(INIT_CKPT)], check=True)

print("\n[parity] Checking Bolt-3 chunked forward vs recurrent decode parity...")
subprocess.run(
    [
        sys.executable, "-m", "bolt3", "diagnose",
        "--ckpt", str(INIT_CKPT),
        "--donor", DONOR_REPO,
        "--parity-only",
        "--parity-T", "1024",
    ],
    check=True,
)
