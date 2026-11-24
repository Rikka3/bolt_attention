"""Cell 05 — Diagnostics & HubCheck (Decode Parity, KL Drift, 4k-32k Passkey, 32k PPL, Memory Bank)."""
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
ONPOLICY_CKPT = CKPT_DIR / "bolt3_qwen3_4b_onpolicy_latest.pth"
HEALED_CKPT = CKPT_DIR / "bolt3_qwen3_4b_latest.pth"
CKPT = ONPOLICY_CKPT if ONPOLICY_CKPT.is_file() else HEALED_CKPT

print(f"[ckpt] Using {CKPT}")
subprocess.run([sys.executable, "-m", "bolt3", "scan", "--ckpt", str(CKPT)], check=True)

subprocess.run(
    [
        sys.executable, "-m", "bolt3", "diagnose",
        "--ckpt", str(CKPT),
        "--donor", DONOR_REPO,
        "--n-prompts", "16",
        "--parity-T", "3072",
        "--out", str(CKPT_DIR / "diagnose_report.json"),
    ],
    check=True,
)

subprocess.run(
    [
        sys.executable, "-m", "bolt3", "hubcheck",
        "--ckpt", str(CKPT),
        "--donor", DONOR_REPO,
        "--passkey-lengths", "4096,16384,32768",
        "--ppl-tokens", "32768",
        "--out", str(CKPT_DIR / "hubcheck_report.json"),
    ],
    check=True,
)
