"""Command-line interface for Bolt-3 (`python -m bolt3 <subcommand>`)."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]

    p = argparse.ArgumentParser(
        prog="bolt3",
        description="Bolt-3: Constant-Memory Dual-Field Attention for Pretrained LLMs",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_conv = sub.add_parser("convert", help="Convert any HF Transformer (e.g., Qwen/Qwen3-4B) to Full Bolt-3 (no pure softmax)")
    p_conv.add_argument("--donor", required=True, help="HuggingFace repo ID or local snapshot directory")
    p_conv.add_argument("--out", default="bolt3_init.pth", help="Output .pth checkpoint path")
    p_conv.add_argument("--max-T", type=int, default=2048, help="Training sequence length (default: 2048)")
    p_conv.add_argument("--S", type=int, default=128, help="Attractor slot count S (default: 128)")
    p_conv.add_argument("--C", type=int, default=128, help="Local chunk size C (default: 128)")
    p_conv.add_argument("--no-freeze-proj", action="store_true", help="Allow training q/k/v/o projections during healing")
    p_conv.add_argument("--cold-layers", default=None, help="Optional cold-armed layer indices (e.g. '11,19,27', 'auto3', 'auto4', 'auto6')")
    p_conv.add_argument("--max-cold-blocks", type=int, default=256, help="Max chunk blocks retained per cold-armed layer (default: 256)")
    p_conv.add_argument("--cold-topk", type=int, default=0, help="Top-k cold block selection per query (0 = attend all cold blocks)")
    p_conv.add_argument("--hf-token", default=None, help="Optional HuggingFace token for gated models")

    p_heal = sub.add_parser("heal", help="Distill/heal a converted Bolt-3 checkpoint against its donor")
    p_heal.add_argument("--ckpt", required=True, help="Path to converted Bolt-3 checkpoint")
    p_heal.add_argument("--donor", default=None, help="Optional HF donor repo ID")
    p_heal.add_argument("--work", default=".", help="Working directory")
    p_heal.add_argument("--out-prefix", default="bolt3_healed", help="Output checkpoint prefix")
    p_heal.add_argument("--tokens", type=int, default=16_000_000, help="Healing token budget (default: 16M)")
    p_heal.add_argument("--data-tokens", type=int, default=100_000_000, help="Token cache size")
    p_heal.add_argument("--stage1-tokens", type=int, default=0, help="Stage-1 per-layer attention transfer tokens (0 = skip)")
    p_heal.add_argument("--stage1-onpolicy", action="store_true", help="Use on-policy hidden states after 25%% of Stage-1")
    p_heal.add_argument("--bs", type=int, default=8)
    p_heal.add_argument("--lr", type=float, default=8e-5)
    p_heal.add_argument("--warmup", type=int, default=150)
    p_heal.add_argument("--kd", type=float, default=1.0)
    p_heal.add_argument("--eval-every", type=int, default=250)
    p_heal.add_argument("--save-every", type=int, default=50)
    p_heal.add_argument("--seed", type=int, default=7777)
    p_heal.add_argument("--hf-repo", default=None, help="Optional HF repo for checkpoint backups")
    p_heal.add_argument("--highspeed", action="store_true", help="Enable fast VRAM mode on 48-96 GB GPUs")
    p_heal.add_argument("--resume", action="store_true", help="Resume from <work>/<out-prefix>_latest.pth (+ optim sidecar)")
    p_heal.add_argument("--freeze-mlp", action="store_true", default=True, help="Freeze donor MLPs/norms (default: True)")
    p_heal.add_argument("--unfreeze-mlp", action="store_true", help="Unfreeze donor MLPs/norms during healing")
    p_heal.add_argument("--train-proj", action="store_true", help="Also train q/k/v/o projections")
    p_heal.add_argument("--freeze-proj", action="store_true", help="Force freeze q/k/v/o projections during healing")
    p_heal.add_argument("--evict-loss", type=float, default=0.0, help="Weight of L_mem evicted-mass supervision loss (Step 2b)")
    p_heal.add_argument("--mres-tbptt", type=int, default=None, help="Chunk horizon before M_res detach (0 = full-seq unrolled)")
    p_heal.add_argument("--profile", choices=["reasoning", "chat", "base"], default=None)
    p_heal.add_argument("--ckpt-stride", type=int, default=None, help="Activation checkpoint block stride override")
    p_heal.add_argument("--micro-bs", type=int, default=None, help="Micro-batch size override")

    p_onp = sub.add_parser("onpolicy", help="Stage-3 chain distillation (teacher- or student-generated chains)")
    p_onp.add_argument("--ckpt", required=True, help="Healed Bolt-3 checkpoint")
    p_onp.add_argument("--donor", default=None)
    p_onp.add_argument("--work", default=".")
    p_onp.add_argument("--out-prefix", default="bolt3_onpolicy")
    p_onp.add_argument("--rounds", type=int, default=10)
    p_onp.add_argument("--prompts-per-round", type=int, default=256)
    p_onp.add_argument("--gen-bs", type=int, default=16)
    p_onp.add_argument("--max-new", type=int, default=None)
    p_onp.add_argument("--temperature", type=float, default=None)
    p_onp.add_argument("--lr", type=float, default=5e-5)
    p_onp.add_argument("--seqs-per-step", type=int, default=8)
    p_onp.add_argument("--corpus-every", type=int, default=4)
    p_onp.add_argument("--seed", type=int, default=4242)
    p_onp.add_argument("--resume", action="store_true")
    p_onp.add_argument("--unfreeze-mlp", action="store_true")
    p_onp.add_argument("--train-proj", action="store_true")
    p_onp.add_argument("--stop-capped-frac", type=float, default=0.5)
    p_onp.add_argument("--chain-source", choices=["student", "teacher"], default="student")
    p_onp.add_argument("--profile", choices=["reasoning", "chat", "base"], default=None)

    p_scan = sub.add_parser("scan", help="Static CPU scan of Bolt-3 gamma_delta, kappa, beta, eta, and res_norm")
    p_scan.add_argument("--ckpt", required=True, help="Path to .pth checkpoint")

    p_diag = sub.add_parser("diagnose", help="Decode parity + KL drift-by-position vs HF donor")
    p_diag.add_argument("--ckpt", required=True, help="Path to Bolt-3 .pth checkpoint")
    p_diag.add_argument("--donor", default=None, help="Optional HF donor repo ID")
    p_diag.add_argument("--n-prompts", type=int, default=16)
    p_diag.add_argument("--max-new", type=int, default=None)
    p_diag.add_argument("--parity-T", type=int, default=3072)
    p_diag.add_argument("--out", default=None)
    p_diag.add_argument("--parity-only", action="store_true")
    p_diag.add_argument("--profile", choices=["reasoning", "chat", "base"], default=None)

    p_hub = sub.add_parser("hubcheck", help="End-to-end release check (parity, passkey, 32k PPL, memory bank, server)")
    p_hub.add_argument("--ckpt", required=True, help="Path to Bolt-3 .pth checkpoint")
    p_hub.add_argument("--donor", default=None)
    p_hub.add_argument("--profile", choices=["reasoning", "chat", "base"], default=None)
    p_hub.add_argument("--passkey-lengths", default="4096,16384,32768")
    p_hub.add_argument("--ppl-tokens", type=int, default=32768)
    p_hub.add_argument("--skip", default="")
    p_hub.add_argument("--out", default=None)
    p_hub.add_argument("--no-donor", action="store_true")

    p_bench = sub.add_parser("benchmark", help="Run 7-task evaluation suite + 2k-64k speed/VRAM scaling")
    p_bench.add_argument("--ckpt", required=True, help="Path to Bolt-3 .pth checkpoint")
    p_bench.add_argument("--donor", default=None)
    p_bench.add_argument("--no-donor-compare", action="store_true")
    p_bench.add_argument("--limit", type=int, default=200)
    p_bench.add_argument("--gsm8k-limit", type=int, default=200)
    p_bench.add_argument("--batch-size", type=int, default=None, help="Eval batch size override (default: auto 32-64 on 40-96GB GPU)")
    p_bench.add_argument("--tasks", default="", help="Comma-separated task subset (e.g. MMLU,GSM8K)")
    p_bench.add_argument("--fresh-student", action="store_true", help="Ignore cached student/systems results and re-evaluate Bolt-3 cleanly")
    p_bench.add_argument("--fresh", action="store_true", help="Ignore entire cached benchmark JSON and re-evaluate both Bolt-3 and donor")
    p_bench.add_argument("--out", default=None)

    p_serve = sub.add_parser("serve", help="Start OpenAI-compatible /v1/chat/completions server")
    p_serve.add_argument("--ckpt", required=True, help="Path to Bolt-3 .pth checkpoint")
    p_serve.add_argument("--donor", default=None)
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--no-quantize", action="store_true")
    p_serve.add_argument(
        "--tier2",
        action="store_true",
        help="Enable Tier-2 .vsa cold-block overflow paging past GPU cold blocks (default topk=8)",
    )
    p_serve.add_argument(
        "--tier2-topk",
        type=int,
        default=None,
        help="Tier-2 interval fault-in top-k blocks per query (implies --tier2 when > 0)",
    )
    p_serve.add_argument(
        "--tier2-dir",
        default=None,
        help="Optional directory for Tier-2 .vsa block page files (default: CPU host RAM)",
    )
    p_serve.add_argument(
        "--tier2-max-cold",
        type=int,
        default=None,
        help="Optional GPU cold-block cap override on cold-armed layers when Tier-2 is active",
    )

    sub.add_parser("test", help="Run Bolt-3 parity, gradient, boundary, and cartridge test suite")

    args = p.parse_args(argv)

    if args.cmd == "convert":
        from .converter import convert_donor_model

        convert_donor_model(
            args.donor,
            args.out,
            max_T=args.max_T,
            bolt3_S=args.S,
            bolt3_C=args.C,
            freeze_proj=not args.no_freeze_proj,
            cold_layers=args.cold_layers,
            max_cold_blocks=args.max_cold_blocks,
            cold_topk=args.cold_topk,
            hf_token=args.hf_token,
        )
    elif args.cmd == "heal":
        from .converter import heal_bolt3_model

        heal_bolt3_model(
            args.ckpt,
            donor_id=args.donor,
            work=args.work,
            out_prefix=args.out_prefix,
            tokens_budget=args.tokens,
            data_tokens=args.data_tokens,
            stage1_tokens=args.stage1_tokens,
            bs=args.bs,
            lr=args.lr,
            warmup=args.warmup,
            kd=args.kd,
            eval_every=args.eval_every,
            save_every=args.save_every,
            seed=args.seed,
            hf_repo=args.hf_repo,
            highspeed=args.highspeed,
            resume=args.resume,
            freeze_mlp=(not args.unfreeze_mlp),
            train_proj=args.train_proj,
            force_freeze_proj=args.freeze_proj,
            stage1_onpolicy=args.stage1_onpolicy,
            profile=args.profile,
            ckpt_stride=args.ckpt_stride,
            micro_bs_override=args.micro_bs,
            evict_loss_w=args.evict_loss,
            mres_tbptt=args.mres_tbptt,
        )
    elif args.cmd == "onpolicy":
        from .converter import onpolicy_distill

        onpolicy_distill(
            args.ckpt,
            donor_id=args.donor,
            work=args.work,
            out_prefix=args.out_prefix,
            rounds=args.rounds,
            prompts_per_round=args.prompts_per_round,
            gen_bs=args.gen_bs,
            max_new=args.max_new,
            temperature=args.temperature,
            lr=args.lr,
            seqs_per_step=args.seqs_per_step,
            corpus_every=args.corpus_every,
            seed=args.seed,
            resume=args.resume,
            freeze_mlp=(not args.unfreeze_mlp),
            train_proj=args.train_proj,
            stop_capped_frac=args.stop_capped_frac,
            chain_source=args.chain_source,
            profile=args.profile,
        )
    elif args.cmd == "scan":
        from .converter import scan_checkpoint

        scan_checkpoint(args.ckpt)
    elif args.cmd == "diagnose":
        from .diagnose import run_diagnostics

        run_diagnostics(
            args.ckpt,
            donor_id=args.donor,
            n_prompts=args.n_prompts,
            max_new=args.max_new,
            parity_T=args.parity_T,
            out_json=args.out,
            parity_only=args.parity_only,
            profile=args.profile,
        )
    elif args.cmd == "hubcheck":
        from .hubcheck import run_hubcheck

        run_hubcheck(
            args.ckpt,
            donor_id=args.donor,
            profile=args.profile,
            passkey_lengths=tuple(int(x) for x in args.passkey_lengths.split(",") if x),
            ppl_tokens=args.ppl_tokens,
            skip=tuple(x for x in args.skip.split(",") if x),
            out_json=args.out,
            compare_donor=not args.no_donor,
        )
    elif args.cmd == "benchmark":
        from .benchmark import run_benchmark_suite

        run_benchmark_suite(
            args.ckpt,
            donor_id=args.donor,
            compare_donor=not args.no_donor_compare,
            limit=args.limit,
            gsm8k_limit=args.gsm8k_limit,
            out_json=args.out,
            batch_size=args.batch_size,
            tasks=tuple(x for x in args.tasks.split(",") if x),
            fresh_student=args.fresh_student,
            fresh=args.fresh,
        )
    elif args.cmd == "serve":
        from .server import run_server

        eff_t2_topk = int(args.tier2_topk) if args.tier2_topk is not None else (8 if args.tier2 else 0)
        run_server(
            args.ckpt,
            donor_tokenizer=args.donor,
            host=args.host,
            port=args.port,
            quantize_cpu=not args.no_quantize,
            tier2_topk=eff_t2_topk,
            tier2_dir=args.tier2_dir,
            tier2_max_cold=args.tier2_max_cold,
        )
    elif args.cmd == "test":
        from .tests import run_all

        run_all()


if __name__ == "__main__":
    main()
