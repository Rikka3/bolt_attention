"""Bolt-3 parity, gradient, boundary, and checkpoint test suite (`python -m bolt3 test`)."""
from __future__ import annotations

import os
import tempfile

import torch

from .attention import Bolt3Decoder
from .converter import scan_checkpoint
from .model import FP32_KEYS, Bolt3LM, load_bolt3_checkpoint


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_forward_decode_parity():
    dev = _device()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32 if dev.type == "cuda" else None
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    torch.manual_seed(0)
    model = (
        Bolt3LM(
            vocab_size=512,
            dim=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate=256,
            max_T=512,
            bolt3_S=16,
            bolt3_C=32,
            qk_norm=True,
        )
        .to(dev)
        .float()
        .eval()
    )
    with torch.no_grad():
        for blk in model.blocks:
            blk.attn.eta_proj.bias.fill_(0.5)
            blk.attn.res_norm.weight.fill_(0.5)

    dec = Bolt3Decoder(model)
    T = 160  # 5 chunks of C=32 -> exercises sinks, attractors, merge/evict, and M_res
    pat = torch.randint(0, 512, (2, 32), device=dev)
    ids = torch.cat([pat, pat, torch.randint(0, 512, (2, T - 64), device=dev)], dim=1)
    with torch.no_grad():
        full_logits = model.lm_head(model(ids))
        states, kv = Bolt3Decoder.blank_state(model, 2, dev, state_dtype=torch.float32)
        max_err = 0.0
        agree = 0
        for t in range(T):
            h, states, kv = dec.step(ids[:, t : t + 1], t, states, kv)
            rec = model.lm_head(h[:, -1])
            ref = full_logits[:, t]
            max_err = max(max_err, float((rec - ref).abs().max().item()))
            agree += int((rec.argmax(-1) == ref.argmax(-1)).sum().item())

    if old_tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    top1_pct = 100.0 * agree / (2 * T)
    print(f"[parity C=32 S=16 T={T}]  max logit diff: {max_err:.2e}  top1: {top1_pct:.2f}%")
    assert max_err < 5e-3 and top1_pct == 100.0


def test_chunk_boundary_step():
    dev = _device()
    torch.manual_seed(1)
    model = (
        Bolt3LM(
            vocab_size=256,
            dim=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate=128,
            max_T=256,
            bolt3_S=8,
            bolt3_C=16,
        )
        .to(dev)
        .float()
        .eval()
    )
    with torch.no_grad():
        for blk in model.blocks:
            blk.attn.eta_proj.bias.fill_(0.4)
            blk.attn.res_norm.weight.fill_(0.4)

    dec = Bolt3Decoder(model)
    ids = torch.randint(0, 256, (1, 80), device=dev)
    with torch.no_grad():
        full_logits = model.lm_head(model(ids))
        states, kv = Bolt3Decoder.blank_state(model, 1, dev, state_dtype=torch.float32)
        pos = 0
        chunks = []
        for size in (5, 29, 1, 45):
            h, states, kv = dec.step(ids[:, pos : pos + size], pos, states, kv)
            chunks.append(model.lm_head(h))
            pos += size
        rec_logits = torch.cat(chunks, dim=1)
        err = float((rec_logits - full_logits).abs().max().item())
    print(f"[boundary step]  max logit diff: {err:.2e}")
    assert err < 5e-3


def test_gradients_flow():
    dev = _device()
    torch.manual_seed(2)
    model = (
        Bolt3LM(
            vocab_size=256,
            dim=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate=128,
            max_T=256,
            bolt3_S=8,
            bolt3_C=16,
        )
        .to(dev)
        .float()
        .train()
    )
    for T in (24, 80):
        model.zero_grad(set_to_none=True)
        x = torch.randint(0, 256, (2, T), device=dev)
        h = model(x)
        loss = model.lm_head(h).pow(2).mean()
        loss.backward()
        for blk in model.blocks:
            for n, p in blk.attn.named_parameters():
                if any(k in n for k in FP32_KEYS):
                    assert p.grad is not None and torch.isfinite(p.grad).all(), f"Missing/non-finite grad for {n}"

    # Verify non-zero gradients across both merge (kappa) and M_res paths once eta_proj opens
    with torch.no_grad():
        for blk in model.blocks:
            blk.attn.eta_proj.bias.fill_(0.25)
    model.zero_grad(set_to_none=True)
    pat = torch.randint(0, 256, (2, 16), device=dev)
    x = torch.cat([pat, pat, pat, torch.randint(0, 256, (2, 32), device=dev)], dim=1)
    model.lm_head(model(x)).pow(2).mean().backward()
    for blk in model.blocks:
        for n, p in blk.attn.named_parameters():
            if any(k in n for k in FP32_KEYS):
                assert p.grad.abs().max().item() > 0.0, f"Zero grad for {n}"
    print("[gradients]  short & long sequence autograd OK")


def test_memorize_forget_and_state_bytes():
    dev = _device()
    torch.manual_seed(3)
    model = (
        Bolt3LM(
            vocab_size=256,
            dim=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate=128,
            max_T=256,
            bolt3_S=8,
            bolt3_C=16,
        )
        .to(dev)
        .float()
        .eval()
    )
    flat, per_tok = Bolt3Decoder.state_bytes(model)
    assert flat > 0 and per_tok == 0

    dec = Bolt3Decoder(model)
    doc = torch.randint(0, 256, (1, 48), device=dev)
    probe = torch.randint(0, 256, (1, 16), device=dev)

    states, kv = Bolt3Decoder.blank_state(model, 1, dev)
    states, kv, pos, traces = dec.memorize(doc, states, kv, 0)
    states = dec.forget(traces, states, next_pos=pos)
    zero_err = max(float(s.abs().max().item()) for s in states)

    st2, kv2 = Bolt3Decoder.blank_state(model, 1, dev)
    st2, kv2, pos2, tr2 = dec.memorize(doc, st2, kv2, 0)
    st_doc_only = [s.clone() for s in st2]
    _, st2, kv2 = dec.step(probe, pos2, st2, kv2)
    pos2 += probe.shape[1]
    pre_forget = [s.clone() for s in st2]
    st2 = dec.forget(tr2, st2, next_pos=pos2)
    decay_err = max(
        float(
            (st2[i] - (pre_forget[i] - model.blocks[i].attn.gamma_delta().view(1, -1, 1, 1) * st_doc_only[i]))
            .abs()
            .max()
            .item()
        )
        for i in range(len(st2))
    )
    print(f"[memorize/forget]  zero-erase: {zero_err:.2e}  decay-bookkeeping: {decay_err:.2e}  per_tok_bytes: {per_tok}")
    assert zero_err < 1e-5 and decay_err < 1e-5


def test_qwen3_style_gqa_and_legacy_remap():
    dev = _device()
    torch.manual_seed(4)
    model = (
        Bolt3LM(
            vocab_size=256,
            dim=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=1,
            head_dim=64,
            intermediate=256,
            max_T=256,
            qk_norm=True,
            qkv_bias=False,
            tie_word_embeddings=True,
            bolt3_S=128,
            bolt3_C=128,
        )
        .to(dev)
        .float()
        .eval()
    )
    legacy_sd = {k.replace(".attn.", ".vsa.", 1): v.cpu() for k, v in model.state_dict().items()}
    legacy_cfg = dict(model.config, layer_types=["vsa", "vsa"])
    with tempfile.TemporaryDirectory() as d:
        ckpt_p = os.path.join(d, "legacy_bolt3.pth")
        torch.save(
            {"model": legacy_sd, "step": 10, "config": legacy_cfg, "injection": {"donor": "Qwen/Qwen3-4B"}},
            ckpt_p,
        )
        reloaded, _, donor_id, step, _ = load_bolt3_checkpoint(ckpt_p, device=dev)
        assert donor_id == "Qwen/Qwen3-4B" and step == 10
        x = torch.randint(0, 256, (1, 40), device=dev)
        with torch.no_grad():
            diff = (model(x).float() - reloaded(x).float()).abs().max().item()
        tol = 5e-2 if dev.type == "cuda" else 1e-3
        assert diff < tol, f"legacy remap diff too high: {diff}"
        scan_out = scan_checkpoint(ckpt_p)
        assert len(scan_out) == 2
    print("[qwen3 GQA & legacy remap]  PASS")


def test_left_padded_batch_parity():
    dev = _device()
    torch.manual_seed(5)
    model = (
        Bolt3LM(
            vocab_size=256,
            dim=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate=128,
            max_T=256,
            bolt3_S=8,
            bolt3_C=16,
            qk_norm=True,
        )
        .to(dev)
        .float()
        .eval()
    )
    with torch.no_grad():
        for blk in model.blocks:
            blk.attn.eta_proj.bias.fill_(0.4)
            blk.attn.res_norm.weight.fill_(0.4)

    dec = Bolt3Decoder(model)
    p1 = torch.randint(1, 256, (1, 14), device=dev)   # < 1 chunk
    p2 = torch.randint(1, 256, (1, 53), device=dev)   # > 3 chunks (triggers sinks, attractors, M_res)
    max_l = 53
    ids_b = torch.zeros((2, max_l), dtype=torch.long, device=dev)
    mask_b = torch.zeros((2, max_l), dtype=torch.long, device=dev)
    ids_b[0, max_l - 14 :] = p1[0]
    mask_b[0, max_l - 14 :] = 1
    ids_b[1] = p2[0]
    mask_b[1] = 1

    with torch.no_grad():
        ref1 = dec.generate(p1, 10, temperature=0.0)[0, 14:].tolist()
        ref2 = dec.generate(p2, 10, temperature=0.0)[0, 53:].tolist()
        batched, _ = dec.generate_batch(ids_b, 10, temperature=0.0, attn_mask=mask_b)
        b1, b2 = batched[0].tolist(), batched[1].tolist()
    assert b1 == ref1 and b2 == ref2

    # Also test short prompts (< C = 16) where prev_valid is None during decode
    s1 = torch.randint(1, 256, (1, 5), device=dev)
    s2 = torch.randint(1, 256, (1, 11), device=dev)
    ids_s = torch.zeros((2, 11), dtype=torch.long, device=dev)
    mask_s = torch.zeros((2, 11), dtype=torch.long, device=dev)
    ids_s[0, 6:] = s1[0]
    mask_s[0, 6:] = 1
    ids_s[1] = s2[0]
    mask_s[1] = 1
    with torch.no_grad():
        sref1 = dec.generate(s1, 18, temperature=0.0)[0, 5:].tolist()
        sref2 = dec.generate(s2, 18, temperature=0.0)[0, 11:].tolist()
        sbat, _ = dec.generate_batch(ids_s, 18, temperature=0.0, attn_mask=mask_s)
        sb1, sb2 = sbat[0].tolist(), sbat[1].tolist()
    assert sb1 == sref1 and sb2 == sref2

    # Verify exact multi-chunk (5+ chunks) hidden-state parity when S == C and rows transition
    # from partial left-padded fill -> full-chunk consolidation.
    model2 = (
        Bolt3LM(
            vocab_size=128,
            dim=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate=128,
            max_T=512,
            rope_theta=1000000.0,
            eps=1e-6,
            qk_norm=True,
            bolt3_S=16,
            bolt3_C=16,
            bolt3_n_sinks=4,
        )
        .to(dev)
        .float()
        .eval()
    )
    with torch.no_grad():
        for blk in model2.blocks:
            blk.attn.eta_proj.bias.fill_(0.5)
            blk.attn.beta_proj.bias.fill_(0.5)
            blk.attn.res_norm.weight.fill_(1.0)
    dec2 = Bolt3Decoder(model2)
    q1 = torch.randint(1, 128, (1, 7), device=dev)
    q2 = torch.randint(1, 128, (1, 13), device=dev)
    g1 = torch.randint(1, 128, (1, 64), device=dev)
    g2 = torch.randint(1, 128, (1, 64), device=dev)
    with torch.no_grad():
        h_u2, st2, kv2 = dec2.prefill(q2)
        hs_u2 = [h_u2[:, -1:]]
        for i in range(64):
            hi, st2, kv2 = dec2.step(g2[:, i : i + 1], q2.shape[1] + i, st2, kv2)
            hs_u2.append(hi)
        hs_u2 = torch.cat(hs_u2, dim=1)

        xb = torch.zeros((2, 13), dtype=torch.long, device=dev)
        xb[0, 6:] = q1[0]
        xb[1] = q2[0]
        mb = torch.zeros((2, 13), dtype=torch.long, device=dev)
        mb[0, 6:] = 1
        mb[1] = 1
        hb, stb, kvb = dec2.prefill(xb, attn_mask=mb)
        hs_b = [hb[:, -1:]]
        rp = mb.long().sum(dim=-1, keepdim=True)
        gb = torch.cat([g1, g2], dim=0)
        for i in range(64):
            hi, stb, kvb = dec2.step(gb[:, i : i + 1], 13 + i, stb, kvb, pos_ids=rp)
            rp = rp + 1
            hs_b.append(hi)
        hs_b = torch.cat(hs_b, dim=1)

        hp0, stp0, kvp0 = dec2.prefill(xb[0:1], attn_mask=mb[0:1])
        hs_p0 = [hp0[:, -1:]]
        rp0 = mb[0:1].long().sum(dim=-1, keepdim=True)
        for i in range(64):
            hi, stp0, kvp0 = dec2.step(g1[:, i : i + 1], 13 + i, stp0, kvp0, pos_ids=rp0)
            rp0 = rp0 + 1
            hs_p0.append(hi)
        hs_p0 = torch.cat(hs_p0, dim=1)

    d_unpad = float((hs_b[1:2] - hs_u2).abs().max().item())
    d_pad0 = float((hs_b[0:1] - hs_p0).abs().max().item())
    dec2.close()
    print(f"[left-padded batch parity]  long=({b1 == ref1},{b2 == ref2})  short<C=({sb1 == sref1},{sb2 == sref2})  5chunk_diff=({d_pad0:.2e},{d_unpad:.2e})")
    assert d_unpad < 1e-4 and d_pad0 < 1e-4


def test_cold_armed_parity():
    dev = _device()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32 if dev.type == "cuda" else None
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for cold_topk in (0, 1):
            torch.manual_seed(7 + cold_topk)
            model = (
                Bolt3LM(
                    vocab_size=256,
                    dim=64,
                    num_layers=2,
                    num_heads=4,
                    num_kv_heads=2,
                    intermediate=128,
                    max_T=512,
                    rope_theta=1000000.0,
                    eps=1e-6,
                    qk_norm=True,
                    bolt3_S=8,
                    bolt3_C=16,
                    bolt3_n_sinks=4,
                    bolt3_cold_layers=[1],
                    bolt3_max_cold_blocks=2,
                    bolt3_cold_topk=cold_topk,
                )
                .to(dev)
                .float()
                .eval()
            )
            with torch.no_grad():
                for blk in model.blocks:
                    blk.attn.eta_proj.bias.fill_(0.5)
                    blk.attn.beta_proj.bias.fill_(0.5)
                    blk.attn.res_norm.weight.fill_(0.5)

            flat_b, per_tok = Bolt3Decoder.state_bytes(model)
            assert flat_b > 0 and per_tok == 2 * 2 * 16 * 2, f"unexpected cold state_bytes: {(flat_b, per_tok)}"

            x = torch.randint(1, 256, (1, 110), device=dev)
            dec = Bolt3Decoder(model)
            with torch.no_grad():
                h_fwd = model(x)
                h_pre, st, kv = dec.prefill(x[:, :70])
                steps = [h_pre]
                for p in range(70, 110):
                    hp, st, kv = dec.step(x[:, p : p + 1], p, st, kv)
                    steps.append(hp)
                h_dec = torch.cat(steps, dim=1)
            diff = float((h_fwd - h_dec).abs().max().item())
            assert kv[0].get("cold_k_blk") is None
            assert kv[1]["cold_k_blk"].shape[2] == 2
            assert kv[1]["ak"] is not None and float(st[1].abs().max().item()) > 0.0
            dec.close()
            assert diff < 1e-4, f"cold_topk={cold_topk} forward/decode diff too high: {diff}"

        # Left-padded multi-chunk + overflow batch parity with cold layer [1]
        torch.manual_seed(11)
        mb_model = (
            Bolt3LM(
                vocab_size=128,
                dim=64,
                num_layers=2,
                num_heads=4,
                num_kv_heads=2,
                intermediate=128,
                max_T=512,
                rope_theta=1000000.0,
                eps=1e-6,
                qk_norm=True,
                bolt3_S=8,
                bolt3_C=16,
                bolt3_n_sinks=4,
                bolt3_cold_layers=[1],
                bolt3_max_cold_blocks=2,
                bolt3_cold_topk=0,
            )
            .to(dev)
            .float()
            .eval()
        )
        with torch.no_grad():
            for blk in mb_model.blocks:
                blk.attn.eta_proj.bias.fill_(0.5)
                blk.attn.beta_proj.bias.fill_(0.5)
                blk.attn.res_norm.weight.fill_(1.0)
        dec_b = Bolt3Decoder(mb_model)
        q1 = torch.randint(1, 128, (1, 7), device=dev)
        q2 = torch.randint(1, 128, (1, 13), device=dev)
        g1 = torch.randint(1, 128, (1, 80), device=dev)
        g2 = torch.randint(1, 128, (1, 80), device=dev)
        with torch.no_grad():
            hu2, su2, kvu2 = dec_b.prefill(q2)
            l_u2 = [hu2[:, -1:]]
            for i in range(80):
                hi, su2, kvu2 = dec_b.step(g2[:, i : i + 1], 13 + i, su2, kvu2)
                l_u2.append(hi)
            l_u2 = torch.cat(l_u2, dim=1)

            xb = torch.zeros((2, 13), dtype=torch.long, device=dev)
            xb[0, 6:] = q1[0]
            xb[1] = q2[0]
            am = torch.zeros((2, 13), dtype=torch.long, device=dev)
            am[0, 6:] = 1
            am[1] = 1
            hb, sb, kvb = dec_b.prefill(xb, attn_mask=am)
            l_b = [hb[:, -1:]]
            rp = am.long().sum(dim=-1, keepdim=True)
            gb = torch.cat([g1, g2], dim=0)
            for i in range(80):
                hi, sb, kvb = dec_b.step(gb[:, i : i + 1], 13 + i, sb, kvb, pos_ids=rp)
                rp = rp + 1
                l_b.append(hi)
            l_b = torch.cat(l_b, dim=1)

            hp0, sp0, kvp0 = dec_b.prefill(xb[0:1], attn_mask=am[0:1])
            l_p0 = [hp0[:, -1:]]
            rp0 = am[0:1].long().sum(dim=-1, keepdim=True)
            for i in range(80):
                hi, sp0, kvp0 = dec_b.step(g1[:, i : i + 1], 13 + i, sp0, kvp0, pos_ids=rp0)
                rp0 = rp0 + 1
                l_p0.append(hi)
            l_p0 = torch.cat(l_p0, dim=1)
        d_bu = float((l_b[1:2] - l_u2).abs().max().item())
        d_bp = float((l_b[0:1] - l_p0).abs().max().item())
        dec_b.close()
        assert d_bu < 1e-4 and d_bp < 1e-4, f"cold batch parity diff too high: ({d_bp}, {d_bu})"

        bf16_msg = "cpu"
        if dev.type == "cuda":
            m_bf = mb_model.to(torch.bfloat16).eval()
            for blk in m_bf.blocks:
                for p in (blk.attn.kappa, blk.attn.log_gamma):
                    p.data = p.data.float()
                blk.attn.beta_proj.float()
                blk.attn.eta_proj.float()
                blk.attn.res_norm.float()
            d_cg = Bolt3Decoder(m_bf, use_cuda_graph=True)
            d_eg = Bolt3Decoder(m_bf, use_cuda_graph=False)
            with torch.no_grad():
                xc = torch.randint(1, 128, (1, 120), device=dev)
                hc, sc_c, kvc = d_cg.prefill(xc[:, :20])
                he, se, kve = d_eg.prefill(xc[:, :20])
                o_c, o_e = [hc], [he]
                for p in range(20, 120):
                    hc, sc_c, kvc = d_cg.step(xc[:, p : p + 1], p, sc_c, kvc)
                    he, se, kve = d_eg.step(xc[:, p : p + 1], p, se, kve)
                    o_c.append(hc)
                    o_e.append(he)
                d_bf = float((torch.cat(o_c, dim=1) - torch.cat(o_e, dim=1)).abs().max().item())
            d_cg.close()
            d_eg.close()
            assert d_bf < 5e-2, f"cuda bf16 graph vs eager diff too high: {d_bf}"
            bf16_msg = f"{d_bf:.2e}"

        print(f"[cold-armed parity]         fwd_vs_dec={diff:.2e}  batch=({d_bp:.2e},{d_bu:.2e})  bf16_graph={bf16_msg}")
    finally:
        if old_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32


def test_qwen3_5_hybrid_gdn_and_packed_gate_parity():
    dev = _device()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32 if dev.type == "cuda" else None
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.manual_seed(11)
        cfg = dict(
            vocab_size=256,
            dim=64,
            num_layers=4,
            num_heads=4,
            num_kv_heads=2,
            head_dim=32,
            intermediate=128,
            max_T=256,
            rope_theta=10_000_000.0,
            partial_rotary_factor=0.25,
            qk_norm=True,
            rms_norm_offset=1.0,
            packed_q_gate=True,
            bolt3_S=16,
            bolt3_C=16,
            bolt3_n_sinks=4,
            layer_types=["linear_attention", "bolt3", "linear_attention", "bolt3"],
            linear_conv_kernel_dim=4,
            linear_key_head_dim=16,
            linear_num_key_heads=2,
            linear_value_head_dim=16,
            linear_num_value_heads=4,
        )
        m = Bolt3LM(**cfg).to(dev).float().eval()
        with torch.no_grad():
            for blk in m.blocks:
                if hasattr(blk.attn, "eta_proj"):
                    blk.attn.eta_proj.bias.fill_(0.4)
                    blk.attn.res_norm.weight.fill_(0.5)

        x = torch.randint(1, 256, (1, 70), device=dev)
        with torch.no_grad():
            h_fwd = m(x)
            dec = Bolt3Decoder(m)
            hp, st, kv = dec.prefill(x[:, :37])
            outs = [hp]
            for pos in range(37, 70):
                hs, st, kv = dec.step(x[:, pos : pos + 1], pos, st, kv)
                outs.append(hs)
            h_dec = torch.cat(outs, dim=1)
            diff = float((h_fwd - h_dec).abs().max().item())
        tol = 5e-2 if dev.type == "cuda" else 1e-4
        assert diff < tol, f"qwen3.5 hybrid forward vs decode diff={diff}"

        # Gradient flow check across both GatedDeltaNet and Bolt3Attn parameters
        m.train()
        mt = Bolt3LM(**cfg).to(dev).float().train()
        with torch.no_grad():
            for p_src, p_dst in zip(m.parameters(), mt.parameters()):
                p_dst.copy_(p_src)
        xt = torch.randint(0, 256, (2, 70), device=dev)
        ht = mt(xt)
        loss = ht.pow(2).mean()
        loss.backward()
        assert mt.blocks[0].attn.A_log.grad is not None and mt.blocks[0].attn.conv1d.weight.grad is not None
        assert mt.blocks[1].attn.log_gamma.grad is not None and mt.blocks[1].attn.q_proj.weight.grad is not None

        # Left-padded batch generation parity (crossing both prefill & chunk consolidation boundaries)
        m.eval()
        dec_b = Bolt3Decoder(m)
        p1 = torch.randint(1, 256, (1, 5), device=dev)
        p2 = torch.randint(1, 256, (1, 45), device=dev)
        ids_b = torch.zeros((2, 45), dtype=torch.long, device=dev)
        mask_b = torch.zeros((2, 45), dtype=torch.long, device=dev)
        ids_b[0, 40:] = p1[0]
        mask_b[0, 40:] = 1
        ids_b[1] = p2[0]
        mask_b[1] = 1
        with torch.no_grad():
            ref1 = dec_b.generate(p1, 20, temperature=0.0)[0, 5:].tolist()
            ref2 = dec_b.generate(p2, 20, temperature=0.0)[0, 45:].tolist()
            bat, _ = dec_b.generate_batch(ids_b, 20, temperature=0.0, attn_mask=mask_b)
            assert bat[0].tolist() == ref1 and bat[1].tolist() == ref2

        flat_b, per_tok_b = Bolt3Decoder.state_bytes(m)
        assert flat_b > 0 and per_tok_b == 0

        bf16_msg = "cpu-skip"
        if dev.type == "cuda":
            m_bf = Bolt3LM(**cfg).to(dev).to(torch.bfloat16).eval()
            for n, p in m_bf.named_parameters():
                if any(k in n for k in ("log_gamma", "kappa", "beta_proj", "eta_proj", "res_norm", "A_log", "dt_bias")):
                    p.data = p.data.float()
            d_cg = Bolt3Decoder(m_bf, use_cuda_graph=True)
            d_eg = Bolt3Decoder(m_bf, use_cuda_graph=False)
            with torch.no_grad():
                xc = torch.randint(1, 256, (1, 90), device=dev)
                hc, sc_c, kvc = d_cg.prefill(xc[:, :20])
                he, se, kve = d_eg.prefill(xc[:, :20])
                oc, oe = [hc], [he]
                for p in range(20, 90):
                    hc, sc_c, kvc = d_cg.step(xc[:, p : p + 1], p, sc_c, kvc)
                    he, se, kve = d_eg.step(xc[:, p : p + 1], p, se, kve)
                    oc.append(hc)
                    oe.append(he)
                d_bf = float((torch.cat(oc, dim=1) - torch.cat(oe, dim=1)).abs().max().item())
            d_cg.close()
            d_eg.close()
            assert d_bf < 5e-2, f"qwen3.5 bf16 graph vs eager diff={d_bf}"
            bf16_msg = f"{d_bf:.2e}"

        print(f"[qwen3.5 hybrid GDN+gate]  fwd_vs_dec={diff:.2e}  batch=exact  bf16_graph={bf16_msg}")
    finally:
        if old_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32


def run_all():
    print(f"Running Bolt-3 test suite on {_device()}...")
    test_forward_decode_parity()
    test_chunk_boundary_step()
    test_gradients_flow()
    test_memorize_forget_and_state_bytes()
    test_qwen3_style_gqa_and_legacy_remap()
    test_left_padded_batch_parity()
    test_cold_armed_parity()
    test_qwen3_5_hybrid_gdn_and_packed_gate_parity()
    print("ALL BOLT-3 TESTS PASSED")


if __name__ == "__main__":
    run_all()
