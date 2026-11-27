"""Bolt-3 Dual-Field Attention: Attention Sinks + Local Window, Diversity Softmax Attractors,
and Gershgorin Eviction-to-Delta Residual Associative Memory (M_res).
"""
from __future__ import annotations

import itertools
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import GatedDeltaNet, RMSNorm, RoPE, apply_rope_cos_sin, apply_rope_pos, layer_rope_params, repeat_kv

_RUNNER_UID = itertools.count(1)


class StaticKV:
    """Optional dynamic KV buffer used only if stepping a Softmax teacher replica."""

    def __init__(self, B: int, HKV: int, Tmax: int, HD: int, dev, dt=torch.bfloat16):
        self.k = torch.empty(B, HKV, Tmax, HD, device=dev, dtype=dt)
        self.v = torch.empty_like(self.k)
        self.n = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        L = k_new.shape[2]
        if self.n + L > self.k.shape[2]:
            new_max = max(self.n + L + 1024, self.k.shape[2] * 2)
            nk = torch.empty(
                self.k.shape[0], self.k.shape[1], new_max, self.k.shape[3],
                device=self.k.device, dtype=self.k.dtype,
            )
            nv = torch.empty_like(nk)
            nk[:, :, : self.n] = self.k[:, :, : self.n]
            nv[:, :, : self.n] = self.v[:, :, : self.n]
            self.k, self.v = nk, nv
        self.k[:, :, self.n : self.n + L] = k_new
        self.v[:, :, self.n : self.n + L] = v_new
        self.n += L
        return self.k[:, :, : self.n], self.v[:, :, : self.n]


class Bolt3Attn(nn.Module):
    def __init__(
        self,
        cfg: dict,
        max_T: int = 2048,
        S: int | None = None,
        C: int | None = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        D, H, HKV = int(cfg["dim"]), int(cfg["num_heads"]), int(cfg["num_kv_heads"])
        HD = int(cfg.get("head_dim") or (D // H))
        qkv_bias = bool(cfg.get("qkv_bias", False))
        self.H, self.HKV, self.HD, self.G = H, HKV, HD, H // HKV
        q_scalar = cfg.get("query_pre_attn_scalar")
        self.attn_scale = (float(q_scalar) ** -0.5) if q_scalar is not None else (HD ** -0.5)
        self.S = int(S if S is not None else cfg.get("bolt3_S", 128))
        self.C = int(C if C is not None else cfg.get("bolt3_C", 128))
        self.loc_chunks = max(2, int(cfg.get("bolt3_loc_chunks", 4)))
        self.n_sinks = int(cfg.get("bolt3_n_sinks", 4))
        self.proto_cos = float(cfg.get("bolt3_proto_cos", 0.90))
        self.hh_decay = float(cfg.get("bolt3_hh_decay", 0.92))
        self.sim_k = float(cfg.get("bolt3_sim_k", 0.94))
        self.sim_v = float(cfg.get("bolt3_sim_v", 0.88))
        self.gamma_u = float(cfg.get("bolt3_gamma_u", 0.985))
        s_hh_cfg = cfg.get("bolt3_S_hh")
        s_land_cfg = cfg.get("bolt3_S_land")
        s_proto_cfg = cfg.get("bolt3_S_proto")
        if (
            s_hh_cfg is not None
            and s_land_cfg is not None
            and s_proto_cfg is not None
            and (int(s_hh_cfg) + int(s_land_cfg) + int(s_proto_cfg) == self.S)
        ):
            self.S_hh, self.S_land, self.S_proto = int(s_hh_cfg), int(s_land_cfg), int(s_proto_cfg)
        elif self.S >= 8:
            self.S_proto = max(1, int(round(self.S * 0.125)))
            self.S_land = max(1, int(round(self.S * 0.375)))
            self.S_hh = self.S - self.S_land - self.S_proto
        elif self.S >= 3:
            self.S_hh = max(1, self.S // 2)
            self.S_land = max(1, self.S - self.S_hh - 1)
            self.S_proto = self.S - self.S_hh - self.S_land
        else:
            self.S_hh, self.S_land, self.S_proto = self.S, 0, 0
        cold_layers = {int(i) for i in (cfg.get("bolt3_cold_layers") or ())}
        self.layer_idx = int(layer_idx)
        self.is_cold_armed = self.layer_idx in cold_layers
        self.max_cold_blocks = max(0, int(cfg.get("bolt3_max_cold_blocks", 256)))
        self.cold_topk = max(0, int(cfg.get("bolt3_cold_topk", 0)))
        self.mres_tbptt = max(0, int(cfg.get("bolt3_mres_tbptt", 1)))

        self.packed_q_gate = bool(cfg.get("packed_q_gate", False))
        q_out_dim = (H * HD * 2) if self.packed_q_gate else (H * HD)
        self.q_proj = nn.Linear(D, q_out_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.v_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.o_proj = nn.Linear(H * HD, D, bias=False)
        if cfg.get("qk_norm", False):
            norm_off = float(cfg.get("rms_norm_offset", 0.0))
            self.q_norm = RMSNorm(HD, float(cfg["eps"]), weight_offset=norm_off)
            self.k_norm = RMSNorm(HD, float(cfg["eps"]), weight_offset=norm_off)
        else:
            self.q_norm, self.k_norm = None, None
        r_theta, r_scaling = layer_rope_params(cfg, self.layer_idx)
        self.rope = RoPE(
            HD,
            r_theta,
            r_scaling,
            float(cfg.get("partial_rotary_factor", 1.0)),
        )

        self.kappa = nn.Parameter(torch.ones(HKV))
        self.log_gamma = nn.Parameter(torch.full((HKV,), 5.5))
        self.beta_proj = nn.Linear(D, HKV, bias=True)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, -1.0)
        self.eta_proj = nn.Linear(D, H, bias=True)
        nn.init.zeros_(self.eta_proj.weight)
        nn.init.zeros_(self.eta_proj.bias)
        self.res_norm = RMSNorm(HD, float(cfg["eps"]))
        with torch.no_grad():
            self.res_norm.weight.fill_(0.05)

    def gamma_delta(self) -> torch.Tensor:
        return (0.9995 * torch.sigmoid(self.log_gamma)).clamp(1e-3, 0.9995).float()

    def read_mres(self, q_r: torch.Tensor, m_res: torch.Tensor, ec: torch.Tensor) -> torch.Tensor:
        """Scale-preserving soft-clipped M_res lookup: 0 stays 0, weak recalls stay linear, spikes cap smoothly."""
        b, _, t_q, _ = q_r.shape
        q_rn = F.normalize(q_r.view(b, self.HKV, self.G, t_q, self.HD).float(), dim=-1)
        m_4d = m_res.view(b, self.HKV, self.HD, self.HD).unsqueeze(2)
        ym = torch.matmul(q_rn, m_4d)
        ym_rms = ym.pow(2).mean(dim=-1, keepdim=True).add(self.res_norm.eps).sqrt().clamp_min(1.0)
        rw = self.res_norm.weight.float().view(1, 1, 1, 1, self.HD)
        return ec.float() * ((ym / ym_rms) * rw).reshape(b, self.H, t_q, self.HD)

    def head_qkv_gate(self, x: torch.Tensor):
        b, t, _ = x.shape
        if self.packed_q_gate:
            q, gate = torch.chunk(self.q_proj(x).view(b, t, self.H, self.HD * 2), 2, dim=-1)
            gate = gate.reshape(b, t, self.H * self.HD)
        else:
            q = self.q_proj(x).view(b, t, self.H, self.HD)
            gate = None
        k = self.k_proj(x).view(b, t, self.HKV, self.HD)
        v = self.v_proj(x).view(b, t, self.HKV, self.HD)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), gate

    def head_qkv(self, x: torch.Tensor):
        q, k, v, _ = self.head_qkv_gate(x)
        return q, k, v

    def _consolidate_functional(
        self,
        ak,
        av,
        mass,
        util,
        pm,
        act,
        m_res,
        k_raw,
        k_rope,
        v_old,
        pos_old,
        beta_old,
        inv_freq,
        diag_eye,
        gam,
        n_filled: int = 0,
        tok_valid: torch.Tensor | None = None,
        obs_hit: torch.Tensor | None = None,
    ):
        # Run attractor & M_res consolidation with BF16 autocast disabled so explicit .float()
        # GEMMs match across forward(), prefill(), eager step(), and un-autocast CUDA-Graph step().
        ctx = torch.autocast("cuda", enabled=False) if k_raw.is_cuda else nullcontext()
        with ctx:
            return self._consolidate_functional_impl(
                ak,
                av,
                mass,
                util,
                pm,
                act,
                m_res,
                k_raw,
                k_rope,
                v_old,
                pos_old,
                beta_old,
                inv_freq,
                diag_eye,
                gam,
                n_filled=n_filled,
                tok_valid=tok_valid,
                obs_hit=obs_hit,
            )

    def _consolidate_functional_impl(
        self,
        ak,
        av,
        mass,
        util,
        pm,
        act,
        m_res,
        k_raw,
        k_rope,
        v_old,
        pos_old,
        beta_old,
        inv_freq,
        diag_eye,
        gam,
        n_filled: int = 0,
        tok_valid: torch.Tensor | None = None,
        obs_hit: torch.Tensor | None = None,
    ):
        b, hkv, l_tok, hd = k_raw.shape
        bh = b * hkv
        dt = k_raw.dtype
        kr_f = k_raw.reshape(bh, l_tok, hd)
        v_f = v_old.reshape(bh, l_tok, hd)
        p_f = pos_old.expand(b, hkv, l_tok).reshape(bh, l_tok).float()
        b_f = beta_old.reshape(bh, l_tok)
        ehi_f = (
            obs_hit.reshape(bh, l_tok).float()
            if obs_hit is not None
            else torch.zeros(bh, l_tok, device=k_raw.device, dtype=torch.float32)
        )
        if tok_valid is not None:
            ehi_f = torch.where(tok_valid, ehi_f, torch.full_like(ehi_f, -1e9))

        if ak is None:
            dev = k_raw.device
            ak = torch.zeros(bh, self.S, hd, device=dev, dtype=dt)
            av = torch.zeros(bh, self.S, hd, device=dev, dtype=dt)
            mass = torch.ones(bh, self.S, device=dev, dtype=torch.float32)
            util = torch.zeros(bh, self.S, device=dev, dtype=torch.float32)
            pm = torch.zeros(bh, self.S, device=dev, dtype=torch.float32)
            act = torch.zeros(bh, self.S, device=dev, dtype=torch.bool)
            m_res = torch.zeros(bh, hd, hd, device=dev, dtype=torch.float32)
            n_filled = 0

        s_hh, s_land, s_proto = self.S_hh, self.S_land, self.S_proto
        l0, p0 = s_hh, s_hh + s_land

        with torch.no_grad():
            if s_hh > 0:
                if tok_valid is not None:
                    row_has = tok_valid.any(dim=-1, keepdim=True)
                    u_hh = torch.where(row_has, util[:, :s_hh] * self.hh_decay, util[:, :s_hh])
                else:
                    u_hh = util[:, :s_hh] * self.hh_decay
            else:
                u_hh = util[:, :0]

            if s_land + s_proto > 0:
                ek_n = F.normalize(kr_f.float(), dim=-1)
                lp_n = F.normalize(ak[:, l0:].float(), dim=-1)
                cos_lp = torch.bmm(ek_n, lp_n.transpose(-1, -2)).masked_fill(~act[:, l0:].unsqueeze(1), -1.0)
                if tok_valid is not None:
                    cos_lp = cos_lp.masked_fill(~tok_valid.unsqueeze(-1), -1.0)
                hits_lp = (cos_lp >= self.proto_cos).float().sum(dim=1)
                mass_p = mass[:, p0:] + hits_lp[:, s_land:]
                u_land = util[:, l0:p0] / (1.0 + hits_lp[:, :s_land]).pow(2)

                cc_full = torch.bmm(ek_n, ek_n.transpose(-1, -2))
                if tok_valid is not None:
                    pair_ok = tok_valid.unsqueeze(-1) & tok_valid.unsqueeze(1)
                    cc_full = cc_full.masked_fill(~pair_ok, -1.0)
                    intra_cnt = torch.where(
                        tok_valid, (cc_full >= self.proto_cos).float().sum(dim=-1).clamp_min(1.0), torch.ones_like(ehi_f)
                    )
                else:
                    intra_cnt = (cc_full >= self.proto_cos).float().sum(dim=-1).clamp_min(1.0)

                if s_proto > 0:
                    cos_p = cos_lp[:, :, s_land:]
                    best_p, _ = cos_p.max(dim=-1)
                    is_proto = best_p >= self.proto_cos
                    causal_tri = torch.tril(
                        torch.ones(l_tok, l_tok, device=k_raw.device, dtype=torch.bool), diagonal=-1
                    ).unsqueeze(0)
                    intra_prev_max, _ = cc_full.masked_fill(~causal_tri, -1.0).max(dim=-1)
                    p_cand = (~is_proto) & (intra_prev_max < self.proto_cos) & (intra_cnt >= 2.0)
                    if tok_valid is not None:
                        p_cand = p_cand & tok_valid
                    p_sc = torch.where(p_cand, intra_cnt, torch.full_like(intra_cnt, -1e9))

                    kc_p = min(l_tok, s_proto)
                    tvp, tip = torch.topk(p_sc, k=kc_p, dim=-1, largest=True)
                    pm_cur = torch.where(act[:, p0:], mass_p, torch.zeros_like(mass_p))
                    bvp, bsp = torch.topk(pm_cur, k=kc_p, dim=-1, largest=False)
                    wp = (tvp > bvp) & (tvp > 0.0)
                    slot_p = p0 + bsp
                    wp3 = wp.unsqueeze(-1)
                    ak_p_up = torch.where(
                        wp3,
                        torch.gather(kr_f, 1, tip.unsqueeze(-1).expand(-1, -1, hd)),
                        torch.gather(ak, 1, slot_p.unsqueeze(-1).expand(-1, -1, hd)),
                    )
                    av_p_up = torch.where(
                        wp3,
                        torch.gather(v_f, 1, tip.unsqueeze(-1).expand(-1, -1, hd)),
                        torch.gather(av, 1, slot_p.unsqueeze(-1).expand(-1, -1, hd)),
                    )
                    pm_p_up = torch.where(wp, torch.gather(p_f, 1, tip), torch.gather(pm, 1, slot_p))
                    mass_p_up = torch.where(wp, tvp, torch.gather(mass_p, 1, bsp))
                    act_p_up = torch.gather(act, 1, slot_p) | wp
                    ak = ak.scatter(1, slot_p.unsqueeze(-1).expand(-1, -1, hd), ak_p_up)
                    av = av.scatter(1, slot_p.unsqueeze(-1).expand(-1, -1, hd), av_p_up)
                    pm = pm.scatter(1, slot_p, pm_p_up)
                    mass = mass.scatter(1, slot_p, torch.where(wp, mass_p_up, torch.gather(mass_p, 1, bsp)))
                    # Also persist mass_p increments on non-replaced prototype slots
                    mass_p_all = mass_p.scatter(1, bsp, mass_p_up)
                    mass = torch.cat([mass[:, :p0], mass_p_all], dim=1)
                    act = act.scatter(1, slot_p, act_p_up)

                    lp_n2 = F.normalize(ak[:, p0:].float(), dim=-1)
                    cos_p2 = torch.bmm(ek_n, lp_n2.transpose(-1, -2)).masked_fill(~act[:, p0:].unsqueeze(1), -1.0)
                    best_p, _ = cos_p2.max(dim=-1)
                    is_proto = best_p >= self.proto_cos
                else:
                    best_p = torch.zeros(bh, l_tok, device=k_raw.device, dtype=torch.float32)
                    is_proto = torch.zeros(bh, l_tok, device=k_raw.device, dtype=torch.bool)

                if s_land > 0:
                    nov = (1.0 - best_p.clamp(0.0, 1.0)) / intra_cnt.sqrt()
                    l_cand = ~is_proto
                    if tok_valid is not None:
                        l_cand = l_cand & tok_valid
                    tok_nov = torch.where(l_cand, nov, torch.full_like(ehi_f, -1e9))
                    lk_cur = torch.where(act[:, l0:p0], u_land, torch.full_like(u_land, -1e6))
                    kc_l = min(l_tok, s_land)
                    tv, ti = torch.topk(tok_nov, k=kc_l, dim=-1, largest=True)
                    bv, bsl = torch.topk(lk_cur, k=kc_l, dim=-1, largest=False)
                    wl = (tv > bv) & (tv > 0.02)
                    slot_l = l0 + bsl
                    wl3 = wl.unsqueeze(-1)
                    ak = ak.scatter(
                        1,
                        slot_l.unsqueeze(-1).expand(-1, -1, hd),
                        torch.where(
                            wl3,
                            torch.gather(kr_f, 1, ti.unsqueeze(-1).expand(-1, -1, hd)),
                            torch.gather(ak, 1, slot_l.unsqueeze(-1).expand(-1, -1, hd)),
                        ),
                    )
                    av = av.scatter(
                        1,
                        slot_l.unsqueeze(-1).expand(-1, -1, hd),
                        torch.where(
                            wl3,
                            torch.gather(v_f, 1, ti.unsqueeze(-1).expand(-1, -1, hd)),
                            torch.gather(av, 1, slot_l.unsqueeze(-1).expand(-1, -1, hd)),
                        ),
                    )
                    pm = pm.scatter(1, slot_l, torch.where(wl, torch.gather(p_f, 1, ti), torch.gather(pm, 1, slot_l)))
                    act = act.scatter(1, slot_l, torch.gather(act, 1, slot_l) | wl)
                    u_land = u_land.scatter(1, bsl, torch.where(wl, tv, torch.gather(u_land, 1, bsl)))
                    util = torch.cat([u_hh, u_land, util[:, p0:]], dim=1)
                else:
                    util = torch.cat([u_hh, util[:, l0:]], dim=1)
            elif s_hh > 0:
                util = torch.cat([u_hh, util[:, s_hh:]], dim=1)

            if s_hh > 0:
                kc_h = min(l_tok, s_hh)
                hv, hi = torch.topk(ehi_f, k=kc_h, dim=-1, largest=True)
                hk_cur = torch.where(act[:, :s_hh], util[:, :s_hh], torch.full_like(u_hh, -1e6))
                bv, bsh = torch.topk(hk_cur, k=kc_h, dim=-1, largest=False)
                wh = hv > bv
                wh3 = wh.unsqueeze(-1)
                ak = ak.scatter(
                    1,
                    bsh.unsqueeze(-1).expand(-1, -1, hd),
                    torch.where(
                        wh3,
                        torch.gather(kr_f, 1, hi.unsqueeze(-1).expand(-1, -1, hd)),
                        torch.gather(ak, 1, bsh.unsqueeze(-1).expand(-1, -1, hd)),
                    ),
                )
                av = av.scatter(
                    1,
                    bsh.unsqueeze(-1).expand(-1, -1, hd),
                    torch.where(
                        wh3,
                        torch.gather(v_f, 1, hi.unsqueeze(-1).expand(-1, -1, hd)),
                        torch.gather(av, 1, bsh.unsqueeze(-1).expand(-1, -1, hd)),
                    ),
                )
                pm = pm.scatter(1, bsh, torch.where(wh, torch.gather(p_f, 1, hi), torch.gather(pm, 1, bsh)))
                act = act.scatter(1, bsh, torch.gather(act, 1, bsh) | wh)
                u_hh_up = util[:, :s_hh].scatter(1, bsh, torch.where(wh, hv, torch.gather(util[:, :s_hh], 1, bsh)))
                util = torch.cat([u_hh_up, util[:, s_hh:]], dim=1)

        k_ev = F.normalize(kr_f.float(), dim=-1)
        w_ev = b_f.float() * tok_valid.float() if tok_valid is not None else b_f.float()
        gram = torch.bmm(k_ev, k_ev.transpose(-1, -2)).abs()
        gersh = torch.bmm(gram, w_ev.unsqueeze(-1)).squeeze(-1).clamp_min(1.0)
        beta_safe = (w_ev / gersh).unsqueeze(-1)
        v_pred = torch.bmm(k_ev, m_res)
        err_v = (v_f.float() - v_pred) * beta_safe
        gam_bh = gam.repeat(b).view(bh, 1, 1)
        if tok_valid is not None:
            row_has = tok_valid.any(dim=-1).view(bh, 1, 1)
            gam_eff = torch.where(row_has, gam_bh, torch.ones_like(gam_bh))
        else:
            gam_eff = gam_bh
        m_res = gam_eff * m_res + torch.bmm(k_ev.transpose(1, 2), err_v)
        n_filled_out = int(act.sum(dim=-1).min().item()) if tok_valid is not None else min(self.S, n_filled + l_tok)
        return ak, av, mass, util, pm, act, m_res, n_filled_out

    def _cold_block_bounds(self, p_kp: torch.Tensor, p_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, hkv, c_sz, _ = p_kp.shape
        m_exp = p_mask.unsqueeze(-1).expand(b, hkv, c_sz, self.HD)
        kmin = p_kp.masked_fill(~m_exp, 1e4).amin(dim=2, keepdim=True)
        kmax = p_kp.masked_fill(~m_exp, -1e4).amax(dim=2, keepdim=True)
        return kmin, kmax

    def _push_cold_block(
        self,
        cold_k: torch.Tensor | None,
        cold_v: torch.Tensor | None,
        cold_m: torch.Tensor | None,
        cold_pos: torch.Tensor | None,
        cold_b: torch.Tensor | None,
        cold_kmin: torch.Tensor | None,
        cold_kmax: torch.Tensor | None,
        p_kp: torch.Tensor,
        p_v: torch.Tensor,
        p_mask: torch.Tensor,
        p_pos: torch.Tensor,
        p_b: torch.Tensor,
    ):
        """Append one chunk `(b, HKV, C, HD)` to cold memory; pop oldest block if past `max_cold_blocks`."""
        kb = p_kp.unsqueeze(2)
        vb = p_v.unsqueeze(2)
        mb = p_mask.unsqueeze(2)
        pb = p_pos.unsqueeze(2)
        bb = p_b.unsqueeze(2)
        if self.cold_topk > 0:
            kmin_new, kmax_new = self._cold_block_bounds(p_kp, p_mask)
        else:
            kmin_new = kmax_new = None

        if cold_k is None:
            cold_k, cold_v, cold_m, cold_pos, cold_b = (
                kb.clone(),
                vb.clone(),
                mb.clone(),
                pb.clone(),
                bb.clone(),
            )
            cold_kmin, cold_kmax = kmin_new, kmax_new
        else:
            cold_k = torch.cat([cold_k, kb], dim=2)
            cold_v = torch.cat([cold_v, vb], dim=2)
            cold_m = torch.cat([cold_m, mb], dim=2)
            cold_pos = torch.cat([cold_pos, pb], dim=2)
            cold_b = torch.cat([cold_b, bb], dim=2)
            if kmin_new is not None and cold_kmin is not None:
                cold_kmin = torch.cat([cold_kmin, kmin_new], dim=2)
                cold_kmax = torch.cat([cold_kmax, kmax_new], dim=2)

        popped = None
        if cold_k.shape[2] > self.max_cold_blocks:
            popped = (
                cold_k[:, :, 0],
                cold_v[:, :, 0],
                cold_m[:, :, 0],
                cold_pos[:, :, 0],
                cold_b[:, :, 0],
            )
            cold_k = cold_k[:, :, 1:]
            cold_v = cold_v[:, :, 1:]
            cold_m = cold_m[:, :, 1:]
            cold_pos = cold_pos[:, :, 1:]
            cold_b = cold_b[:, :, 1:]
            if cold_kmin is not None:
                cold_kmin = cold_kmin[:, :, 1:]
                cold_kmax = cold_kmax[:, :, 1:]
        return cold_k, cold_v, cold_m, cold_pos, cold_b, cold_kmin, cold_kmax, popped

    def _attend_with_cold(
        self,
        qg: torch.Tensor,
        sp: list[torch.Tensor],
        vp: list[torch.Tensor],
        cold_k: torch.Tensor | None,
        cold_v: torch.Tensor | None,
        cold_m: torch.Tensor | None,
        cold_kmin: torch.Tensor | None,
        cold_kmax: torch.Tensor | None,
        att_slice: tuple[int, int] | None = None,
        loc_slice: tuple[int, int] | None = None,
        q_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Exact attention over `[sink, ak, loc]` plus optional cold blocks (all or interval top-k)."""
        b, hkv, g_sz, t_q, hd = qg.shape
        scale = self.attn_scale
        q_vm = q_valid_mask.view(b, 1, 1, t_q, 1).float() if q_valid_mask is not None else None
        if cold_k is None or cold_k.shape[2] == 0:
            scores = torch.cat(sp, dim=-1) if len(sp) > 1 else sp[0]
            v_cat = torch.cat(vp, dim=2) if len(vp) > 1 else vp[0]
            probs = F.softmax(scores, dim=-1)
            probs_w = probs * q_vm if q_vm is not None else probs
            att_p = probs_w[..., att_slice[0] : att_slice[1]] if att_slice is not None else None
            loc_p = probs_w[..., loc_slice[0] : loc_slice[1]] if loc_slice is not None else None
            y = torch.matmul(probs.to(v_cat.dtype), v_cat.unsqueeze(2)).reshape(b, self.H, t_q, hd)
            return y, att_p, loc_p, None

        n_cb, c_sz = cold_k.shape[2], cold_k.shape[3]
        if self.cold_topk <= 0 or n_cb <= self.cold_topk:
            kc = cold_k.reshape(b, hkv, n_cb * c_sz, hd)
            vc = cold_v.reshape(b, hkv, n_cb * c_sz, hd)
            mc = cold_m.reshape(b, 1, 1, 1, n_cb * c_sz)
            s_cold = torch.matmul(qg.float(), kc.float().unsqueeze(2).transpose(-1, -2)) * scale
            s_cold = s_cold.masked_fill(~mc, float("-inf"))
            scores = torch.cat(sp + [s_cold], dim=-1)
            v_cat = torch.cat(vp + [vc], dim=2)
            cold_off = sum(t.shape[-1] for t in sp)
            probs = F.softmax(scores, dim=-1)
            probs_w = probs * q_vm if q_vm is not None else probs
            att_p = probs_w[..., att_slice[0] : att_slice[1]] if att_slice is not None else None
            loc_p = probs_w[..., loc_slice[0] : loc_slice[1]] if loc_slice is not None else None
            cold_delta = (
                probs_w[..., cold_off : cold_off + n_cb * c_sz].mean(dim=2).sum(dim=2).view(b, hkv, n_cb, c_sz)
            )
            y = torch.matmul(probs.to(v_cat.dtype), v_cat.unsqueeze(2)).reshape(b, self.H, t_q, hd)
            return y, att_p, loc_p, cold_delta

        # Top-k cold blocks selected per query token via interval bounds [kmin, kmax], merged via logsumexp.
        if cold_kmin is None or cold_kmax is None:
            m_exp = cold_m.unsqueeze(-1).expand(b, hkv, n_cb, c_sz, hd)
            cold_kmin = cold_k.masked_fill(~m_exp, 1e4).amin(dim=3)
            cold_kmax = cold_k.masked_fill(~m_exp, -1e4).amax(dim=3)
        qp5 = qg.float().unsqueeze(4)
        kmax6 = cold_kmax.float().unsqueeze(2).unsqueeze(2)
        kmin6 = cold_kmin.float().unsqueeze(2).unsqueeze(2)
        ub = (qp5.clamp_min(0) * kmax6 + qp5.clamp_max(0) * kmin6).sum(-1).amax(dim=2)  # (b, hkv, t_q, n_cb)
        top_idx = torch.topk(ub, k=self.cold_topk, dim=-1, largest=True).indices  # (b, hkv, t_q, K)
        gk = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, c_sz, hd)
        k_sel = torch.gather(cold_k.unsqueeze(2).expand(-1, -1, t_q, -1, -1, -1), 3, gk).reshape(
            b, hkv, 1, t_q, self.cold_topk * c_sz, hd
        )
        v_sel = torch.gather(cold_v.unsqueeze(2).expand(-1, -1, t_q, -1, -1, -1), 3, gk).reshape(
            b, hkv, 1, t_q, self.cold_topk * c_sz, hd
        )
        m_sel = torch.gather(
            cold_m.expand(b, hkv, -1, -1).unsqueeze(2).expand(-1, -1, t_q, -1, -1),
            3,
            top_idx.unsqueeze(-1).expand(-1, -1, -1, -1, c_sz),
        ).reshape(b, hkv, 1, t_q, self.cold_topk * c_sz)
        s_cold = (qp5 * k_sel.float()).sum(-1) * scale
        s_cold = s_cold.masked_fill(~m_sel, float("-inf"))

        scores_o = torch.cat(sp, dim=-1) if len(sp) > 1 else sp[0]
        v_o = torch.cat(vp, dim=2) if len(vp) > 1 else vp[0]
        lse_o = torch.logsumexp(scores_o, dim=-1, keepdim=True)
        lse_c = torch.logsumexp(s_cold, dim=-1, keepdim=True)
        lse_all = torch.logaddexp(lse_o, lse_c)
        w_o = torch.exp(lse_o - lse_all)
        w_c = torch.exp(lse_c - lse_all)

        p_o = F.softmax(scores_o, dim=-1)
        p_c = torch.where(m_sel, F.softmax(s_cold, dim=-1), 0.0)
        y_o = torch.matmul(p_o.to(v_o.dtype), v_o.unsqueeze(2)).float()
        y_c = (p_c.to(v_sel.dtype).unsqueeze(-1) * v_sel).sum(dim=4).float()
        y = (w_o * y_o + w_c * y_c).to(v_o.dtype).reshape(b, self.H, t_q, hd)
        p_ow = p_o * (w_o * q_vm if q_vm is not None else w_o)
        p_cw = (p_c * (w_c * q_vm if q_vm is not None else w_c)).mean(dim=2)
        att_p = p_ow[..., att_slice[0] : att_slice[1]] if att_slice is not None else None
        loc_p = p_ow[..., loc_slice[0] : loc_slice[1]] if loc_slice is not None else None
        cold_delta = torch.zeros(b, hkv, n_cb, c_sz, device=qg.device, dtype=torch.float32)
        sc_idx = top_idx.unsqueeze(-1).expand(-1, -1, -1, -1, c_sz).reshape(b, hkv, t_q * self.cold_topk, c_sz)
        cold_delta.scatter_add_(2, sc_idx, p_cw.reshape(b, hkv, t_q * self.cold_topk, c_sz))
        return y, att_p, loc_p, cold_delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, T, _ = x.shape
        q_raw, k_raw, v_all_in, gate_all = self.head_qkv_gate(x)
        q_rope, k_rope = self.rope(q_raw), self.rope(k_raw)
        xp_f = x if (x.dtype == self.beta_proj.weight.dtype or torch.is_autocast_enabled("cuda")) else x.to(self.beta_proj.weight.dtype)
        beta_all = torch.sigmoid(self.beta_proj(xp_f)).transpose(1, 2)
        eta_all = torch.tanh(self.eta_proj(xp_f)).transpose(1, 2).unsqueeze(-1)
        gam = self.gamma_delta()
        inv_freq = self.rope.inv_freq.to(x.device)
        diag_eye = torch.eye(self.S, device=x.device, dtype=torch.bool).unsqueeze(0)
        scale = self.attn_scale

        sink_k = sink_v = ak = av = mass = util = pm = act = m_res = None
        loc_chunks: list[list[torch.Tensor]] = []
        cold_k = cold_v = cold_m = cold_pos = cold_b = cold_hit = cold_kmin = cold_kmax = None
        n_filled = 0
        ys = []
        cap_aux = bool(getattr(self, "_capture_mres_aux", False))
        aux_att_parts: list[torch.Tensor] = []
        aux_mr_parts: list[torch.Tensor] = []

        for s in range(0, T, self.C):
            e = min(T, s + self.C)
            tc = e - s
            qc_r, qc_p = q_raw[:, :, s:e], q_rope[:, :, s:e]
            kc_r, kc_p = k_raw[:, :, s:e], k_rope[:, :, s:e]
            vc, bc = v_all_in[:, :, s:e], beta_all[:, :, s:e]
            ec = eta_all[:, :, s:e]
            pos_c = torch.arange(s, e, device=x.device, dtype=torch.float32)[None, None, :]

            if loc_chunks:
                loc_k = torch.cat([ch[1] for ch in loc_chunks] + [kc_p], dim=2)
                loc_v = torch.cat([ch[2] for ch in loc_chunks] + [vc], dim=2)
                prev_len = sum(ch[1].shape[2] for ch in loc_chunks)
            else:
                loc_k, loc_v, prev_len = kc_p, vc, 0

            qg = qc_p.view(b, self.HKV, self.G, tc, self.HD)
            s_loc = torch.matmul(qg.float(), loc_k.float().unsqueeze(2).transpose(-1, -2)) * scale
            qi = torch.arange(tc, device=x.device)[:, None]
            kj = torch.arange(loc_k.shape[2], device=x.device)[None, :]
            s_loc = s_loc.masked_fill(~(kj <= (prev_len + qi)).view(1, 1, 1, tc, -1), float("-inf"))

            sp, vp = [], []
            if sink_k is not None:
                sp.append(torch.matmul(qg.float(), sink_k.float().unsqueeze(2).transpose(-1, -2)) * scale)
                vp.append(sink_v)
            if ak is not None:
                ak_eff = apply_rope_pos(
                    ak.view(b, self.HKV, self.S, self.HD), pm.view(b, self.HKV, self.S), inv_freq
                )
                s_att = torch.matmul(qg.float(), ak_eff.float().unsqueeze(2).transpose(-1, -2)) * scale
                kap = self.kappa.float().view(1, self.HKV, 1, 1, 1)
                m_bias = kap * torch.log(mass.view(b, self.HKV, 1, 1, self.S).clamp_min(1.0))
                act_m = act.view(b, self.HKV, 1, 1, self.S)
                s_att = (s_att + m_bias).masked_fill(~act_m, float("-inf"))
                sp.append(s_att)
                vp.append(av.view(b, self.HKV, self.S, self.HD))

            sink_len = sink_k.shape[2] if sink_k is not None else 0
            loc_off = sink_len + (self.S if ak is not None else 0)
            att_slice = (sink_len, sink_len + self.S) if ak is not None else None
            loc_slice = (loc_off, loc_off + loc_k.shape[2])
            sp.append(s_loc)
            vp.append(loc_v)
            yc, att_p, loc_p, cold_delta = self._attend_with_cold(
                qg,
                sp,
                vp,
                cold_k,
                cold_v,
                cold_m,
                cold_kmin,
                cold_kmax,
                att_slice=att_slice,
                loc_slice=loc_slice,
            )

            with torch.no_grad():
                p_loc_sum = loc_p.mean(dim=2).sum(dim=2)
                off_c = 0
                for ch in loc_chunks:
                    l_ch = ch[1].shape[2]
                    ch[5].add_(p_loc_sum[:, :, off_c : off_c + l_ch])
                    off_c += l_ch
                cur_hit = p_loc_sum[:, :, off_c : off_c + tc].clone()
                if cold_delta is not None and cold_hit is not None:
                    cold_hit.add_(cold_delta)
                if ak is not None and att_p is not None and self.S_hh > 0:
                    att_hit = (att_p.mean(dim=2).sum(dim=2) / self.C).reshape(b * self.HKV, self.S)
                    u_hh_m = torch.maximum(util[:, : self.S_hh], att_hit[:, : self.S_hh])
                    util = torch.cat([u_hh_m, util[:, self.S_hh :]], dim=1)

            if m_res is not None:
                ym_c = self.read_mres(qc_r, m_res, ec).to(yc.dtype)
                if cap_aux:
                    aux_att_parts.append(yc.detach())
                    aux_mr_parts.append(ym_c)
                yc = yc + ym_c
            else:
                if cap_aux:
                    aux_att_parts.append(yc.detach())
                    aux_mr_parts.append(torch.zeros_like(yc))
                if e == T:
                    yc = yc + 0.0 * (
                        ec.sum() + self.res_norm.weight.sum() + self.kappa.sum() + gam.sum() + bc.sum()
                    )

            ys.append(yc)

            if tc == self.C and e < T:
                loc_chunks.append([kc_r, kc_p, vc, pos_c, bc, cur_hit])
                if len(loc_chunks) >= self.loc_chunks:
                    p_kr, p_kp, p_v, p_pos, p_b, p_hit = loc_chunks.pop(0)
                    c_mask = torch.ones(b, 1, self.C, device=x.device, dtype=torch.bool)
                    if sink_k is None:
                        ns = min(self.n_sinks, p_kp.shape[2])
                        sink_k, sink_v = p_kp[:, :, :ns], p_v[:, :, :ns]
                        c_mask[:, :, :ns] = False
                        p_kr_t, p_kp_t, p_v_t, p_pos_t, p_b_t, p_hit_t = (
                            p_kr[:, :, ns:],
                            p_kp[:, :, ns:],
                            p_v[:, :, ns:],
                            p_pos[:, :, ns:],
                            p_b[:, :, ns:],
                            p_hit[:, :, ns:],
                        )
                    else:
                        p_kr_t, p_kp_t, p_v_t, p_pos_t, p_b_t, p_hit_t = p_kr, p_kp, p_v, p_pos, p_b, p_hit

                    if self.is_cold_armed and self.max_cold_blocks > 0:
                        cold_k, cold_v, cold_m, cold_pos, cold_b, cold_kmin, cold_kmax, popped = self._push_cold_block(
                            cold_k,
                            cold_v,
                            cold_m,
                            cold_pos,
                            cold_b,
                            cold_kmin,
                            cold_kmax,
                            p_kp,
                            p_v,
                            c_mask,
                            p_pos.expand(b, 1, self.C),
                            p_b,
                        )
                        hb = p_hit.unsqueeze(2)
                        cold_hit = hb.clone() if cold_hit is None else torch.cat([cold_hit, hb], dim=2)
                        if popped is not None:
                            ok_p, ok_v, ok_m, ok_pos, ok_b = popped
                            ok_h = cold_hit[:, :, 0]
                            cold_hit = cold_hit[:, :, 1:]
                            ok_kr = apply_rope_pos(ok_p, -ok_pos.expand(b, self.HKV, self.C), inv_freq)
                            tok_v_in = (
                                None
                                if bool(ok_m.all())
                                else ok_m.expand(b, self.HKV, -1).reshape(b * self.HKV, -1)
                            )
                            ak_d = ak.detach() if ak is not None else None
                            av_d = av.detach() if av is not None else None
                            pm_d = pm.detach() if pm is not None else None
                            detach_mr = self.mres_tbptt > 0 and (((s // self.C) + 1) % self.mres_tbptt == 0)
                            mr_d = m_res.detach() if (m_res is not None and detach_mr) else m_res
                            ak, av, mass, util, pm, act, m_res, n_filled = self._consolidate_functional(
                                ak_d,
                                av_d,
                                mass,
                                util,
                                pm_d,
                                act,
                                mr_d,
                                ok_kr,
                                ok_p,
                                ok_v,
                                ok_pos,
                                ok_b,
                                inv_freq,
                                diag_eye,
                                gam,
                                n_filled=n_filled,
                                tok_valid=tok_v_in,
                                obs_hit=ok_h,
                            )
                    else:
                        ak_d = ak.detach() if ak is not None else None
                        av_d = av.detach() if av is not None else None
                        pm_d = pm.detach() if pm is not None else None
                        detach_mr = self.mres_tbptt > 0 and (((s // self.C) + 1) % self.mres_tbptt == 0)
                        mr_d = m_res.detach() if (m_res is not None and detach_mr) else m_res
                        ak, av, mass, util, pm, act, m_res, n_filled = self._consolidate_functional(
                            ak_d,
                            av_d,
                            mass,
                            util,
                            pm_d,
                            act,
                            mr_d,
                            p_kr_t,
                            p_kp_t,
                            p_v_t,
                            p_pos_t,
                            p_b_t,
                            inv_freq,
                            diag_eye,
                            gam,
                            n_filled=n_filled,
                            obs_hit=p_hit_t,
                        )

        if cap_aux:
            self._mres_aux = (
                torch.cat(aux_att_parts, dim=2).transpose(1, 2).reshape(b, T, self.H * self.HD),
                torch.cat(aux_mr_parts, dim=2).transpose(1, 2).reshape(b, T, self.H * self.HD),
            )
        out = torch.cat(ys, dim=2).transpose(1, 2).reshape(b, T, self.H * self.HD)
        if gate_all is not None:
            out = out * torch.sigmoid(gate_all)
        return self.o_proj(out)


def _steady_block_fast(
    ly: Bolt3Attn,
    norm1,
    norm2,
    mlp,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_n: torch.Tensor,
    slot_buf_t: torch.Tensor,
    slot_loc_t: torch.Tensor,
    zero_mask: torch.Tensor,
    buf_k: torch.Tensor,
    buf_v: torch.Tensor,
    mask_buf: torch.Tensor,
    util_acc: torch.Tensor,
    loc_pos: torch.Tensor,
    loc_b: torch.Tensor,
    m_res: torch.Tensor,
    post_attn_norm=None,
    post_mlp_norm=None,
) -> torch.Tensor:
    """Zero-cat single-token decode step across unified `[sink | ak_eff | loc_ring | cold]` buffers."""
    b = x.shape[0]
    h = norm1(x)
    q_r, k_r, v_n, gate = ly.head_qkv_gate(h)
    rot_dim = cos.shape[-1]
    q_p = apply_rope_cos_sin(q_r, cos, sin, rot_dim)
    k_p = apply_rope_cos_sin(k_r, cos, sin, rot_dim)
    beta_n = torch.sigmoid(ly.beta_proj(h)).transpose(1, 2)
    eta_n = torch.tanh(ly.eta_proj(h)).transpose(1, 2).unsqueeze(-1)

    buf_k.index_copy_(2, slot_buf_t, k_p.to(buf_k.dtype))
    buf_v.index_copy_(2, slot_buf_t, v_n.to(buf_v.dtype))
    mask_buf.index_copy_(3, slot_buf_t, zero_mask)
    loc_pos.index_copy_(2, slot_loc_t, pos_n)
    loc_b.index_copy_(2, slot_loc_t, beta_n.to(loc_b.dtype))

    qg = q_p.view(b, ly.HKV, ly.G, 1, ly.HD)
    n_tot = buf_k.shape[2]
    if x.is_cuda and n_tot > 512:
        scores = torch.matmul(qg, buf_k.unsqueeze(2).transpose(-1, -2)).float() * ly.attn_scale
    else:
        scores = torch.matmul(qg.float(), buf_k.float().unsqueeze(2).transpose(-1, -2)) * ly.attn_scale
    scores = scores + mask_buf.view(b, ly.HKV, 1, 1, n_tot)
    probs = F.softmax(scores, dim=-1)
    util_acc.add_(probs[..., ly.n_sinks : n_tot].mean(dim=2).reshape(b * ly.HKV, n_tot - ly.n_sinks))
    y = torch.matmul(probs.to(buf_v.dtype), buf_v.unsqueeze(2)).reshape(b, ly.H, 1, ly.HD)
    y = y + ly.read_mres(q_r, m_res, eta_n).to(y.dtype)

    out = y.transpose(1, 2).reshape(b, 1, ly.H * ly.HD)
    if gate is not None:
        out = out * torch.sigmoid(gate)
    attn_delta = ly.o_proj(out)
    if post_attn_norm is not None:
        attn_delta = post_attn_norm(attn_delta)
    x = x + attn_delta
    mlp_delta = mlp(norm2(x))
    if post_mlp_norm is not None:
        mlp_delta = post_mlp_norm(mlp_delta)
    return x + mlp_delta


class _Bolt3GraphRunner:
    """CUDA Graph runner for steady-state (`cur_len < C`) single-token Bolt-3 / Hybrid GDN decode."""

    def __init__(
        self,
        model,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        cold_caps: tuple[int, ...] | None = None,
    ):
        self.uid = next(_RUNNER_UID)
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.cold_caps = cold_caps or tuple(0 for _ in model.blocks)
        self.L = len(model.blocks)
        b3_layers = [blk.attn for blk in model.blocks if isinstance(blk.attn, Bolt3Attn)]
        ly0 = b3_layers[0] if b3_layers else None
        self.C = ly0.C if ly0 is not None else 128
        self.loc_span = (ly0.loc_chunks * ly0.C) if ly0 is not None else self.C
        self.W1 = (ly0.n_sinks + ly0.S + self.loc_span) if ly0 is not None else self.C

        self.states: list[torch.Tensor] = []
        self.conv_states: list[torch.Tensor | None] = []
        self.buf_k: list[torch.Tensor | None] = []
        self.buf_v: list[torch.Tensor | None] = []
        self.mask_buf: list[torch.Tensor | None] = []
        self.util_acc: list[torch.Tensor | None] = []
        self.loc_pos: list[torch.Tensor | None] = []
        self.loc_b: list[torch.Tensor | None] = []

        for li, blk in enumerate(model.blocks):
            ly = blk.attn
            if isinstance(ly, Bolt3Attn):
                c_cap = self.cold_caps[li]
                l_span = ly.loc_chunks * ly.C
                n_tot = ly.n_sinks + ly.S + l_span + c_cap * ly.C
                self.states.append(torch.zeros(batch_size, ly.HKV, ly.HD, ly.HD, device=device, dtype=torch.float32))
                self.conv_states.append(None)
                self.buf_k.append(torch.zeros(batch_size, ly.HKV, n_tot, ly.HD, device=device, dtype=dtype))
                self.buf_v.append(torch.zeros(batch_size, ly.HKV, n_tot, ly.HD, device=device, dtype=dtype))
                self.mask_buf.append(torch.zeros(batch_size, ly.HKV, 1, n_tot, device=device, dtype=torch.float32))
                self.util_acc.append(torch.zeros(batch_size * ly.HKV, n_tot - ly.n_sinks, device=device, dtype=torch.float32))
                self.loc_pos.append(torch.zeros(batch_size, 1, l_span, device=device, dtype=torch.float32))
                self.loc_b.append(torch.zeros(batch_size, ly.HKV, l_span, device=device, dtype=dtype))
            elif isinstance(ly, GatedDeltaNet):
                self.states.append(
                    torch.zeros(batch_size, ly.num_v_heads, ly.head_k_dim, ly.head_v_dim, device=device, dtype=torch.float32)
                )
                self.conv_states.append(
                    torch.zeros(batch_size, ly.conv_dim, ly.conv_kernel_size - 1, device=device, dtype=dtype)
                )
                for lst in (self.buf_k, self.buf_v, self.mask_buf, self.util_acc, self.loc_pos, self.loc_b):
                    lst.append(None)
            else:
                raise TypeError(f"Unsupported layer type inside _Bolt3GraphRunner: {type(ly).__name__}")

        self.g_tok = torch.zeros(batch_size, 1, device=device, dtype=torch.long)
        self.g_pos = torch.zeros(batch_size, 1, 1, device=device, dtype=torch.float32)
        self.g_slot_buf = torch.zeros(1, device=device, dtype=torch.long)
        self.g_slot_loc = torch.zeros(1, device=device, dtype=torch.long)
        hkv_z = ly0.HKV if ly0 is not None else 1
        self.g_zero_mask = torch.zeros(batch_size, hkv_z, 1, 1, device=device, dtype=torch.float32)
        self.g_out_h = torch.zeros(batch_size, 1, int(model.config["dim"]), device=device, dtype=dtype)
        self.slot_idxs = [torch.tensor([i], device=device, dtype=torch.long) for i in range(max(self.W1, self.loc_span))]
        self.rope_inv_freqs: list[torch.Tensor] = []
        self.layer_rope_gid: list[int | None] = []
        rope_sig_to_gid: dict[tuple, int] = {}
        for blk in model.blocks:
            ly = blk.attn
            if isinstance(ly, Bolt3Attn):
                sig = (int(ly.rope.rotary_dim), tuple(round(float(v), 12) for v in ly.rope.inv_freq.detach().cpu().tolist()))
                gid = rope_sig_to_gid.get(sig)
                if gid is None:
                    gid = len(self.rope_inv_freqs)
                    rope_sig_to_gid[sig] = gid
                    self.rope_inv_freqs.append(ly.rope.inv_freq.to(device))
                self.layer_rope_gid.append(gid)
            else:
                self.layer_rope_gid.append(None)
        use_bf16_auto = dtype == torch.bfloat16

        def _forward_steady():
            with torch.autocast("cuda", torch.bfloat16, enabled=use_bf16_auto):
                x = self.model.embed_tokens(self.g_tok)
                rope_cs_list = []
                for inv_f in self.rope_inv_freqs:
                    freqs = self.g_pos.unsqueeze(-1) * inv_f.view(1, 1, 1, -1)
                    emb = torch.cat((freqs, freqs), dim=-1)
                    rope_cs_list.append((emb.cos().to(x.dtype), emb.sin().to(x.dtype)))
                for li, blk in enumerate(self.model.blocks):
                    ly = blk.attn
                    p_an = getattr(blk, "post_attn_norm", None)
                    p_mn = getattr(blk, "post_mlp_norm", None)
                    if isinstance(ly, Bolt3Attn):
                        cos_l, sin_l = rope_cs_list[self.layer_rope_gid[li]]
                        x = _steady_block_fast(
                            ly,
                            blk.norm1,
                            blk.norm2,
                            blk.mlp,
                            x,
                            cos_l,
                            sin_l,
                            self.g_pos,
                            self.g_slot_buf,
                            self.g_slot_loc,
                            self.g_zero_mask,
                            self.buf_k[li],
                            self.buf_v[li],
                            self.mask_buf[li],
                            self.util_acc[li],
                            self.loc_pos[li],
                            self.loc_b[li],
                            self.states[li],
                            post_attn_norm=p_an,
                            post_mlp_norm=p_mn,
                        )
                    else:
                        a_d = ly.step_graph(blk.norm1(x), self.conv_states[li], self.states[li])
                        if p_an is not None:
                            a_d = p_an(a_d)
                        x = x + a_d
                        m_d = blk.mlp(blk.norm2(x))
                        if p_mn is not None:
                            m_d = p_mn(m_d)
                        x = x + m_d
                return self.model.norm(x)

        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream(device=device))
        with torch.cuda.stream(s):
            for _ in range(2):
                _ = _forward_steady()
        torch.cuda.current_stream(device=device).wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.g_out_h.copy_(_forward_steady())

    def close(self):
        self.graph = None
        for lst in (
            self.states,
            self.conv_states,
            self.buf_k,
            self.buf_v,
            self.mask_buf,
            self.util_acc,
            self.loc_pos,
            self.loc_b,
            self.slot_idxs,
            self.rope_inv_freqs,
            self.layer_rope_gid,
        ):
            lst.clear()
        self.model = None

    @torch.no_grad()
    def replay(
        self, tokens: torch.Tensor, pos_start: int, pos_ids: torch.Tensor | None, slot_buf: int, slot_loc: int
    ) -> torch.Tensor:
        self.g_tok.copy_(tokens)
        if pos_ids is None:
            self.g_pos.fill_(float(pos_start))
        else:
            self.g_pos.copy_(pos_ids.view(self.batch_size, 1, 1))
        self.g_slot_buf.copy_(self.slot_idxs[slot_buf])
        self.g_slot_loc.copy_(self.slot_idxs[slot_loc])
        self.graph.replay()
        return self.g_out_h.clone()


class Bolt3Decoder:
    """Constant-memory recurrent chunk/token decoder for Bolt-3 models."""

    def __init__(self, model, max_cache_pos: int = 65536, **kwargs):
        self.model = model.eval()
        self.max_T = int(model.config["max_T"])
        self.max_cache_pos = max_cache_pos
        self._diag_eye_cache: dict[torch.device, torch.Tensor] = {}
        self.C = int(model.config.get("bolt3_C", 128))
        self.use_cuda_graph = bool(kwargs.get("use_cuda_graph", True))
        self._graph_runners: dict[tuple, _Bolt3GraphRunner] = {}
        self._gam_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._slot_t_cache: dict[tuple, list[torch.Tensor]] = {}
        self._zero_m_cache: dict[tuple, torch.Tensor] = {}
        self._b3_indices = [li for li, blk in enumerate(model.blocks) if isinstance(blk.attn, Bolt3Attn)]
        self._all_fast_step = all(isinstance(blk.attn, (Bolt3Attn, GatedDeltaNet)) for blk in model.blocks)
        self._all_bolt3 = self._all_fast_step and (len(self._b3_indices) == len(model.blocks))
        self._first_bolt3 = model.blocks[self._b3_indices[0]].attn if self._b3_indices else None
        if self._first_bolt3 is not None:
            inv0 = self._first_bolt3.rope.inv_freq
            self._shared_rope = self._all_fast_step and all(
                model.blocks[li].attn.rope.rotary_dim == self._first_bolt3.rope.rotary_dim
                and torch.equal(model.blocks[li].attn.rope.inv_freq, inv0)
                for li in self._b3_indices
            )
        else:
            self._shared_rope = self._all_fast_step

    @staticmethod
    def state_bytes(model) -> tuple[int, int]:
        flat, per_tok = 0, 0
        for blk in model.blocks:
            ly = blk.attn
            if isinstance(ly, Bolt3Attn):
                flat += (
                    ly.HKV * ly.HD * ly.HD * 4
                    + 2 * ly.S * ly.HKV * ly.HD * 2
                    + 3 * ly.S * ly.HKV * 4
                    + 2 * ly.loc_chunks * ly.C * ly.HKV * ly.HD * 2
                    + 2 * ly.n_sinks * ly.HKV * ly.HD * 2
                )
                if ly.is_cold_armed and ly.max_cold_blocks > 0:
                    per_tok += 2 * ly.HKV * ly.HD * 2
            elif isinstance(ly, GatedDeltaNet):
                flat += ly.num_v_heads * ly.head_k_dim * ly.head_v_dim * 4 + ly.conv_dim * (ly.conv_kernel_size - 1) * 2
            else:
                per_tok += 2 * ly.HKV * ly.HD * 2
        return flat, per_tok

    @staticmethod
    def blank_state(
        model,
        batch_size: int,
        device,
        max_kv_len: int = 4096,
        state_dtype=torch.float32,
        **kwargs,
    ):
        states, kv_caches = [None] * len(model.blocks), [None] * len(model.blocks)
        emb_dt = model.embed_tokens.weight.dtype
        dt = (
            emb_dt
            if emb_dt in (torch.bfloat16, torch.float16, torch.float32)
            else (torch.bfloat16 if str(device).startswith("cuda") else torch.float32)
        )
        for li, blk in enumerate(model.blocks):
            ly = blk.attn
            if isinstance(ly, Bolt3Attn):
                states[li] = torch.zeros(batch_size, ly.HKV, ly.HD, ly.HD, device=device, dtype=state_dtype)
                kv_caches[li] = {
                    "ak": None,
                    "ak_eff": None,
                    "att_bias": None,
                    "av": None,
                    "mass": None,
                    "util": None,
                    "util_acc": None,
                    "pm": None,
                    "act": None,
                    "n_filled": 0,
                    "sink_k": None,
                    "sink_v": None,
                    "sink_valid": None,
                    "sink_filled": None,
                    "loc_k": None,
                    "loc_v": None,
                    "loc_pos": None,
                    "loc_b": None,
                    "loc_valid": None,
                    "loc_slot_pad": [False] * ly.loc_chunks,
                    "cur_slot": 0,
                    "n_loc_filled": 0,
                    "cur_len": 0,
                    "cold_k_blk": None,
                    "cold_v_blk": None,
                    "cold_mask": None,
                    "cold_pos": None,
                    "cold_b": None,
                    "cold_hit": None,
                    "cold_kmin": None,
                    "cold_kmax": None,
                    "cold_cap_blks": 0,
                    "buf_k": None,
                    "buf_v": None,
                    "mask_buf": None,
                    "_synced_ids": None,
                    "_runner_id": None,
                }
            elif isinstance(ly, GatedDeltaNet):
                states[li] = torch.zeros(
                    batch_size, ly.num_v_heads, ly.head_k_dim, ly.head_v_dim, device=device, dtype=state_dtype
                )
                kv_caches[li] = {
                    "conv_state": torch.zeros(
                        batch_size, ly.conv_dim, ly.conv_kernel_size - 1, device=device, dtype=dt
                    ),
                    "_runner_id": None,
                }
            else:
                kv_caches[li] = StaticKV(batch_size, ly.HKV, max_kv_len, ly.HD, device, dt=dt)
        return states, kv_caches

    @staticmethod
    def reset_state(states, kv_caches):
        for s in states:
            if s is not None:
                s.zero_()
        for c in kv_caches:
            if isinstance(c, dict):
                if "conv_state" in c:
                    c["conv_state"].zero_()
                    c["_runner_id"] = None
                else:
                    for k in list(c.keys()):
                        if k in ("n_filled", "cur_len", "cur_slot", "n_loc_filled", "cold_cap_blks"):
                            c[k] = 0
                        elif k == "loc_slot_pad":
                            c[k] = [False] * len(c[k])
                        else:
                            c[k] = None
            elif isinstance(c, StaticKV):
                c.n = 0

    def close(self):
        runners = getattr(self, "_graph_runners", None)
        if runners:
            for r in list(runners.values()):
                try:
                    r.close()
                except Exception:
                    pass
            runners.clear()
        for attr in ("_diag_eye_cache", "_gam_cache", "_slot_t_cache", "_zero_m_cache"):
            d = getattr(self, attr, None)
            if isinstance(d, dict):
                d.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        history_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = logits.float()
        if repetition_penalty != 1.0 and history_ids is not None and history_ids.numel() > 0:
            for r in range(scores.shape[0]):
                seen = torch.unique(history_ids[r])
                sel = scores[r, seen]
                scores[r, seen] = torch.where(sel < 0, sel * repetition_penalty, sel / repetition_penalty)
        if temperature <= 1e-5:
            return scores.argmax(-1, keepdim=True)
        scaled = scores / temperature
        if top_k is not None and top_k > 0:
            k = min(int(top_k), scaled.size(-1))
            kth = torch.topk(scaled, k, dim=-1).values[..., -1:]
            scaled = scaled.masked_fill(scaled < kth, float("-inf"))
        probs = F.softmax(scaled, dim=-1)
        if top_p < 1.0:
            sp, si = torch.sort(probs, descending=True, dim=-1)
            cp = torch.cumsum(sp, dim=-1)
            sp[cp - sp > top_p] = 0.0
            sp.div_(sp.sum(dim=-1, keepdim=True).clamp_min(1e-12))
            return si.gather(-1, torch.multinomial(sp, 1))
        return torch.multinomial(probs, 1)

    @staticmethod
    def _refresh_attractor_cache(sc: dict, ly: Bolt3Attn, b: int, inv_freq: torch.Tensor):
        sc["ak_eff"] = apply_rope_pos(
            sc["ak"].view(b, ly.HKV, ly.S, ly.HD), sc["pm"].view(b, ly.HKV, ly.S), inv_freq
        )
        kap = ly.kappa.float().view(1, ly.HKV, 1, 1)
        mb = kap * torch.log(sc["mass"].view(b, ly.HKV, 1, ly.S).clamp_min(1.0))
        am = sc["act"].view(b, ly.HKV, 1, ly.S)
        sc["att_bias"] = mb.masked_fill(~am, float("-inf"))

    def _get_gam_scale(self, ly: Bolt3Attn, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = id(ly)
        cached = self._gam_cache.get(key)
        if cached is None or cached[0].device != device:
            gam = ly.gamma_delta().to(device)
            res_scale = (1.0 - gam).view(1, ly.HKV, 1, 1, 1) / (ly.HD ** 0.5)
            cached = (gam, res_scale)
            self._gam_cache[key] = cached
        return cached

    def _get_slot_idx(self, ly: Bolt3Attn, idx: int, device: torch.device) -> torch.Tensor:
        l_span = ly.loc_chunks * ly.C
        key = (ly.n_sinks + ly.S + l_span, device)
        arr = self._slot_t_cache.get(key)
        if arr is None:
            n_tot = max(key[0], l_span)
            arr = [torch.tensor([i], device=device, dtype=torch.long) for i in range(n_tot)]
            self._slot_t_cache[key] = arr
        return arr[idx]

    def _get_zero_mask(self, b: int, hkv: int, device: torch.device) -> torch.Tensor:
        key = (b, hkv, device)
        zm = self._zero_m_cache.get(key)
        if zm is None:
            zm = torch.zeros(b, hkv, 1, 1, device=device, dtype=torch.float32)
            self._zero_m_cache[key] = zm
        return zm

    @staticmethod
    def _cold_cap_for(ly: Bolt3Attn, sc: dict | None) -> int:
        if not (ly.is_cold_armed and ly.max_cold_blocks > 0 and ly.cold_topk == 0):
            return 0
        ck = sc.get("cold_k_blk") if isinstance(sc, dict) else None
        n_cb = int(ck.shape[2]) if ck is not None else 0
        if n_cb <= 0:
            return 0
        cap = 16 if n_cb <= 16 else (1 << (n_cb - 1).bit_length())
        return min(ly.max_cold_blocks, cap)

    def _sync_unified_bufs(self, sc: dict, ly: Bolt3Attn, b: int, device: torch.device, dt: torch.dtype):
        cold_cap_blks = self._cold_cap_for(ly, sc)
        sig = (
            id(sc.get("buf_k")),
            id(sc.get("sink_k")),
            id(sc.get("ak")),
            cold_cap_blks,
            id(sc.get("cold_k_blk")),
        )
        if sc.get("buf_k") is not None and sc.get("_synced_ids") == sig:
            return
        w0 = ly.n_sinks + ly.S
        l_span = ly.loc_chunks * ly.C
        w1 = w0 + l_span
        n_tot = w1 + cold_cap_blks * ly.C
        inv_freq = ly.rope.inv_freq.to(device)
        buf_k = sc.get("buf_k")
        if buf_k is None or buf_k.shape != (b, ly.HKV, n_tot, ly.HD) or buf_k.dtype != dt or buf_k.device != device:
            buf_k = torch.zeros(b, ly.HKV, n_tot, ly.HD, device=device, dtype=dt)
            buf_v = torch.zeros_like(buf_k)
            mask_buf = torch.full((b, ly.HKV, 1, n_tot), float("-inf"), device=device, dtype=torch.float32)
        else:
            buf_v = sc["buf_v"]
            mask_buf = sc["mask_buf"]
            mask_buf.fill_(float("-inf"))

        if sc["sink_k"] is not None:
            ns = sc["sink_k"].shape[2]
            buf_k[:, :, :ns] = sc["sink_k"].to(device=device, dtype=dt)
            buf_v[:, :, :ns] = sc["sink_v"].to(device=device, dtype=dt)
            if sc.get("sink_valid") is not None:
                sv = sc["sink_valid"].view(b, 1, 1, ns).expand(b, ly.HKV, 1, ns)
                mask_buf[:, :, :, :ns] = torch.where(sv, 0.0, float("-inf"))
            else:
                mask_buf[:, :, :, :ns] = 0.0

        if sc["ak"] is not None:
            if sc.get("ak_eff") is None or sc.get("att_bias") is None:
                self._refresh_attractor_cache(sc, ly, b, inv_freq)
            buf_k[:, :, ly.n_sinks : w0] = sc["ak_eff"].to(device=device, dtype=dt)
            buf_v[:, :, ly.n_sinks : w0] = sc["av"].view(b, ly.HKV, ly.S, ly.HD).to(device=device, dtype=dt)
            mask_buf[:, :, :, ly.n_sinks : w0] = sc["att_bias"].view(b, ly.HKV, 1, ly.S)

        acc_len = n_tot - ly.n_sinks
        old_acc = sc.get("util_acc")
        if old_acc is None or old_acc.shape != (b * ly.HKV, acc_len) or old_acc.device != device:
            new_acc = torch.zeros(b * ly.HKV, acc_len, device=device, dtype=torch.float32)
            if old_acc is not None and old_acc.shape[0] == b * ly.HKV:
                copy_w = min(old_acc.shape[1], ly.S + l_span)
                new_acc[:, :copy_w] = old_acc[:, :copy_w]
            sc["util_acc"] = new_acc

        if sc.get("loc_pos") is None or sc["loc_pos"].shape != (b, 1, l_span):
            sc["loc_pos"] = torch.zeros(b, 1, l_span, device=device, dtype=torch.float32)
            sc["loc_b"] = torch.zeros(b, ly.HKV, l_span, device=device, dtype=dt)
            sc["loc_valid"] = torch.zeros(b, 1, l_span, device=device, dtype=torch.bool)
        else:
            sc["loc_b"] = sc["loc_b"].to(device=device, dtype=dt)
            if sc["loc_k"] is not None and (sc["loc_k"].data_ptr() != buf_k[:, :, w0:w1].data_ptr()):
                buf_k[:, :, w0:w1] = sc["loc_k"].to(device=device, dtype=dt)
                buf_v[:, :, w0:w1] = sc["loc_v"].to(device=device, dtype=dt)
        lv = sc["loc_valid"].view(b, 1, 1, l_span).expand(b, ly.HKV, 1, l_span)
        mask_buf[:, :, :, w0:w1] = torch.where(lv, 0.0, float("-inf"))
        sc["loc_k"] = buf_k[:, :, w0:w1]
        sc["loc_v"] = buf_v[:, :, w0:w1]

        if cold_cap_blks > 0 and sc.get("cold_k_blk") is not None:
            n_cb = int(sc["cold_k_blk"].shape[2])
            n_tok_c = n_cb * ly.C
            buf_k[:, :, w1 : w1 + n_tok_c] = sc["cold_k_blk"].reshape(b, ly.HKV, n_tok_c, ly.HD).to(device=device, dtype=dt)
            buf_v[:, :, w1 : w1 + n_tok_c] = sc["cold_v_blk"].reshape(b, ly.HKV, n_tok_c, ly.HD).to(device=device, dtype=dt)
            cm = sc["cold_mask"].reshape(b, 1, 1, n_tok_c).expand(b, ly.HKV, 1, n_tok_c)
            mask_buf[:, :, :, w1 : w1 + n_tok_c] = torch.where(cm, 0.0, float("-inf"))
            if sc.get("cold_hit") is not None:
                c_acc_lo = ly.S + l_span
                sc["util_acc"][:, c_acc_lo : c_acc_lo + n_tok_c] = sc["cold_hit"].reshape(b * ly.HKV, n_tok_c)

        sc["cold_cap_blks"] = cold_cap_blks
        sc["buf_k"], sc["buf_v"], sc["mask_buf"] = buf_k, buf_v, mask_buf
        sc["_synced_ids"] = (
            id(buf_k),
            id(sc.get("sink_k")),
            id(sc.get("ak")),
            cold_cap_blks,
            id(sc.get("cold_k_blk")),
        )

    def _consolidate_layer(
        self,
        li: int,
        ly: Bolt3Attn,
        sc: dict,
        states: list,
        b: int,
        device: torch.device,
        inv_freq: torch.Tensor,
        gam: torch.Tensor,
        unified: bool,
    ):
        l_span = ly.loc_chunks * ly.C
        w0 = ly.n_sinks + ly.S
        w1 = w0 + l_span
        c_acc_lo = ly.S + l_span

        if sc.get("util_acc") is not None and sc["util"] is not None and ly.S_hh > 0:
            att_u = sc["util_acc"][:, : ly.S_hh] / float(ly.C)
            u_hh_m = torch.maximum(sc["util"][:, : ly.S_hh], att_u)
            sc["util"] = torch.cat([u_hh_m, sc["util"][:, ly.S_hh :]], dim=1)
        if sc.get("util_acc") is not None:
            sc["util_acc"][:, : ly.S].zero_()

        cur_slot = int(sc.get("cur_slot", 0))
        n_loc_filled = int(sc.get("n_loc_filled", 0))
        slot_pad = sc.get("loc_slot_pad")
        if slot_pad is None or len(slot_pad) != ly.loc_chunks:
            slot_pad = [False] * ly.loc_chunks
            sc["loc_slot_pad"] = slot_pad

        cold_bucket_ok = True
        if n_loc_filled < ly.loc_chunks - 1:
            sc["n_loc_filled"] = n_loc_filled + 1
            next_slot = (cur_slot + 1) % ly.loc_chunks
        else:
            pop_slot = (cur_slot + 1) % ly.loc_chunks
            p_s, p_e = pop_slot * ly.C, (pop_slot + 1) * ly.C
            p_kp = (sc["buf_k"][:, :, w0 + p_s : w0 + p_e] if unified else sc["loc_k"][:, :, p_s:p_e]).clone()
            p_v = (sc["buf_v"][:, :, w0 + p_s : w0 + p_e] if unified else sc["loc_v"][:, :, p_s:p_e]).clone()
            p_pos = sc["loc_pos"][:, :, p_s:p_e].clone()
            p_b = sc["loc_b"][:, :, p_s:p_e].clone()
            p_val = sc["loc_valid"][:, :, p_s:p_e].clone()
            p_kr = apply_rope_pos(p_kp, -p_pos.expand(b, ly.HKV, ly.C), inv_freq)
            p_hit = (
                sc["util_acc"][:, ly.S + p_s : ly.S + p_e].clone().view(b, ly.HKV, ly.C)
                if sc.get("util_acc") is not None
                else torch.zeros(b, ly.HKV, ly.C, device=device, dtype=torch.float32)
            )
            if unified and ly.is_cold_armed and ly.max_cold_blocks > 0 and sc.get("cold_k_blk") is not None:
                n_cb_old = int(sc["cold_k_blk"].shape[2])
                if sc.get("util_acc") is not None and sc["util_acc"].shape[1] >= c_acc_lo + n_cb_old * ly.C:
                    sc["cold_hit"] = sc["util_acc"][:, c_acc_lo : c_acc_lo + n_cb_old * ly.C].clone().view(b, ly.HKV, n_cb_old, ly.C)

            p_kp_f, p_v_f, p_pos_f, p_b_f, p_hit_f = p_kp, p_v, p_pos, p_b, p_hit
            all_valid_prev = bool(p_val.all())
            if sc["sink_k"] is None and all_valid_prev:
                ns = min(ly.n_sinks, p_kp.shape[2])
                sc["sink_k"] = p_kp[:, :, :ns].clone()
                sc["sink_v"] = p_v[:, :, :ns].clone()
                sc["sink_valid"] = torch.ones(b, 1, ns, device=device, dtype=torch.bool)
                sc["sink_filled"] = torch.full((b,), ns, device=device, dtype=torch.long)
                c_mask_f = torch.ones(b, 1, p_kp.shape[2], device=device, dtype=torch.bool)
                c_mask_f[:, :, :ns] = False
                p_kr, p_kp, p_v, p_pos, p_b, p_hit = (
                    p_kr[:, :, ns:],
                    p_kp[:, :, ns:],
                    p_v[:, :, ns:],
                    p_pos[:, :, ns:],
                    p_b[:, :, ns:],
                    p_hit[:, :, ns:],
                )
                tok_valid_in = (
                    torch.ones(b * ly.HKV, p_kp.shape[2], device=device, dtype=torch.bool)
                    if (sc["act"] is not None and not bool(sc["act"].all()))
                    else None
                )
            elif not all_valid_prev or (sc["sink_filled"] is not None and bool((sc["sink_filled"] < ly.n_sinks).any())):
                if sc["sink_k"] is None:
                    ns = min(ly.n_sinks, p_kp.shape[2])
                    sc["sink_k"] = torch.zeros(b, ly.HKV, ns, ly.HD, device=device, dtype=p_kp.dtype)
                    sc["sink_v"] = torch.zeros_like(sc["sink_k"])
                    sc["sink_valid"] = torch.zeros(b, 1, ns, device=device, dtype=torch.bool)
                    sc["sink_filled"] = torch.zeros(b, device=device, dtype=torch.long)
                p_val_work = p_val.clone()
                ns = sc["sink_k"].shape[2]
                for r in range(b):
                    sf = int(sc["sink_filled"][r].item())
                    if sf < ns:
                        v_idx = torch.nonzero(p_val_work[r, 0], as_tuple=False).view(-1)
                        take = min(ns - sf, int(v_idx.numel()))
                        if take > 0:
                            sel = v_idx[:take]
                            sc["sink_k"][r, :, sf : sf + take] = p_kp[r, :, sel]
                            sc["sink_v"][r, :, sf : sf + take] = p_v[r, :, sel]
                            sc["sink_valid"][r, 0, sf : sf + take] = True
                            sc["sink_filled"][r] = sf + take
                            p_val_work[r, 0, sel] = False
                c_mask_f = p_val_work
                tok_valid_in = p_val_work.expand(b, ly.HKV, -1).reshape(b * ly.HKV, -1)
            elif sc["act"] is not None and not bool(sc["act"].all()):
                c_mask_f = torch.ones(b, 1, p_kp.shape[2], device=device, dtype=torch.bool)
                tok_valid_in = torch.ones(b * ly.HKV, p_kp.shape[2], device=device, dtype=torch.bool)
            else:
                c_mask_f = torch.ones(b, 1, p_kp.shape[2], device=device, dtype=torch.bool)
                tok_valid_in = None

            if ly.is_cold_armed and ly.max_cold_blocks > 0:
                (
                    sc["cold_k_blk"],
                    sc["cold_v_blk"],
                    sc["cold_mask"],
                    sc["cold_pos"],
                    sc["cold_b"],
                    sc["cold_kmin"],
                    sc["cold_kmax"],
                    popped,
                ) = ly._push_cold_block(
                    sc.get("cold_k_blk"),
                    sc.get("cold_v_blk"),
                    sc.get("cold_mask"),
                    sc.get("cold_pos"),
                    sc.get("cold_b"),
                    sc.get("cold_kmin"),
                    sc.get("cold_kmax"),
                    p_kp_f,
                    p_v_f,
                    c_mask_f,
                    p_pos_f,
                    p_b_f,
                )
                hb = p_hit_f.unsqueeze(2)
                sc["cold_hit"] = hb.clone() if sc.get("cold_hit") is None else torch.cat([sc["cold_hit"], hb], dim=2)
                if popped is not None:
                    ok_p, ok_v, ok_m, ok_pos, ok_b = popped
                    ok_h = sc["cold_hit"][:, :, 0]
                    sc["cold_hit"] = sc["cold_hit"][:, :, 1:]
                    ok_kr = apply_rope_pos(ok_p, -ok_pos.expand(b, ly.HKV, ly.C), inv_freq)
                    tok_valid_pop = (
                        ok_m.expand(b, ly.HKV, -1).reshape(b * ly.HKV, -1)
                        if ((sc["act"] is not None and not bool(sc["act"].all())) or not bool(ok_m.all()))
                        else None
                    )
                    eye = self._diag_eye_cache.get(device)
                    if eye is None:
                        eye = torch.eye(ly.S, device=device, dtype=torch.bool).unsqueeze(0)
                        self._diag_eye_cache[device] = eye
                    m_in = states[li].view(b * ly.HKV, ly.HD, ly.HD)
                    sc["ak"], sc["av"], sc["mass"], sc["util"], sc["pm"], sc["act"], m_out, sc["n_filled"] = (
                        ly._consolidate_functional(
                            sc["ak"],
                            sc["av"],
                            sc["mass"],
                            sc["util"],
                            sc["pm"],
                            sc["act"],
                            m_in,
                            ok_kr,
                            ok_p,
                            ok_v,
                            ok_pos,
                            ok_b,
                            inv_freq,
                            eye,
                            gam,
                            n_filled=sc.get("n_filled", 0),
                            tok_valid=tok_valid_pop,
                            obs_hit=ok_h,
                        )
                    )
                    states[li].copy_(m_out.view(b, ly.HKV, ly.HD, ly.HD))
                    self._refresh_attractor_cache(sc, ly, b, inv_freq)
            else:
                eye = self._diag_eye_cache.get(device)
                if eye is None:
                    eye = torch.eye(ly.S, device=device, dtype=torch.bool).unsqueeze(0)
                    self._diag_eye_cache[device] = eye
                m_in = states[li].view(b * ly.HKV, ly.HD, ly.HD)
                sc["ak"], sc["av"], sc["mass"], sc["util"], sc["pm"], sc["act"], m_out, sc["n_filled"] = (
                    ly._consolidate_functional(
                        sc["ak"],
                        sc["av"],
                        sc["mass"],
                        sc["util"],
                        sc["pm"],
                        sc["act"],
                        m_in,
                        p_kr,
                        p_kp,
                        p_v,
                        p_pos,
                        p_b,
                        inv_freq,
                        eye,
                        gam,
                        n_filled=sc.get("n_filled", 0),
                        tok_valid=tok_valid_in,
                        obs_hit=p_hit,
                    )
                )
                states[li].copy_(m_out.view(b, ly.HKV, ly.HD, ly.HD))
                self._refresh_attractor_cache(sc, ly, b, inv_freq)

            if unified:
                ns = sc["sink_k"].shape[2]
                sc["buf_k"][:, :, :ns] = sc["sink_k"]
                sc["buf_v"][:, :, :ns] = sc["sink_v"]
                if sc.get("sink_valid") is not None:
                    sv = sc["sink_valid"].view(b, 1, 1, ns).expand(b, ly.HKV, 1, ns)
                    sc["mask_buf"][:, :, :, :ns] = torch.where(sv, 0.0, float("-inf"))
                else:
                    sc["mask_buf"][:, :, :, :ns] = 0.0
                if sc["ak"] is not None:
                    sc["buf_k"][:, :, ly.n_sinks : w0] = sc["ak_eff"]
                    sc["buf_v"][:, :, ly.n_sinks : w0] = sc["av"].view(b, ly.HKV, ly.S, ly.HD)
                    sc["mask_buf"][:, :, :, ly.n_sinks : w0] = sc["att_bias"].view(b, ly.HKV, 1, ly.S)
                if ly.is_cold_armed and ly.max_cold_blocks > 0 and ly.cold_topk == 0:
                    new_cap = self._cold_cap_for(ly, sc)
                    if new_cap != sc.get("cold_cap_blks", 0):
                        cold_bucket_ok = False
                    elif new_cap > 0 and sc.get("cold_k_blk") is not None:
                        n_tok_c = int(sc["cold_k_blk"].shape[2]) * ly.C
                        sc["buf_k"][:, :, w1 : w1 + n_tok_c] = sc["cold_k_blk"].reshape(b, ly.HKV, n_tok_c, ly.HD)
                        sc["buf_v"][:, :, w1 : w1 + n_tok_c] = sc["cold_v_blk"].reshape(b, ly.HKV, n_tok_c, ly.HD)
                        cm = sc["cold_mask"].reshape(b, 1, 1, n_tok_c).expand(b, ly.HKV, 1, n_tok_c)
                        sc["mask_buf"][:, :, :, w1 : w1 + n_tok_c] = torch.where(cm, 0.0, float("-inf"))
                        if sc.get("util_acc") is not None and sc.get("cold_hit") is not None:
                            sc["util_acc"][:, c_acc_lo : c_acc_lo + n_tok_c] = sc["cold_hit"].reshape(b * ly.HKV, n_tok_c)
            next_slot = pop_slot

        ns_lo, ns_hi = next_slot * ly.C, (next_slot + 1) * ly.C
        sc["cur_slot"] = next_slot
        slot_pad[next_slot] = False
        sc["loc_valid"][:, :, ns_lo:ns_hi].fill_(False)
        if sc.get("util_acc") is not None:
            sc["util_acc"][:, ly.S + ns_lo : ly.S + ns_hi].zero_()
        sc["cur_len"] = 0
        if unified:
            sc["mask_buf"][:, :, :, w0 + ns_lo : w0 + ns_hi].fill_(float("-inf"))
            if cold_bucket_ok:
                sc["_synced_ids"] = (
                    id(sc["buf_k"]),
                    id(sc.get("sink_k")),
                    id(sc.get("ak")),
                    sc.get("cold_cap_blks", 0),
                    id(sc.get("cold_k_blk")),
                )
                if sc.get("_runner_id") is not None:
                    sc["_runner_id"] = (
                        sc["_runner_id"][0],
                        id(states[li]),
                        id(sc["buf_k"]),
                        id(sc.get("sink_k")),
                        id(sc.get("ak")),
                        sc.get("cold_cap_blks", 0),
                        id(sc.get("cold_k_blk")),
                    )
            else:
                sc["_synced_ids"] = None
                sc["_runner_id"] = None
        else:
            sc["cold_cap_blks"] = 0
            sc["buf_k"] = None
            sc["_synced_ids"] = None
            sc["_runner_id"] = None

    def step_block(
        self,
        li: int,
        blk,
        x: torch.Tensor,
        pos_start: int,
        states,
        kv_caches,
        pos_ids: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        _rope_cs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Step one Transformer block on a chunk-aligned slice `(B, t_s, D)` where `cur_len + t_s <= C`."""
        b, t_s, _ = x.shape
        device = x.device
        ly = blk.attn
        if isinstance(ly, Bolt3Attn):
            sc: dict = kv_caches[li]
            inv_freq = ly.rope.inv_freq.to(device)
            gam, _ = self._get_gam_scale(ly, device)

            # Fast zero-cat unified buffer path for unpadded single-token decode steps
            if t_s == 1 and valid_mask is None and not (ly.is_cold_armed and ly.max_cold_blocks > 0 and ly.cold_topk > 0):
                self._sync_unified_bufs(sc, ly, b, device, x.dtype)
                if _rope_cs is None:
                    if pos_ids is None:
                        cos, sin = ly.rope.cos_sin(pos_start, 1, device, x.dtype)
                        pos_n = torch.full((b, 1, 1), float(pos_start), device=device, dtype=torch.float32)
                    else:
                        cos, sin = ly.rope.cos_sin(pos_start, 1, device, x.dtype, pos_ids=pos_ids)
                        pos_n = pos_ids.unsqueeze(1).float()
                else:
                    cos, sin = _rope_cs
                    pos_n = (
                        torch.full((b, 1, 1), float(pos_start), device=device, dtype=torch.float32)
                        if pos_ids is None
                        else pos_ids.unsqueeze(1).float()
                    )
                len_before = int(sc["cur_len"])
                cur_slot = int(sc.get("cur_slot", 0))
                slot_loc = cur_slot * ly.C + len_before
                slot_buf = ly.n_sinks + ly.S + slot_loc
                x = _steady_block_fast(
                    ly,
                    blk.norm1,
                    blk.norm2,
                    blk.mlp,
                    x,
                    cos,
                    sin,
                    pos_n,
                    self._get_slot_idx(ly, slot_buf, device),
                    self._get_slot_idx(ly, slot_loc, device),
                    self._get_zero_mask(b, ly.HKV, device),
                    sc["buf_k"],
                    sc["buf_v"],
                    sc["mask_buf"],
                    sc["util_acc"],
                    sc["loc_pos"],
                    sc["loc_b"],
                    states[li],
                    post_attn_norm=getattr(blk, "post_attn_norm", None),
                    post_mlp_norm=getattr(blk, "post_mlp_norm", None),
                )
                sc["loc_valid"][:, :, slot_loc].fill_(True)
                new_len = len_before + 1
                sc["cur_len"] = new_len
                if new_len >= ly.C:
                    self._consolidate_layer(li, ly, sc, states, b, device, inv_freq, gam, unified=True)
                return x

            # Multi-token prefill / valid_mask causal path
            sc["_synced_ids"] = None
            sc["buf_k"] = None
            h = blk.norm1(x)
            q_r, k_r, v_n, gate = ly.head_qkv_gate(h)
            if pos_ids is None:
                if _rope_cs is not None:
                    cos, sin = _rope_cs
                else:
                    cos, sin = ly.rope.cos_sin(pos_start, t_s, device, q_r.dtype)
                pos_n = (
                    torch.arange(pos_start, pos_start + t_s, device=device, dtype=torch.float32)
                    .view(1, 1, t_s)
                    .expand(b, 1, t_s)
                )
            else:
                if _rope_cs is not None:
                    cos, sin = _rope_cs
                else:
                    cos, sin = ly.rope.cos_sin(pos_start, t_s, device, q_r.dtype, pos_ids=pos_ids)
                pos_n = pos_ids.unsqueeze(1).float()
            q_p = apply_rope_cos_sin(q_r, cos, sin, ly.rope.rotary_dim)
            k_p = apply_rope_cos_sin(k_r, cos, sin, ly.rope.rotary_dim)

            beta_n = torch.sigmoid(ly.beta_proj(h)).transpose(1, 2)
            if valid_mask is not None:
                beta_n = beta_n * valid_mask.unsqueeze(1).to(beta_n.dtype)
            eta_n = torch.tanh(ly.eta_proj(h)).transpose(1, 2).unsqueeze(-1)
            scale = ly.attn_scale

            l_span = ly.loc_chunks * ly.C
            if sc.get("loc_k") is None or sc["loc_k"].shape != (b, ly.HKV, l_span, ly.HD) or sc["loc_k"].dtype != k_p.dtype:
                sc["loc_k"] = torch.zeros(b, ly.HKV, l_span, ly.HD, device=device, dtype=k_p.dtype)
                sc["loc_v"] = torch.zeros(b, ly.HKV, l_span, ly.HD, device=device, dtype=v_n.dtype)
                sc["loc_pos"] = torch.zeros(b, 1, l_span, device=device, dtype=torch.float32)
                sc["loc_b"] = torch.zeros(b, ly.HKV, l_span, device=device, dtype=beta_n.dtype)
                sc["loc_valid"] = torch.zeros(b, 1, l_span, device=device, dtype=torch.bool)
                sc["loc_slot_pad"] = [False] * ly.loc_chunks
                sc["cur_slot"] = 0
                sc["n_loc_filled"] = 0
                sc["cur_len"] = 0

            len_before = int(sc["cur_len"])
            cur_slot = int(sc["cur_slot"])
            new_len = len_before + t_s
            s_lo = cur_slot * ly.C + len_before
            s_hi = cur_slot * ly.C + new_len
            sc["loc_k"][:, :, s_lo:s_hi] = k_p
            sc["loc_v"][:, :, s_lo:s_hi] = v_n
            sc["loc_pos"][:, :, s_lo:s_hi] = pos_n
            sc["loc_b"][:, :, s_lo:s_hi] = beta_n
            if valid_mask is not None:
                sc["loc_valid"][:, :, s_lo:s_hi] = valid_mask.unsqueeze(1)
                if not bool(valid_mask.all()):
                    sc["loc_slot_pad"][cur_slot] = True
            else:
                sc["loc_valid"][:, :, s_lo:s_hi] = True
            sc["cur_len"] = new_len

            qg = q_p.view(b, ly.HKV, ly.G, t_s, ly.HD)
            s_loc = torch.matmul(qg.float(), sc["loc_k"].float().unsqueeze(2).transpose(-1, -2)) * scale
            kj_all = torch.arange(l_span, device=device)[None, :]
            cur_base = cur_slot * ly.C
            in_cur = (kj_all >= cur_base) & (kj_all < cur_base + ly.C)
            qi = torch.arange(t_s, device=device)[:, None]
            cur_ok = (~in_cur) | ((kj_all - cur_base) <= (len_before + qi))
            cm = cur_ok.view(1, 1, 1, t_s, l_span) & sc["loc_valid"].view(b, 1, 1, 1, l_span)
            has_pad = any(sc["loc_slot_pad"])
            if has_pad:
                q_ok = valid_mask if valid_mask is not None else torch.ones(b, t_s, device=device, dtype=torch.bool)
                self_pos = (kj_all == (cur_base + len_before + qi)).view(1, 1, 1, t_s, l_span)
                cm = cm | (self_pos & (~q_ok).view(b, 1, 1, t_s, 1))
            s_loc = s_loc.masked_fill(~cm, float("-inf"))

            sp, vp = [], []
            if sc["sink_k"] is not None:
                s_sink = torch.matmul(qg.float(), sc["sink_k"].float().unsqueeze(2).transpose(-1, -2)) * scale
                if sc.get("sink_valid") is not None and not bool(sc["sink_valid"].all()):
                    s_sink = s_sink.masked_fill(~sc["sink_valid"].view(b, 1, 1, 1, -1), float("-inf"))
                sp.append(s_sink)
                vp.append(sc["sink_v"])
            if sc["ak"] is not None:
                if sc.get("ak_eff") is None or sc.get("att_bias") is None:
                    self._refresh_attractor_cache(sc, ly, b, inv_freq)
                s_att = (
                    torch.matmul(qg.float(), sc["ak_eff"].float().unsqueeze(2).transpose(-1, -2)) * scale
                    + sc["att_bias"].view(b, ly.HKV, 1, 1, ly.S)
                )
                sp.append(s_att)
                vp.append(sc["av"].view(b, ly.HKV, ly.S, ly.HD))

            sink_off = sc["sink_k"].shape[2] if sc["sink_k"] is not None else 0
            att_slice = (sink_off, sink_off + ly.S) if sc["ak"] is not None else None
            loc_off = sink_off + (ly.S if sc["ak"] is not None else 0)
            loc_slice = (loc_off, loc_off + l_span)
            sp.append(s_loc)
            vp.append(sc["loc_v"])
            y, att_p, loc_p, cold_delta = ly._attend_with_cold(
                qg,
                sp,
                vp,
                sc.get("cold_k_blk"),
                sc.get("cold_v_blk"),
                sc.get("cold_mask"),
                sc.get("cold_kmin"),
                sc.get("cold_kmax"),
                att_slice=att_slice,
                loc_slice=loc_slice,
                q_valid_mask=valid_mask,
            )

            need_acc_len = ly.S + l_span
            if sc.get("util_acc") is None or sc["util_acc"].shape[0] != b * ly.HKV or sc["util_acc"].shape[1] < need_acc_len:
                sc["util_acc"] = torch.zeros(b * ly.HKV, need_acc_len, device=device, dtype=torch.float32)
            if att_p is not None and ly.S_hh > 0:
                sc["util_acc"][:, : ly.S].add_(att_p.mean(dim=2).sum(dim=2).reshape(b * ly.HKV, ly.S))
            if loc_p is not None:
                sc["util_acc"][:, ly.S : ly.S + l_span].add_(loc_p.mean(dim=2).sum(dim=2).reshape(b * ly.HKV, l_span))
            if cold_delta is not None and sc.get("cold_hit") is not None:
                sc["cold_hit"].add_(cold_delta)

            if states[li] is not None:
                y = y + ly.read_mres(q_r, states[li], eta_n).to(y.dtype)

            out_blk = y.transpose(1, 2).reshape(b, t_s, ly.H * ly.HD)
            if gate is not None:
                out_blk = out_blk * torch.sigmoid(gate)
            attn_d = ly.o_proj(out_blk)
            p_an = getattr(blk, "post_attn_norm", None)
            if p_an is not None:
                attn_d = p_an(attn_d)
            x = x + attn_d

            if new_len >= ly.C:
                self._consolidate_layer(li, ly, sc, states, b, device, inv_freq, gam, unified=False)
        elif isinstance(ly, GatedDeltaNet):
            sc = kv_caches[li]
            h = blk.norm1(x)
            attn_d = ly.step_with_state(h, sc["conv_state"], states[li], valid_mask=valid_mask)
            p_an = getattr(blk, "post_attn_norm", None)
            if p_an is not None:
                attn_d = p_an(attn_d)
            x = x + attn_d
        else:
            h = blk.norm1(x)
            q, k, v, gate = ly.head_qkv_gate(h)
            q, k = ly.rope(q, pos_start=pos_start), ly.rope(k, pos_start=pos_start)
            n_prev = kv_caches[li].n
            pk, pv = kv_caches[li].append(k, v)
            sw = getattr(ly, "sliding_window", None)
            if t_s > 1 or (sw is not None and (n_prev + t_s) > sw):
                qi = torch.arange(t_s, device=device)[:, None] + n_prev
                kj = torch.arange(n_prev + t_s, device=device)[None, :]
                cmask = kj <= qi
                if sw is not None:
                    cmask = cmask & (kj > (qi - sw))
                cmask = cmask.view(1, 1, t_s, n_prev + t_s)
                try:
                    y = F.scaled_dot_product_attention(
                        q, pk.to(q.dtype), pv.to(q.dtype), attn_mask=cmask, scale=ly.attn_scale, enable_gqa=(ly.G > 1)
                    )
                except TypeError:
                    y = F.scaled_dot_product_attention(
                        q, repeat_kv(pk.to(q.dtype), ly.G), repeat_kv(pv.to(q.dtype), ly.G), attn_mask=cmask, scale=ly.attn_scale
                    )
            else:
                try:
                    y = F.scaled_dot_product_attention(
                        q, pk.to(q.dtype), pv.to(q.dtype), scale=ly.attn_scale, enable_gqa=(ly.G > 1)
                    )
                except TypeError:
                    y = F.scaled_dot_product_attention(
                        q, repeat_kv(pk.to(q.dtype), ly.G), repeat_kv(pv.to(q.dtype), ly.G), scale=ly.attn_scale
                    )
            out_blk = y.transpose(1, 2).reshape(b, t_s, ly.H * ly.HD)
            if gate is not None:
                out_blk = out_blk * torch.sigmoid(gate)
            elif getattr(ly, "g_proj", None) is not None:
                out_blk = out_blk * torch.sigmoid(ly.g_proj(h))
            attn_d = ly.o_proj(out_blk)
            p_an = getattr(blk, "post_attn_norm", None)
            if p_an is not None:
                attn_d = p_an(attn_d)
            x = x + attn_d

        mlp_d = blk.mlp(blk.norm2(x))
        p_mn = getattr(blk, "post_mlp_norm", None)
        if p_mn is not None:
            mlp_d = p_mn(mlp_d)
        return x + mlp_d

    def _can_use_graph(
        self, tokens: torch.Tensor, valid_mask, kv_caches, dtype: torch.dtype, cold_caps: tuple[int, ...]
    ) -> bool:
        if not (tokens.is_cuda and tokens.shape[1] == 1 and valid_mask is None and self.use_cuda_graph):
            return False
        if not (self._all_fast_step and len(kv_caches) == len(self.model.blocks)):
            return False
        if any(
            isinstance(blk.attn, Bolt3Attn) and blk.attn.is_cold_armed and blk.attn.max_cold_blocks > 0 and blk.attn.cold_topk > 0
            for blk in self.model.blocks
        ):
            return False
        b = tokens.shape[0]
        if b == 1:
            return True
        if b > 64:
            return False
        key = (b, str(tokens.device), dtype, cold_caps)
        if key in self._graph_runners:
            return True
        try:
            free_b, _ = torch.cuda.mem_get_info(tokens.device)
            cold_mb = sum(
                b * cap * blk.attn.C * blk.attn.HKV * blk.attn.HD * 5 // (1024 * 1024)
                for cap, blk in zip(cold_caps, self.model.blocks)
                if cap > 0 and isinstance(blk.attn, Bolt3Attn)
            )
            return free_b >= (b * 128 + cold_mb + 2048) * 1024 * 1024
        except Exception:
            return False

    def _get_graph_runner(
        self, b: int, device: torch.device, dtype: torch.dtype, cold_caps: tuple[int, ...]
    ) -> _Bolt3GraphRunner:
        key = (b, str(device), dtype, cold_caps)
        runner = self._graph_runners.get(key)
        if runner is None:
            dropped = False
            for k_old in list(self._graph_runners.keys()):
                drop_batch = (b > 1 and k_old[0] != 1 and k_old != key)
                drop_cold_b1 = (b == 1 and any(cold_caps) and k_old[0] == 1 and any(k_old[3]) and k_old != key)
                if drop_batch or drop_cold_b1:
                    r_old = self._graph_runners.pop(k_old)
                    r_old.close()
                    del r_old
                    dropped = True
            if dropped and torch.cuda.is_available():
                torch.cuda.empty_cache()
            runner = _Bolt3GraphRunner(self.model, b, device, dtype, cold_caps=cold_caps)
            self._graph_runners[key] = runner
        return runner

    @staticmethod
    def _bind_graph_state(
        dec,
        runner: _Bolt3GraphRunner,
        states: list,
        kv_caches: list[dict],
        b: int,
        device: torch.device,
        dt: torch.dtype,
    ):
        rid = runner.uid
        for li, blk in enumerate(dec.model.blocks):
            sc = kv_caches[li]
            ly = blk.attn
            if isinstance(ly, GatedDeltaNet):
                expected_sig = (rid, id(states[li]), id(sc.get("conv_state")))
                if sc.get("_runner_id") == expected_sig and runner.states[li] is states[li]:
                    continue
                if runner.states[li] is not states[li]:
                    runner.states[li].copy_(states[li])
                    states[li] = runner.states[li]
                if runner.conv_states[li] is not sc["conv_state"]:
                    runner.conv_states[li].copy_(sc["conv_state"])
                    sc["conv_state"] = runner.conv_states[li]
                sc["_runner_id"] = (rid, id(states[li]), id(sc["conv_state"]))
                continue

            cold_cap = runner.cold_caps[li]
            expected_sig = (
                rid,
                id(states[li]),
                id(sc.get("buf_k")),
                id(sc.get("sink_k")),
                id(sc.get("ak")),
                cold_cap,
                id(sc.get("cold_k_blk")),
            )
            if (
                sc.get("_runner_id") == expected_sig
                and runner.states[li] is states[li]
                and runner.buf_k[li] is sc.get("buf_k")
            ):
                continue
            dec._sync_unified_bufs(sc, ly, b, device, dt)
            if runner.states[li] is not states[li]:
                runner.states[li].copy_(states[li])
                states[li] = runner.states[li]
            if runner.buf_k[li] is not sc["buf_k"]:
                runner.buf_k[li].copy_(sc["buf_k"])
                runner.buf_v[li].copy_(sc["buf_v"])
                runner.mask_buf[li].copy_(sc["mask_buf"])
                sc["buf_k"] = runner.buf_k[li]
                sc["buf_v"] = runner.buf_v[li]
                sc["mask_buf"] = runner.mask_buf[li]
            if runner.util_acc[li] is not sc["util_acc"]:
                runner.util_acc[li].copy_(sc["util_acc"])
                sc["util_acc"] = runner.util_acc[li]
            if runner.loc_pos[li] is not sc["loc_pos"]:
                runner.loc_pos[li].copy_(sc["loc_pos"])
                runner.loc_b[li].copy_(sc["loc_b"])
                sc["loc_pos"] = runner.loc_pos[li]
                sc["loc_b"] = runner.loc_b[li]

            w0 = ly.n_sinks + ly.S
            w1 = w0 + ly.loc_chunks * ly.C
            sc["loc_k"] = sc["buf_k"][:, :, w0:w1]
            sc["loc_v"] = sc["buf_v"][:, :, w0:w1]
            sc["_synced_ids"] = (
                id(sc["buf_k"]),
                id(sc.get("sink_k")),
                id(sc.get("ak")),
                cold_cap,
                id(sc.get("cold_k_blk")),
            )
            sc["_runner_id"] = (
                rid,
                id(states[li]),
                id(sc["buf_k"]),
                id(sc.get("sink_k")),
                id(sc.get("ak")),
                cold_cap,
                id(sc.get("cold_k_blk")),
            )

    @torch.no_grad()
    def step(
        self,
        tokens: torch.Tensor,
        pos_start: int,
        states,
        kv_caches,
        pos_ids: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ):
        m = self.model
        t_tot = tokens.shape[1]
        emb_dt = m.embed_tokens.weight.dtype
        use_bf16 = tokens.is_cuda and emb_dt == torch.bfloat16
        dt = (
            emb_dt
            if (tokens.is_cuda and emb_dt in (torch.bfloat16, torch.float16, torch.float32))
            else (torch.bfloat16 if tokens.is_cuda else torch.float32)
        )
        if t_tot == 1:
            b = tokens.shape[0]
            cold_caps = (
                tuple(
                    self._cold_cap_for(blk.attn, kv_caches[li])
                    if (isinstance(blk.attn, Bolt3Attn) and isinstance(kv_caches[li], dict))
                    else 0
                    for li, blk in enumerate(m.blocks)
                )
                if len(kv_caches) == len(m.blocks)
                else ()
            )
            if self._can_use_graph(tokens, valid_mask, kv_caches, dt, cold_caps):
                b3 = self._b3_indices
                len0 = int(kv_caches[b3[0]].get("cur_len", 0)) if b3 else 0
                slot0 = int(kv_caches[b3[0]].get("cur_slot", 0)) if b3 else 0
                if (not b3) or all(
                    kv_caches[li].get("cur_len", 0) == len0 and kv_caches[li].get("cur_slot", 0) == slot0
                    for li in b3
                ):
                    runner = self._get_graph_runner(b, tokens.device, dt, cold_caps)
                    self._bind_graph_state(self, runner, states, kv_caches, b, tokens.device, dt)
                    ly0 = self._first_bolt3
                    slot_loc = (slot0 * ly0.C + len0) if ly0 is not None else 0
                    slot_buf = (ly0.n_sinks + ly0.S + slot_loc) if ly0 is not None else 0
                    out_h = runner.replay(tokens, pos_start, pos_ids, slot_buf, slot_loc)
                    new_len = len0 + 1
                    for li in b3:
                        sc_li = kv_caches[li]
                        sc_li["loc_valid"][:, :, slot_loc].fill_(True)
                        sc_li["cur_len"] = new_len
                    if ly0 is not None and new_len >= ly0.C:
                        for li in b3:
                            ly = m.blocks[li].attn
                            inv_freq = ly.rope.inv_freq.to(tokens.device)
                            gam, _ = self._get_gam_scale(ly, tokens.device)
                            self._consolidate_layer(li, ly, kv_caches[li], states, b, tokens.device, inv_freq, gam, unified=True)
                    return out_h, states, kv_caches

            with torch.autocast("cuda", torch.bfloat16, enabled=use_bf16):
                x = m.embed_tokens(tokens)
                rope_cs = None
                if self._shared_rope and self._first_bolt3 is not None:
                    rope_cs = self._first_bolt3.rope.cos_sin(pos_start, 1, tokens.device, x.dtype, pos_ids=pos_ids)
                for li, blk in enumerate(m.blocks):
                    x = self.step_block(
                        li, blk, x, pos_start, states, kv_caches, pos_ids=pos_ids, valid_mask=valid_mask, _rope_cs=rope_cs
                    )
                return m.norm(x), states, kv_caches

        outs = []
        s = 0
        pos = pos_start
        while s < t_tot:
            cur_len = 0
            for c in kv_caches:
                if isinstance(c, dict) and "cur_len" in c:
                    cur_len = c.get("cur_len", 0)
                    break
            take = min(t_tot - s, self.C - cur_len)
            p_slice = pos_ids[:, s : s + take] if pos_ids is not None else None
            v_slice = valid_mask[:, s : s + take] if valid_mask is not None else None
            with torch.autocast("cuda", torch.bfloat16, enabled=use_bf16):
                x = m.embed_tokens(tokens[:, s : s + take])
                rope_cs = None
                if self._shared_rope and self._first_bolt3 is not None:
                    rope_cs = self._first_bolt3.rope.cos_sin(pos, take, tokens.device, x.dtype, pos_ids=p_slice)
                for li, blk in enumerate(m.blocks):
                    x = self.step_block(
                        li, blk, x, pos, states, kv_caches, pos_ids=p_slice, valid_mask=v_slice, _rope_cs=rope_cs
                    )
                outs.append(m.norm(x))
            s += take
            pos += take
        out_h = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
        return out_h, states, kv_caches

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor, max_kv_len: int = 4096, attn_mask: torch.Tensor | None = None):
        b, t = ids.shape
        states, kv = self.blank_state(self.model, b, ids.device, max_kv_len=max(max_kv_len, t + 1024))
        if attn_mask is not None:
            valid_m = attn_mask.bool()
            pos_ids = (valid_m.long().cumsum(dim=-1) - 1).clamp_min(0)
        else:
            valid_m, pos_ids = None, None
        h, states, kv = self.step(ids, 0, states, kv, pos_ids=pos_ids, valid_mask=valid_m)
        return h, states, kv

    @torch.no_grad()
    def generate_batch(
        self,
        ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.6,
        top_k: int = 40,
        top_p: float = 0.95,
        eos_ids: set[int] | None = None,
        close_think_ids: list[int] | None = None,
        repetition_penalty: float = 1.0,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = ids.shape
        h, states, kv = self.prefill(ids, max_kv_len=T + max_new_tokens + 64, attn_mask=attn_mask)
        logits = self.model.lm_head(h[:, -1, :])
        eos = torch.tensor(sorted(eos_ids or ()), device=ids.device, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=ids.device)
        lengths = torch.full((B,), max_new_tokens, dtype=torch.long, device=ids.device)
        end_think_id = close_think_ids[1] if (close_think_ids and len(close_think_ids) > 1) else None
        closed_think = torch.zeros(B, dtype=torch.bool, device=ids.device)
        q_pos = torch.full((B,), -1, dtype=torch.long, device=ids.device)
        budget_trigger = max(64, max_new_tokens - 64)
        hist = ids if repetition_penalty != 1.0 else None

        if attn_mask is not None:
            row_pos = attn_mask.long().sum(dim=-1, keepdim=True)
        else:
            row_pos = None

        out, pos = [], T
        for i in range(max_new_tokens):
            nxt = self._sample(
                logits,
                temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                history_ids=hist,
            )
            if close_think_ids:
                if i == budget_trigger:
                    need = (~closed_think) & (~done) & (q_pos < 0)
                    q_pos[need] = 0
                active_q = q_pos >= 0
                if bool(active_q.any()):
                    for r in torch.nonzero(active_q, as_tuple=False).view(-1).tolist():
                        nxt[r, 0] = close_think_ids[int(q_pos[r])]
                        q_pos[r] += 1
                        if q_pos[r] >= len(close_think_ids):
                            q_pos[r] = -1
                            closed_think[r] = True
                if end_think_id is not None:
                    closed_think |= nxt[:, 0] == end_think_id
            out.append(nxt)
            if hist is not None:
                hist = torch.cat([hist, nxt], dim=1)
            if eos.numel():
                hit = torch.isin(nxt[:, 0], eos) & ~done
                lengths[hit] = i + 1
                done |= hit
                if i % 8 == 7 and bool(done.all()):
                    break
            h, states, kv = self.step(nxt, pos, states, kv, pos_ids=row_pos)
            logits = self.model.lm_head(h[:, -1, :])
            pos += 1
            if row_pos is not None:
                row_pos = row_pos + 1
        return torch.cat(out, dim=1), lengths

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.95,
        eos_ids: set[int] | None = None,
        close_think_ids: list[int] | None = None,
        repetition_penalty: float = 1.0,
    ):
        toks, n = self.generate_batch(
            prompt,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_ids=eos_ids,
            close_think_ids=close_think_ids,
            repetition_penalty=repetition_penalty,
        )
        return torch.cat([prompt, toks[:, : int(n[0].item())]], dim=1)

    @torch.no_grad()
    def continue_generate(
        self,
        ids: torch.Tensor,
        states,
        kv_caches,
        start_pos: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.95,
        eos_ids: set[int] | None = None,
    ):
        h, states, kv_caches = self.step(ids, start_pos, states, kv_caches)
        pos = start_pos + ids.shape[1]
        logits = self.model.lm_head(h[:, -1, :])
        out = []
        for _ in range(max_new_tokens):
            nxt = self._sample(logits, temperature, top_k=top_k, top_p=top_p)
            out.append(nxt)
            if eos_ids and int(nxt[0, 0].item()) in eos_ids:
                break
            h, states, kv_caches = self.step(nxt, pos, states, kv_caches)
            logits = self.model.lm_head(h[:, -1, :])
            pos += 1
        toks = (
            torch.cat(out, dim=1)
            if out
            else torch.empty(ids.shape[0], 0, dtype=torch.long, device=ids.device)
        )
        return toks, states, kv_caches, pos

    @torch.no_grad()
    def memorize(self, ids: torch.Tensor, states, kv_caches, start_pos: int):
        pre_s = [s.clone() if s is not None else None for s in states]
        _, states, kv_caches = self.step(ids, start_pos, states, kv_caches)
        n_tok = ids.shape[1]
        pos = start_pos + n_tok
        traces = []
        for li, blk in enumerate(self.model.blocks):
            if isinstance(blk.attn, Bolt3Attn):
                ly = blk.attn
                gam = ly.gamma_delta().view(1, -1, 1, 1)
                off = max(1, ly.loc_chunks - 1) + (ly.max_cold_blocks if ly.is_cold_armed else 0)
                start_chunks = start_pos // ly.C
                end_chunks = pos // ly.C
                n_res_steps = max(0, max(0, end_chunks - off) - max(0, start_chunks - off))
                delta = states[li] - (gam ** n_res_steps) * pre_s[li]
                traces.append((pos, delta))
            else:
                traces.append(None)
        return states, kv_caches, pos, traces

    @torch.no_grad()
    def forget(self, traces, states, next_pos: int, kv_caches=None, pre_kv_lens=None):
        for li, tr in enumerate(traces):
            if tr is not None and states[li] is not None:
                end_p, delta = tr
                ly = self.model.blocks[li].attn
                gam = ly.gamma_delta().view(1, -1, 1, 1)
                off = max(1, ly.loc_chunks - 1) + (ly.max_cold_blocks if ly.is_cold_armed else 0)
                end_chunks = end_p // ly.C
                next_chunks = next_pos // ly.C
                n_res_steps = max(0, max(0, next_chunks - off) - max(0, end_chunks - off))
                states[li] = states[li] - (gam ** n_res_steps) * delta
        return states
