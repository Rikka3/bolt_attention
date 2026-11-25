"""Bolt-3 Dual-Field Attention: Attention Sinks + Local Window, Diversity Softmax Attractors,
and Gershgorin Eviction-to-Delta Residual Associative Memory (M_res).
"""
from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import RMSNorm, RoPE, apply_rope_cos_sin, apply_rope_pos, repeat_kv

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
        self.S = int(S if S is not None else cfg.get("bolt3_S", 128))
        self.C = int(C if C is not None else cfg.get("bolt3_C", 128))
        self.n_sinks = int(cfg.get("bolt3_n_sinks", 4))
        self.sim_k = float(cfg.get("bolt3_sim_k", 0.94))
        self.sim_v = float(cfg.get("bolt3_sim_v", 0.88))
        self.gamma_u = float(cfg.get("bolt3_gamma_u", 0.985))
        cold_layers = {int(i) for i in (cfg.get("bolt3_cold_layers") or ())}
        self.layer_idx = int(layer_idx)
        self.is_cold_armed = self.layer_idx in cold_layers
        self.max_cold_blocks = max(0, int(cfg.get("bolt3_max_cold_blocks", 256)))
        self.cold_topk = max(0, int(cfg.get("bolt3_cold_topk", 0)))

        self.q_proj = nn.Linear(D, H * HD, bias=qkv_bias)
        self.k_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.v_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.o_proj = nn.Linear(H * HD, D, bias=False)
        if cfg.get("qk_norm", False):
            self.q_norm = RMSNorm(HD, float(cfg["eps"]))
            self.k_norm = RMSNorm(HD, float(cfg["eps"]))
        else:
            self.q_norm, self.k_norm = None, None
        self.rope = RoPE(
            HD,
            float(cfg["rope_theta"]),
            cfg.get("rope_scaling"),
            float(cfg.get("partial_rotary_factor", 1.0)),
        )

        self.kappa = nn.Parameter(torch.ones(HKV))
        self.log_gamma = nn.Parameter(torch.full((HKV,), 3.5))
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
        return (0.999 * torch.sigmoid(self.log_gamma)).clamp(1e-3, 0.999).float()

    def head_qkv(self, x: torch.Tensor):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.H, self.HD)
        k = self.k_proj(x).view(b, t, self.HKV, self.HD)
        v = self.v_proj(x).view(b, t, self.HKV, self.HD)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

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
    ):
        b, hkv, l_tok, hd = k_raw.shape
        bh = b * hkv
        dt = k_raw.dtype
        kr_f = k_raw.reshape(bh, l_tok, hd)
        kp_f = k_rope.reshape(bh, l_tok, hd)
        v_f = v_old.reshape(bh, l_tok, hd)
        p_f = pos_old.expand(b, hkv, l_tok).reshape(bh, l_tok).float()
        b_f = beta_old.reshape(bh, l_tok)
        scale = hd ** -0.5

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

        kap = self.kappa.float().repeat(b).view(bh, 1, 1)

        # Left-padded batch inference path: per-row valid token mask prevents pad tokens
        # from occupying attractors or decaying/writing into M_res.
        if tok_valid is not None:
            row_has_valid = tok_valid.any(dim=-1)
            util = torch.where(row_has_valid.unsqueeze(-1), util * self.gamma_u, util)
            eligible = tok_valid

            def _score_masked(ak_in, av_in, mass_in, act_in):
                has_any = act_in.any(dim=-1, keepdim=True)
                k_read = apply_rope_pos(
                    ak_in.view(b, hkv, self.S, hd), pm.view(b, hkv, self.S), inv_freq
                ).view(bh, self.S, hd)
                with torch.no_grad():
                    sc = torch.bmm(kp_f, k_read.transpose(-1, -2)).float() * scale + kap * torch.log(
                        mass_in.clamp_min(1.0)
                    ).unsqueeze(1)
                    sc.masked_fill_(~act_in.unsqueeze(1), -1e9)
                    v_hat = torch.where(
                        has_any.unsqueeze(-1),
                        torch.bmm(F.softmax(sc, dim=-1).to(dt), av_in).float(),
                        0.0,
                    )
                    v_f32 = v_f.float()
                    rel_err = (v_f32 - v_hat).norm(dim=-1) / (v_f32.norm(dim=-1) + 1e-6)
                    ak_n = F.normalize(ak_in.float(), dim=-1)
                    cos_k = torch.bmm(F.normalize(kr_f.float(), dim=-1), ak_n.transpose(-1, -2))
                    cos_k.masked_fill_(~act_in.unsqueeze(1), -1.0)
                    best_sim_k, best_slot = cos_k.max(dim=-1)
                    matched_av = torch.gather(av_in.float(), 1, best_slot.unsqueeze(-1).expand(-1, -1, hd))
                    sim_v = F.cosine_similarity(v_f32, matched_av, dim=-1)
                    var_score = F.normalize(v_f32, dim=-1).var(dim=-1)
                    score_j = torch.where(
                        has_any,
                        rel_err * (1.0 - best_sim_k.clamp(0.0, 1.0)),
                        var_score,
                    )
                    init_u = torch.where(has_any, rel_err, torch.full_like(rel_err, 0.15))
                return has_any, rel_err, best_sim_k, best_slot, sim_v, score_j, init_u

            has_any_act, rel_err, best_sim_k, best_slot, sim_v, score_j, init_u = _score_masked(ak, av, mass, act)

            # Fill any remaining empty attractor slots per (B * HKV) row before merge/eviction
            n_empty_bh = (~act).sum(dim=-1)
            max_empty = int(n_empty_bh.max().item())
            if max_empty > 0:
                k_fill = min(max_empty, l_tok)
                with torch.no_grad():
                    slot_arange = torch.arange(self.S, device=k_raw.device).unsqueeze(0)
                    empty_pref = torch.where(~act, self.S - slot_arange, -1)
                    empty_slots = torch.topk(empty_pref, k=k_fill, dim=-1, largest=True).indices
                    fill_scores = torch.where(tok_valid, score_j, torch.full_like(score_j, -1e9))
                    top_fill_vals, top_fill_toks = torch.topk(fill_scores, k=k_fill, dim=-1, largest=True)
                    fill_pos = torch.arange(k_fill, device=k_raw.device).unsqueeze(0)
                    do_fill = (fill_pos < n_empty_bh.unsqueeze(-1)) & (top_fill_vals > -1e8)
                    eligible = tok_valid & (~torch.zeros_like(tok_valid).scatter(1, top_fill_toks, do_fill))

                f3 = do_fill.unsqueeze(-1)
                e_hd = empty_slots.unsqueeze(-1).expand(-1, -1, hd)
                t_hd = top_fill_toks.unsqueeze(-1).expand(-1, -1, hd)
                ak = ak.scatter(1, e_hd, torch.where(f3, torch.gather(kr_f, 1, t_hd), torch.gather(ak, 1, e_hd)))
                av = av.scatter(1, e_hd, torch.where(f3, torch.gather(v_f, 1, t_hd), torch.gather(av, 1, e_hd)))
                pm = pm.scatter(1, empty_slots, torch.where(do_fill, torch.gather(p_f, 1, top_fill_toks), torch.gather(pm, 1, empty_slots)))
                mass = mass.scatter(1, empty_slots, torch.where(do_fill, torch.ones_like(top_fill_vals), torch.gather(mass, 1, empty_slots)))
                util = util.scatter(1, empty_slots, torch.where(do_fill, torch.gather(init_u, 1, top_fill_toks), torch.gather(util, 1, empty_slots)))
                act = act.scatter(1, empty_slots, torch.where(do_fill, torch.ones_like(do_fill), torch.gather(act, 1, empty_slots)))

                if not bool(eligible.any()):
                    return ak, av, mass, util, pm, act, m_res, int(act.sum(dim=-1).min().item())
                has_any_act, rel_err, best_sim_k, best_slot, sim_v, score_j, init_u = _score_masked(ak, av, mass, act)

            merge_mask = eligible & has_any_act & (best_sim_k >= self.sim_k) & (sim_v >= self.sim_v)

            m_oh = F.one_hot(best_slot, num_classes=self.S).float() * merge_mask.float().unsqueeze(-1)
            slot_cnt = m_oh.sum(dim=1)
            hit = slot_cnt > 0
            new_m = mass + slot_cnt
            ak = torch.where(
                hit.unsqueeze(-1),
                ((mass.unsqueeze(-1) * ak.float() + torch.bmm(m_oh.transpose(1, 2), kr_f.float())) / new_m.unsqueeze(-1)).to(dt),
                ak,
            )
            av = torch.where(
                hit.unsqueeze(-1),
                ((mass.unsqueeze(-1) * av.float() + torch.bmm(m_oh.transpose(1, 2), v_f.float())) / new_m.unsqueeze(-1)).to(dt),
                av,
            )
            pm = torch.where(
                hit,
                (mass * pm + torch.bmm(m_oh.transpose(1, 2), p_f.unsqueeze(-1)).squeeze(-1)) / new_m,
                pm,
            )
            mass = torch.where(hit, new_m, mass)

            ev_mask = eligible & (~merge_mask)
            with torch.no_grad():
                ak_n2 = F.normalize(ak.float(), dim=-1)
                pair_cos = torch.bmm(ak_n2, ak_n2.transpose(-1, -2))
                pair_cos.masked_fill_(diag_eye | (~act.unsqueeze(1)) | (~act.unsqueeze(2)), -1.0)
                keep_val = torch.where(
                    act,
                    util * (1.0 - pair_cos.max(dim=-1).values.clamp(0.0, 0.98)),
                    torch.full_like(util, -1e9),
                )
                n_scores = torch.where(ev_mask, score_j, torch.full_like(score_j, -1e9))
                k_cand = min(l_tok, self.S)
                top_vals, top_toks = torch.topk(n_scores, k=k_cand, dim=-1, largest=True)
                bot_vals, bot_slots = torch.topk(keep_val, k=k_cand, dim=-1, largest=False)
                win = (top_vals > bot_vals) & (top_vals > -1e8)
                ins_mask = torch.zeros_like(ev_mask).scatter(1, top_toks, win)
                delta_mask = (ev_mask & (~ins_mask)).float()

            k_g = torch.gather(kr_f, 1, top_toks.unsqueeze(-1).expand(-1, -1, hd))
            v_g = torch.gather(v_f, 1, top_toks.unsqueeze(-1).expand(-1, -1, hd))
            p_g = torch.gather(p_f, 1, top_toks)
            u_g = torch.gather(init_u, 1, top_toks)
            k_o = torch.gather(ak, 1, bot_slots.unsqueeze(-1).expand(-1, -1, hd))
            v_o = torch.gather(av, 1, bot_slots.unsqueeze(-1).expand(-1, -1, hd))
            p_o = torch.gather(pm, 1, bot_slots)
            m_o = torch.gather(mass, 1, bot_slots)
            u_o = torch.gather(util, 1, bot_slots)
            a_o = torch.gather(act, 1, bot_slots)
            w3 = win.unsqueeze(-1)
            ak = ak.scatter(1, bot_slots.unsqueeze(-1).expand(-1, -1, hd), torch.where(w3, k_g, k_o))
            av = av.scatter(1, bot_slots.unsqueeze(-1).expand(-1, -1, hd), torch.where(w3, v_g, v_o))
            pm = pm.scatter(1, bot_slots, torch.where(win, p_g, p_o))
            mass = mass.scatter(1, bot_slots, torch.where(win, torch.ones_like(m_o), m_o))
            util = util.scatter(1, bot_slots, torch.where(win, u_g, u_o))
            act = act.scatter(1, bot_slots, torch.where(win, torch.ones_like(a_o), a_o))

            k_ev = F.normalize(kr_f.float(), dim=-1)
            w_ev = b_f.float() * delta_mask
            gram = torch.bmm(k_ev, k_ev.transpose(-1, -2)).abs()
            gersh = torch.bmm(gram, w_ev.unsqueeze(-1)).squeeze(-1).clamp_min(1.0)
            beta_safe = (w_ev / gersh).unsqueeze(-1)
            v_pred = torch.bmm(k_ev, m_res)
            err_v = (v_f.float() - v_pred) * beta_safe
            row_has_rem = eligible.any(dim=-1)
            gam_bh = gam.repeat(b).view(bh, 1, 1)
            gam_eff = torch.where(row_has_rem.view(bh, 1, 1), gam_bh, torch.ones_like(gam_bh))
            m_res = gam_eff * m_res + torch.bmm(k_ev.transpose(1, 2), err_v)
            return ak, av, mass, util, pm, act, m_res, int(act.sum(dim=-1).min().item())

        # Standard unpadded branchless path (used during training and unpadded decode)
        util = util * self.gamma_u
        eligible = torch.ones(bh, l_tok, device=k_raw.device, dtype=torch.bool)

        if n_filled < self.S:
            with torch.no_grad():
                n_empty = self.S - n_filled
                m = min(n_empty, l_tok)
                if n_filled > 0:
                    k_read = apply_rope_pos(
                        ak.view(b, hkv, self.S, hd), pm.view(b, hkv, self.S), inv_freq
                    ).view(bh, self.S, hd)
                    sc = torch.bmm(kp_f, k_read.transpose(-1, -2)).float() * scale + kap * torch.log(
                        mass.clamp_min(1.0)
                    ).unsqueeze(1)
                    sc.masked_fill_(~act.unsqueeze(1), -1e9)
                    v_hat = torch.bmm(F.softmax(sc, dim=-1).to(dt), av).float()
                    v_f32 = v_f.float()
                    rel_err = (v_f32 - v_hat).norm(dim=-1) / (v_f32.norm(dim=-1) + 1e-6)
                    cos_k = torch.bmm(
                        F.normalize(kr_f.float(), dim=-1),
                        F.normalize(ak.float(), dim=-1).transpose(-1, -2),
                    )
                    cos_k.masked_fill_(~act.unsqueeze(1), 0.0)
                    score_j = rel_err * (1.0 - cos_k.max(dim=-1).values.clamp(0.0, 1.0))
                    init_u = rel_err
                else:
                    score_j = F.normalize(v_f.float(), dim=-1).var(dim=-1)
                    init_u = torch.full((bh, l_tok), 0.15, device=k_raw.device, dtype=torch.float32)
                top_toks = torch.topk(score_j, k=m, dim=-1, largest=True).indices
                slot_ids = torch.arange(n_filled, n_filled + m, device=k_raw.device).unsqueeze(0).expand(bh, -1)

            ak = ak.scatter(
                1,
                slot_ids.unsqueeze(-1).expand(-1, -1, hd),
                torch.gather(kr_f, 1, top_toks.unsqueeze(-1).expand(-1, -1, hd)),
            )
            av = av.scatter(
                1,
                slot_ids.unsqueeze(-1).expand(-1, -1, hd),
                torch.gather(v_f, 1, top_toks.unsqueeze(-1).expand(-1, -1, hd)),
            )
            pm = pm.scatter(1, slot_ids, torch.gather(p_f, 1, top_toks))
            mass = mass.scatter(1, slot_ids, torch.ones(bh, m, device=k_raw.device, dtype=torch.float32))
            util = util.scatter(1, slot_ids, torch.gather(init_u, 1, top_toks))
            act = act.scatter(1, slot_ids, torch.ones(bh, m, device=k_raw.device, dtype=torch.bool))
            n_filled = n_filled + m
            if l_tok <= m:
                return ak, av, mass, util, pm, act, m_res, n_filled
            eligible = ~torch.zeros(bh, l_tok, device=k_raw.device, dtype=torch.bool).scatter(1, top_toks, True)

        k_read = apply_rope_pos(
            ak.view(b, hkv, self.S, hd), pm.view(b, hkv, self.S), inv_freq
        ).view(bh, self.S, hd)
        with torch.no_grad():
            sc = torch.bmm(kp_f, k_read.transpose(-1, -2)).float() * scale + kap * torch.log(
                mass.clamp_min(1.0)
            ).unsqueeze(1)
            v_hat = torch.bmm(F.softmax(sc, dim=-1).to(dt), av).float()
            v_f32 = v_f.float()
            rel_err = (v_f32 - v_hat).norm(dim=-1) / (v_f32.norm(dim=-1) + 1e-6)
            ak_n = F.normalize(ak.float(), dim=-1)
            cos_k = torch.bmm(F.normalize(kr_f.float(), dim=-1), ak_n.transpose(-1, -2))
            best_sim_k, best_slot = cos_k.max(dim=-1)
            matched_av = torch.gather(av.float(), 1, best_slot.unsqueeze(-1).expand(-1, -1, hd))
            sim_v = F.cosine_similarity(v_f32, matched_av, dim=-1)
            score_j = rel_err * (1.0 - best_sim_k.clamp(0.0, 1.0))
            merge_mask = eligible & (best_sim_k >= self.sim_k) & (sim_v >= self.sim_v)

        m_oh = F.one_hot(best_slot, num_classes=self.S).float() * merge_mask.float().unsqueeze(-1)
        slot_cnt = m_oh.sum(dim=1)
        k_sum = torch.bmm(m_oh.transpose(1, 2), kr_f.float())
        v_sum = torch.bmm(m_oh.transpose(1, 2), v_f.float())
        p_sum = torch.bmm(m_oh.transpose(1, 2), p_f.unsqueeze(-1)).squeeze(-1)
        hit = slot_cnt > 0
        new_m = mass + slot_cnt
        ak = torch.where(
            hit.unsqueeze(-1), ((mass.unsqueeze(-1) * ak.float() + k_sum) / new_m.unsqueeze(-1)).to(dt), ak
        )
        av = torch.where(
            hit.unsqueeze(-1), ((mass.unsqueeze(-1) * av.float() + v_sum) / new_m.unsqueeze(-1)).to(dt), av
        )
        pm = torch.where(hit, (mass * pm + p_sum) / new_m, pm)
        mass = torch.where(hit, new_m, mass)

        ev_mask = eligible & (~merge_mask)
        with torch.no_grad():
            ak_n2 = F.normalize(ak.float(), dim=-1)
            pair_cos = torch.bmm(ak_n2, ak_n2.transpose(-1, -2)).masked_fill(diag_eye, -1.0)
            keep_val = util * (1.0 - pair_cos.max(dim=-1).values.clamp(0.0, 0.98))
            n_scores = torch.where(ev_mask, score_j, torch.full_like(score_j, -1e9))
            k_cand = min(l_tok, self.S)
            top_vals, top_toks = torch.topk(n_scores, k=k_cand, dim=-1, largest=True)
            bot_vals, bot_slots = torch.topk(keep_val, k=k_cand, dim=-1, largest=False)
            win = (top_vals > bot_vals) & (top_vals > -1e8)
            ins_mask = torch.zeros_like(ev_mask).scatter(1, top_toks, win)
            delta_mask = (ev_mask & (~ins_mask)).float()

        k_g = torch.gather(kr_f, 1, top_toks.unsqueeze(-1).expand(-1, -1, hd))
        v_g = torch.gather(v_f, 1, top_toks.unsqueeze(-1).expand(-1, -1, hd))
        p_g = torch.gather(p_f, 1, top_toks)
        k_o = torch.gather(ak, 1, bot_slots.unsqueeze(-1).expand(-1, -1, hd))
        v_o = torch.gather(av, 1, bot_slots.unsqueeze(-1).expand(-1, -1, hd))
        p_o = torch.gather(pm, 1, bot_slots)
        m_o = torch.gather(mass, 1, bot_slots)
        u_o = torch.gather(util, 1, bot_slots)
        w3 = win.unsqueeze(-1)
        ak = ak.scatter(1, bot_slots.unsqueeze(-1).expand(-1, -1, hd), torch.where(w3, k_g, k_o))
        av = av.scatter(1, bot_slots.unsqueeze(-1).expand(-1, -1, hd), torch.where(w3, v_g, v_o))
        pm = pm.scatter(1, bot_slots, torch.where(win, p_g, p_o))
        mass = mass.scatter(1, bot_slots, torch.where(win, torch.ones_like(m_o), m_o))
        util = util.scatter(1, bot_slots, torch.where(win, torch.gather(rel_err, 1, top_toks), u_o))

        k_ev = F.normalize(kr_f.float(), dim=-1)
        w_ev = b_f.float() * delta_mask
        gram = torch.bmm(k_ev, k_ev.transpose(-1, -2)).abs()
        gersh = torch.bmm(gram, w_ev.unsqueeze(-1)).squeeze(-1).clamp_min(1.0)
        beta_safe = (w_ev / gersh).unsqueeze(-1)
        v_pred = torch.bmm(k_ev, m_res)
        err_v = (v_f.float() - v_pred) * beta_safe
        gam_bh = gam.repeat(b).view(bh, 1, 1)
        m_res = gam_bh * m_res + torch.bmm(k_ev.transpose(1, 2), err_v)
        return ak, av, mass, util, pm, act, m_res, n_filled

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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Exact attention over `[sink, ak, loc]` plus optional cold blocks (all or interval top-k)."""
        b, hkv, g_sz, t_q, hd = qg.shape
        scale = hd ** -0.5
        if cold_k is None or cold_k.shape[2] == 0:
            scores = torch.cat(sp, dim=-1) if len(sp) > 1 else sp[0]
            v_cat = torch.cat(vp, dim=2) if len(vp) > 1 else vp[0]
            probs = F.softmax(scores, dim=-1)
            att_p = probs[..., att_slice[0] : att_slice[1]] if att_slice is not None else None
            y = torch.matmul(probs.to(v_cat.dtype), v_cat.unsqueeze(2)).reshape(b, self.H, t_q, hd)
            return y, att_p

        n_cb, c_sz = cold_k.shape[2], cold_k.shape[3]
        if self.cold_topk <= 0 or n_cb <= self.cold_topk:
            kc = cold_k.reshape(b, hkv, n_cb * c_sz, hd)
            vc = cold_v.reshape(b, hkv, n_cb * c_sz, hd)
            mc = cold_m.reshape(b, 1, 1, 1, n_cb * c_sz)
            s_cold = torch.matmul(qg.float(), kc.float().unsqueeze(2).transpose(-1, -2)) * scale
            s_cold = s_cold.masked_fill(~mc, float("-inf"))
            scores = torch.cat(sp + [s_cold], dim=-1)
            v_cat = torch.cat(vp + [vc], dim=2)
            probs = F.softmax(scores, dim=-1)
            att_p = probs[..., att_slice[0] : att_slice[1]] if att_slice is not None else None
            y = torch.matmul(probs.to(v_cat.dtype), v_cat.unsqueeze(2)).reshape(b, self.H, t_q, hd)
            return y, att_p

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
        att_p = (p_o[..., att_slice[0] : att_slice[1]] * w_o) if att_slice is not None else None
        return y, att_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, T, _ = x.shape
        q_raw, k_raw, v_all_in = self.head_qkv(x)
        q_rope, k_rope = self.rope(q_raw), self.rope(k_raw)
        xp_f = x if (x.dtype == self.beta_proj.weight.dtype or torch.is_autocast_enabled("cuda")) else x.to(self.beta_proj.weight.dtype)
        beta_all = torch.sigmoid(self.beta_proj(xp_f)).transpose(1, 2)
        eta_all = torch.tanh(self.eta_proj(xp_f)).transpose(1, 2).unsqueeze(-1)
        gam = self.gamma_delta()
        inv_freq = self.rope.inv_freq.to(x.device)
        diag_eye = torch.eye(self.S, device=x.device, dtype=torch.bool).unsqueeze(0)
        scale = self.HD ** -0.5

        sink_k = sink_v = ak = av = mass = util = pm = act = m_res = None
        prev_kr = prev_kp = prev_v = prev_pos = prev_b = None
        cold_k = cold_v = cold_m = cold_pos = cold_b = cold_kmin = cold_kmax = None
        n_filled = 0
        ys = []

        for s in range(0, T, self.C):
            e = min(T, s + self.C)
            tc = e - s
            qc_r, qc_p = q_raw[:, :, s:e], q_rope[:, :, s:e]
            kc_r, kc_p = k_raw[:, :, s:e], k_rope[:, :, s:e]
            vc, bc = v_all_in[:, :, s:e], beta_all[:, :, s:e]
            ec = eta_all[:, :, s:e]
            pos_c = torch.arange(s, e, device=x.device, dtype=torch.float32)[None, None, :]

            if prev_kp is not None:
                loc_k = torch.cat([prev_kp, kc_p], dim=2)
                loc_v = torch.cat([prev_v, vc], dim=2)
                prev_len = prev_kp.shape[2]
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

            sp.append(s_loc)
            vp.append(loc_v)
            att_slice = ((sink_k.shape[2] if sink_k is not None else 0), (sink_k.shape[2] if sink_k is not None else 0) + self.S) if ak is not None else None
            yc, att_p = self._attend_with_cold(qg, sp, vp, cold_k, cold_v, cold_m, cold_kmin, cold_kmax, att_slice=att_slice)
            if ak is not None and att_p is not None:
                with torch.no_grad():
                    att_hit = (0.25 * att_p.mean(dim=(2, 3)).reshape(b * self.HKV, self.S))
                    util = torch.maximum(util, att_hit)

            if m_res is not None:
                q_rn = F.normalize(qc_r.view(b, self.HKV, self.G, tc, self.HD).float(), dim=-1)
                ym = torch.einsum("bhgtc,bhcd->bhgtd", q_rn, m_res.view(b, self.HKV, self.HD, self.HD))
                ym = ym * ((1.0 - gam).view(1, self.HKV, 1, 1, 1) / (self.HD ** 0.5))
                ym_n = self.res_norm(ym.reshape(b, self.H, tc, self.HD))
                yc = yc + (ec * ym_n).to(yc.dtype)
            elif e == T:
                yc = yc + 0.0 * (
                    ec.sum() + self.res_norm.weight.sum() + self.kappa.sum() + gam.sum() + bc.sum()
                )

            ys.append(yc)

            if tc == self.C and e < T:
                if prev_kp is not None:
                    p_kr, p_kp, p_v, p_pos, p_b = prev_kr, prev_kp, prev_v, prev_pos, prev_b
                    c_mask = torch.ones(b, 1, self.C, device=x.device, dtype=torch.bool)
                    if sink_k is None:
                        ns = min(self.n_sinks, p_kp.shape[2])
                        sink_k, sink_v = p_kp[:, :, :ns], p_v[:, :, :ns]
                        c_mask[:, :, :ns] = False
                        p_kr_t, p_kp_t, p_v_t, p_pos_t, p_b_t = (
                            p_kr[:, :, ns:],
                            p_kp[:, :, ns:],
                            p_v[:, :, ns:],
                            p_pos[:, :, ns:],
                            p_b[:, :, ns:],
                        )
                    else:
                        p_kr_t, p_kp_t, p_v_t, p_pos_t, p_b_t = p_kr, p_kp, p_v, p_pos, p_b

                    if self.is_cold_armed and self.max_cold_blocks > 0:
                        cold_k, cold_v, cold_m, cold_pos, cold_b, cold_kmin, cold_kmax, popped = self._push_cold_block(
                            cold_k, cold_v, cold_m, cold_pos, cold_b, cold_kmin, cold_kmax,
                            p_kp, p_v, c_mask, p_pos.expand(b, 1, self.C), p_b,
                        )
                        if popped is not None:
                            ok_p, ok_v, ok_m, ok_pos, ok_b = popped
                            ok_kr = apply_rope_pos(ok_p, -ok_pos.expand(b, self.HKV, self.C), inv_freq)
                            tok_v_in = None if bool(ok_m.all()) else ok_m.expand(b, self.HKV, -1).reshape(b * self.HKV, -1)
                            ak_d = ak.detach() if ak is not None else None
                            av_d = av.detach() if av is not None else None
                            pm_d = pm.detach() if pm is not None else None
                            mr_d = m_res.detach() if m_res is not None else None
                            ak, av, mass, util, pm, act, m_res, n_filled = self._consolidate_functional(
                                ak_d, av_d, mass, util, pm_d, act, mr_d,
                                ok_kr, ok_p, ok_v, ok_pos, ok_b, inv_freq, diag_eye, gam,
                                n_filled=n_filled, tok_valid=tok_v_in,
                            )
                    else:
                        ak_d = ak.detach() if ak is not None else None
                        av_d = av.detach() if av is not None else None
                        pm_d = pm.detach() if pm is not None else None
                        mr_d = m_res.detach() if m_res is not None else None
                        ak, av, mass, util, pm, act, m_res, n_filled = self._consolidate_functional(
                            ak_d, av_d, mass, util, pm_d, act, mr_d,
                            p_kr_t, p_kp_t, p_v_t, p_pos_t, p_b_t, inv_freq, diag_eye, gam,
                            n_filled=n_filled,
                        )
                prev_kr, prev_kp, prev_v, prev_pos, prev_b = kc_r, kc_p, vc, pos_c, bc

        y_out = torch.cat(ys, dim=2)
        return self.o_proj(y_out.transpose(1, 2).reshape(b, T, self.H * self.HD))


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
    slot_cur_t: torch.Tensor,
    zero_mask: torch.Tensor,
    buf_k: torch.Tensor,
    buf_v: torch.Tensor,
    mask_buf: torch.Tensor,
    util_acc: torch.Tensor,
    cur_kr: torch.Tensor,
    cur_pos: torch.Tensor,
    cur_b: torch.Tensor,
    m_res: torch.Tensor,
    res_scale: torch.Tensor,
) -> torch.Tensor:
    """Zero-cat single-token decode step across unified `[sink | ak_eff | half0 | half1 | cold]` buffers."""
    b = x.shape[0]
    h = norm1(x)
    q_r, k_r, v_n = ly.head_qkv(h)
    rot_dim = cos.shape[-1]
    q_p = apply_rope_cos_sin(q_r, cos, sin, rot_dim)
    k_p = apply_rope_cos_sin(k_r, cos, sin, rot_dim)
    beta_n = torch.sigmoid(ly.beta_proj(h)).transpose(1, 2)
    eta_n = torch.tanh(ly.eta_proj(h)).transpose(1, 2).unsqueeze(-1)

    buf_k.index_copy_(2, slot_buf_t, k_p.to(buf_k.dtype))
    buf_v.index_copy_(2, slot_buf_t, v_n.to(buf_v.dtype))
    mask_buf.index_copy_(3, slot_buf_t, zero_mask)
    cur_kr.index_copy_(2, slot_cur_t, k_r.to(cur_kr.dtype))
    cur_pos.index_copy_(2, slot_cur_t, pos_n)
    cur_b.index_copy_(2, slot_cur_t, beta_n.to(cur_b.dtype))

    qg = q_p.view(b, ly.HKV, ly.G, 1, ly.HD)
    n_tot = buf_k.shape[2]
    if x.is_cuda and n_tot > 512:
        scores = torch.matmul(qg, buf_k.unsqueeze(2).transpose(-1, -2)).float() * (ly.HD ** -0.5)
    else:
        scores = torch.matmul(qg.float(), buf_k.float().unsqueeze(2).transpose(-1, -2)) * (ly.HD ** -0.5)
    scores = scores + mask_buf.view(b, ly.HKV, 1, 1, n_tot)
    probs = F.softmax(scores, dim=-1)
    util_acc.add_(probs[..., ly.n_sinks : ly.n_sinks + ly.S].mean(dim=2).reshape(b * ly.HKV, ly.S))
    y = torch.matmul(probs.to(buf_v.dtype), buf_v.unsqueeze(2)).reshape(b, ly.H, 1, ly.HD)

    q_rn = F.normalize(q_r.view(b, ly.HKV, ly.G, 1, ly.HD).float(), dim=-1)
    ym = torch.matmul(q_rn, m_res.unsqueeze(2)) * res_scale
    ym_n = ly.res_norm(ym.reshape(b, ly.H, 1, ly.HD))
    y = y + (eta_n * ym_n).to(y.dtype)

    x = x + ly.o_proj(y.transpose(1, 2).reshape(b, 1, ly.H * ly.HD))
    return x + mlp(norm2(x))


class _Bolt3GraphRunner:
    """CUDA Graph runner for steady-state (`cur_len < C`) single-token Bolt-3 decode."""

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
        ly0: Bolt3Attn = model.blocks[0].attn
        self.C = ly0.C
        self.W1 = ly0.n_sinks + ly0.S + 2 * ly0.C

        self.states: list[torch.Tensor] = []
        self.buf_k: list[torch.Tensor] = []
        self.buf_v: list[torch.Tensor] = []
        self.mask_buf: list[torch.Tensor] = []
        self.util_acc: list[torch.Tensor] = []
        self.cur_kr: list[torch.Tensor] = []
        self.cur_pos: list[torch.Tensor] = []
        self.cur_b: list[torch.Tensor] = []
        self.res_scales: list[torch.Tensor] = []

        for li, blk in enumerate(model.blocks):
            ly: Bolt3Attn = blk.attn
            c_cap = self.cold_caps[li]
            n_tot = ly.n_sinks + ly.S + 2 * ly.C + c_cap * ly.C
            self.states.append(torch.zeros(batch_size, ly.HKV, ly.HD, ly.HD, device=device, dtype=torch.float32))
            self.buf_k.append(torch.zeros(batch_size, ly.HKV, n_tot, ly.HD, device=device, dtype=dtype))
            self.buf_v.append(torch.zeros(batch_size, ly.HKV, n_tot, ly.HD, device=device, dtype=dtype))
            self.mask_buf.append(torch.zeros(batch_size, ly.HKV, 1, n_tot, device=device, dtype=torch.float32))
            self.util_acc.append(torch.zeros(batch_size * ly.HKV, ly.S, device=device, dtype=torch.float32))
            self.cur_kr.append(torch.zeros(batch_size, ly.HKV, ly.C, ly.HD, device=device, dtype=dtype))
            self.cur_pos.append(torch.zeros(batch_size, 1, ly.C, device=device, dtype=torch.float32))
            self.cur_b.append(torch.zeros(batch_size, ly.HKV, ly.C, device=device, dtype=dtype))
            self.res_scales.append(((1.0 - ly.gamma_delta()).view(1, ly.HKV, 1, 1, 1) / (ly.HD ** 0.5)).to(device))

        self.g_tok = torch.zeros(batch_size, 1, device=device, dtype=torch.long)
        self.g_pos = torch.zeros(batch_size, 1, 1, device=device, dtype=torch.float32)
        self.g_slot_buf = torch.zeros(1, device=device, dtype=torch.long)
        self.g_slot_cur = torch.zeros(1, device=device, dtype=torch.long)
        self.g_zero_mask = torch.zeros(batch_size, ly0.HKV, 1, 1, device=device, dtype=torch.float32)
        self.g_out_h = torch.zeros(batch_size, 1, int(model.config["dim"]), device=device, dtype=dtype)
        self.slot_idxs = [torch.tensor([i], device=device, dtype=torch.long) for i in range(max(self.W1, self.C))]
        self.inv_freq = ly0.rope.inv_freq.to(device)
        use_bf16_auto = dtype == torch.bfloat16

        def _forward_steady():
            with torch.autocast("cuda", torch.bfloat16, enabled=use_bf16_auto):
                x = self.model.embed_tokens(self.g_tok)
                freqs = self.g_pos.unsqueeze(-1) * self.inv_freq.view(1, 1, 1, -1)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos().to(x.dtype)
                sin = emb.sin().to(x.dtype)
                for li, blk in enumerate(self.model.blocks):
                    x = _steady_block_fast(
                        blk.attn,
                        blk.norm1,
                        blk.norm2,
                        blk.mlp,
                        x,
                        cos,
                        sin,
                        self.g_pos,
                        self.g_slot_buf,
                        self.g_slot_cur,
                        self.g_zero_mask,
                        self.buf_k[li],
                        self.buf_v[li],
                        self.mask_buf[li],
                        self.util_acc[li],
                        self.cur_kr[li],
                        self.cur_pos[li],
                        self.cur_b[li],
                        self.states[li],
                        self.res_scales[li],
                    )
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
            self.buf_k,
            self.buf_v,
            self.mask_buf,
            self.util_acc,
            self.cur_kr,
            self.cur_pos,
            self.cur_b,
            self.res_scales,
            self.slot_idxs,
        ):
            lst.clear()
        self.model = None

    @torch.no_grad()
    def replay(
        self, tokens: torch.Tensor, pos_start: int, pos_ids: torch.Tensor | None, slot_buf: int, slot_cur: int
    ) -> torch.Tensor:
        self.g_tok.copy_(tokens)
        if pos_ids is None:
            self.g_pos.fill_(float(pos_start))
        else:
            self.g_pos.copy_(pos_ids.view(self.batch_size, 1, 1))
        self.g_slot_buf.copy_(self.slot_idxs[slot_buf])
        self.g_slot_cur.copy_(self.slot_idxs[slot_cur])
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
        self._all_bolt3 = all(isinstance(blk.attn, Bolt3Attn) for blk in model.blocks)
        inv0 = model.blocks[0].attn.rope.inv_freq
        self._shared_rope = self._all_bolt3 and all(
            blk.attn.rope.rotary_dim == model.blocks[0].attn.rope.rotary_dim
            and torch.equal(blk.attn.rope.inv_freq, inv0)
            for blk in model.blocks
        )

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
                    + 6 * ly.C * ly.HKV * ly.HD * 2
                    + 2 * ly.n_sinks * ly.HKV * ly.HD * 2
                )
                if ly.is_cold_armed and ly.max_cold_blocks > 0:
                    per_tok += 2 * ly.HKV * ly.HD * 2
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
                    "prev_kr": None,
                    "prev_kp": None,
                    "prev_v": None,
                    "prev_pos": None,
                    "prev_b": None,
                    "prev_valid": None,
                    "prev_has_pad": False,
                    "has_prev": False,
                    "cur_kr": None,
                    "cur_kp": None,
                    "cur_v": None,
                    "cur_pos": None,
                    "cur_b": None,
                    "cur_valid": None,
                    "cur_has_pad": False,
                    "cur_len": 0,
                    "cold_k_blk": None,
                    "cold_v_blk": None,
                    "cold_mask": None,
                    "cold_pos": None,
                    "cold_b": None,
                    "cold_kmin": None,
                    "cold_kmax": None,
                    "cold_cap_blks": 0,
                    "buf_k": None,
                    "buf_v": None,
                    "mask_buf": None,
                    "cur_half": 0,
                    "_synced_ids": None,
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
                for k in list(c.keys()):
                    if k in ("n_filled", "cur_len", "cur_half", "cold_cap_blks"):
                        c[k] = 0
                    elif k in ("prev_has_pad", "cur_has_pad", "has_prev"):
                        c[k] = False
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
        probs = F.softmax(scores / temperature, dim=-1)
        if top_p < 1.0:
            sp, si = torch.sort(probs, descending=True, dim=-1)
            cp = torch.cumsum(sp, dim=-1)
            sp[cp - sp > top_p] = 0.0
            sp.div_(sp.sum(dim=-1, keepdim=True).clamp_min(1e-12))
            return si.gather(-1, torch.multinomial(sp, 1))
        tv, ti = torch.topk(scores / temperature, min(top_k, scores.size(-1)), dim=-1)
        return ti.gather(-1, torch.multinomial(F.softmax(tv, -1), 1))

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
        key = (ly.n_sinks + ly.S + 2 * ly.C, device)
        arr = self._slot_t_cache.get(key)
        if arr is None:
            n_tot = max(key[0], ly.C)
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
            id(sc.get("prev_kp")),
            id(sc.get("ak")),
            id(sc.get("cur_kp")),
            cold_cap_blks,
            id(sc.get("cold_k_blk")),
        )
        if sc.get("buf_k") is not None and sc.get("_synced_ids") == sig:
            return
        w0 = ly.n_sinks + ly.S
        w1 = w0 + 2 * ly.C
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

        if sc.get("util_acc") is None or sc["util_acc"].shape != (b * ly.HKV, ly.S):
            sc["util_acc"] = torch.zeros(b * ly.HKV, ly.S, device=device, dtype=torch.float32)

        cur_half = int(sc.get("cur_half", 0))
        prev_half = 1 - cur_half
        p_lo, c_lo = w0 + prev_half * ly.C, w0 + cur_half * ly.C

        if sc["prev_kp"] is not None:
            p_len = sc["prev_kp"].shape[2]
            buf_k[:, :, p_lo : p_lo + p_len] = sc["prev_kp"].to(device=device, dtype=dt)
            buf_v[:, :, p_lo : p_lo + p_len] = sc["prev_v"].to(device=device, dtype=dt)
            if sc.get("prev_valid") is not None:
                pv = sc["prev_valid"].view(b, 1, 1, p_len).expand(b, ly.HKV, 1, p_len)
                mask_buf[:, :, :, p_lo : p_lo + p_len] = torch.where(pv, 0.0, float("-inf"))
            else:
                sc["prev_valid"] = torch.ones(b, 1, p_len, device=device, dtype=torch.bool)
                mask_buf[:, :, :, p_lo : p_lo + p_len] = 0.0
            sc["has_prev"] = True
            sc["prev_kp"] = buf_k[:, :, p_lo : p_lo + ly.C]
            sc["prev_v"] = buf_v[:, :, p_lo : p_lo + ly.C]
        else:
            sc["has_prev"] = False

        cl = int(sc.get("cur_len", 0))
        if sc.get("cur_kr") is None or sc["cur_kr"].shape != (b, ly.HKV, ly.C, ly.HD) or sc["cur_kr"].dtype != dt:
            sc["cur_kr"] = torch.empty(b, ly.HKV, ly.C, ly.HD, device=device, dtype=dt)
            sc["cur_pos"] = torch.zeros(b, 1, ly.C, device=device, dtype=torch.float32)
            sc["cur_b"] = torch.empty(b, ly.HKV, ly.C, device=device, dtype=dt)
            sc["cur_valid"] = torch.ones(b, 1, ly.C, device=device, dtype=torch.bool)
        if cl > 0 and sc["cur_kp"] is not None:
            buf_k[:, :, c_lo : c_lo + cl] = sc["cur_kp"][:, :, :cl].to(device=device, dtype=dt)
            buf_v[:, :, c_lo : c_lo + cl] = sc["cur_v"][:, :, :cl].to(device=device, dtype=dt)
            if sc.get("cur_valid") is not None:
                cv = sc["cur_valid"][:, :, :cl].view(b, 1, 1, cl).expand(b, ly.HKV, 1, cl)
                mask_buf[:, :, :, c_lo : c_lo + cl] = torch.where(cv, 0.0, float("-inf"))
            else:
                mask_buf[:, :, :, c_lo : c_lo + cl] = 0.0
        sc["cur_kp"] = buf_k[:, :, c_lo : c_lo + ly.C]
        sc["cur_v"] = buf_v[:, :, c_lo : c_lo + ly.C]

        if cold_cap_blks > 0 and sc.get("cold_k_blk") is not None:
            n_tok_c = int(sc["cold_k_blk"].shape[2]) * ly.C
            buf_k[:, :, w1 : w1 + n_tok_c] = sc["cold_k_blk"].reshape(b, ly.HKV, n_tok_c, ly.HD).to(device=device, dtype=dt)
            buf_v[:, :, w1 : w1 + n_tok_c] = sc["cold_v_blk"].reshape(b, ly.HKV, n_tok_c, ly.HD).to(device=device, dtype=dt)
            cm = sc["cold_mask"].reshape(b, 1, 1, n_tok_c).expand(b, ly.HKV, 1, n_tok_c)
            mask_buf[:, :, :, w1 : w1 + n_tok_c] = torch.where(cm, 0.0, float("-inf"))

        sc["cold_cap_blks"] = cold_cap_blks
        sc["buf_k"], sc["buf_v"], sc["mask_buf"] = buf_k, buf_v, mask_buf
        sc["_synced_ids"] = (
            id(buf_k),
            id(sc.get("sink_k")),
            id(sc.get("prev_kp")),
            id(sc.get("ak")),
            id(sc.get("cur_kp")),
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
        if sc.get("util_acc") is not None and sc["util"] is not None:
            sc["util"] = torch.maximum(sc["util"], 0.25 * (sc["util_acc"] / float(ly.C)))
            sc["util_acc"].zero_()

        has_prev = sc.get("has_prev", False) if unified else (sc["prev_kp"] is not None)
        cold_bucket_ok = True
        if has_prev:
            if unified:
                prev_half = 1 - int(sc.get("cur_half", 0))
                p_lo = ly.n_sinks + ly.S + prev_half * ly.C
                p_kp = sc["buf_k"][:, :, p_lo : p_lo + ly.C]
                p_v = sc["buf_v"][:, :, p_lo : p_lo + ly.C]
                p_kr, p_pos, p_b, p_val = sc["prev_kr"], sc["prev_pos"], sc["prev_b"], sc["prev_valid"]
            else:
                p_kr, p_kp, p_v, p_pos, p_b, p_val = (
                    sc["prev_kr"],
                    sc["prev_kp"],
                    sc["prev_v"],
                    sc["prev_pos"],
                    sc["prev_b"],
                    sc["prev_valid"],
                )

            p_kp_f, p_v_f, p_pos_f, p_b_f = p_kp, p_v, p_pos, p_b
            all_valid_prev = bool(p_val.all())
            if sc["sink_k"] is None and all_valid_prev:
                ns = min(ly.n_sinks, p_kp.shape[2])
                sc["sink_k"] = p_kp[:, :, :ns].clone()
                sc["sink_v"] = p_v[:, :, :ns].clone()
                sc["sink_valid"] = torch.ones(b, 1, ns, device=device, dtype=torch.bool)
                sc["sink_filled"] = torch.full((b,), ns, device=device, dtype=torch.long)
                c_mask_f = torch.ones(b, 1, p_kp.shape[2], device=device, dtype=torch.bool)
                c_mask_f[:, :, :ns] = False
                p_kr, p_kp, p_v, p_pos, p_b = (
                    p_kr[:, :, ns:],
                    p_kp[:, :, ns:],
                    p_v[:, :, ns:],
                    p_pos[:, :, ns:],
                    p_b[:, :, ns:],
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
                if popped is not None:
                    ok_p, ok_v, ok_m, ok_pos, ok_b = popped
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
                    )
                )
                states[li].copy_(m_out.view(b, ly.HKV, ly.HD, ly.HD))
                self._refresh_attractor_cache(sc, ly, b, inv_freq)

            if unified:
                ns = sc["sink_k"].shape[2]
                w0 = ly.n_sinks + ly.S
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
                        w1 = w0 + 2 * ly.C
                        n_tok_c = int(sc["cold_k_blk"].shape[2]) * ly.C
                        sc["buf_k"][:, :, w1 : w1 + n_tok_c] = sc["cold_k_blk"].reshape(b, ly.HKV, n_tok_c, ly.HD)
                        sc["buf_v"][:, :, w1 : w1 + n_tok_c] = sc["cold_v_blk"].reshape(b, ly.HKV, n_tok_c, ly.HD)
                        cm = sc["cold_mask"].reshape(b, 1, 1, n_tok_c).expand(b, ly.HKV, 1, n_tok_c)
                        sc["mask_buf"][:, :, :, w1 : w1 + n_tok_c] = torch.where(cm, 0.0, float("-inf"))

        if unified:
            if sc["prev_kr"] is None or sc["prev_kr"].shape != sc["cur_kr"].shape:
                sc["prev_kr"] = sc["cur_kr"].clone()
                sc["prev_pos"] = sc["cur_pos"].clone()
                sc["prev_b"] = sc["cur_b"].clone()
                sc["prev_valid"] = sc["cur_valid"].clone()
            else:
                sc["prev_kr"].copy_(sc["cur_kr"])
                sc["prev_pos"].copy_(sc["cur_pos"])
                sc["prev_b"].copy_(sc["cur_b"])
                sc["prev_valid"].copy_(sc["cur_valid"])
            sc["has_prev"] = True
            prev_half = int(sc.get("cur_half", 0))
            sc["cur_half"] = 1 - prev_half
            w0 = ly.n_sinks + ly.S
            p_lo, c_lo = w0 + prev_half * ly.C, w0 + sc["cur_half"] * ly.C
            sc["prev_kp"] = sc["buf_k"][:, :, p_lo : p_lo + ly.C]
            sc["prev_v"] = sc["buf_v"][:, :, p_lo : p_lo + ly.C]
            sc["cur_kp"] = sc["buf_k"][:, :, c_lo : c_lo + ly.C]
            sc["cur_v"] = sc["buf_v"][:, :, c_lo : c_lo + ly.C]
            sc["mask_buf"][:, :, :, c_lo : c_lo + ly.C].fill_(float("-inf"))
            sc["cur_valid"].fill_(True)
            sc["prev_has_pad"], sc["cur_has_pad"] = sc.get("cur_has_pad", False), False
            sc["cur_len"] = 0
            if cold_bucket_ok:
                sc["_synced_ids"] = (
                    id(sc["buf_k"]),
                    id(sc.get("sink_k")),
                    id(sc.get("prev_kp")),
                    id(sc.get("ak")),
                    id(sc.get("cur_kp")),
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
                        id(sc.get("prev_kp")),
                        sc.get("cold_cap_blks", 0),
                        id(sc.get("cold_k_blk")),
                    )
            else:
                sc["_synced_ids"] = None
                sc["_runner_id"] = None
        else:
            sc["prev_kr"], sc["cur_kr"] = sc["cur_kr"], sc["prev_kr"]
            sc["prev_kp"], sc["cur_kp"] = sc["cur_kp"], sc["prev_kp"]
            sc["prev_v"], sc["cur_v"] = sc["cur_v"], sc["prev_v"]
            sc["prev_pos"], sc["cur_pos"] = sc["cur_pos"], sc["prev_pos"]
            sc["prev_b"], sc["cur_b"] = sc["cur_b"], sc["prev_b"]
            sc["prev_valid"], sc["cur_valid"] = sc["cur_valid"], sc["prev_valid"]
            if sc["cur_valid"] is not None:
                sc["cur_valid"].fill_(True)
            sc["prev_has_pad"], sc["cur_has_pad"] = sc.get("cur_has_pad", False), False
            sc["cur_len"] = 0
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
            gam, res_scale = self._get_gam_scale(ly, device)

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
                slot_buf = ly.n_sinks + ly.S + int(sc["cur_half"]) * ly.C + len_before
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
                    self._get_slot_idx(ly, len_before, device),
                    self._get_zero_mask(b, ly.HKV, device),
                    sc["buf_k"],
                    sc["buf_v"],
                    sc["mask_buf"],
                    sc["util_acc"],
                    sc["cur_kr"],
                    sc["cur_pos"],
                    sc["cur_b"],
                    states[li],
                    res_scale,
                )
                new_len = len_before + 1
                sc["cur_len"] = new_len
                if new_len >= ly.C:
                    self._consolidate_layer(li, ly, sc, states, b, device, inv_freq, gam, unified=True)
                return x

            # Multi-token prefill / valid_mask causal path
            sc["_synced_ids"] = None
            sc["buf_k"] = None
            h = blk.norm1(x)
            q_r, k_r, v_n = ly.head_qkv(h)
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
            scale = ly.HD ** -0.5

            if sc["cur_kr"] is None:
                sc["cur_kr"] = torch.empty(b, ly.HKV, ly.C, ly.HD, device=device, dtype=k_r.dtype)
                sc["cur_kp"] = torch.empty_like(sc["cur_kr"])
                sc["cur_v"] = torch.empty_like(sc["cur_kr"])
                sc["cur_pos"] = torch.zeros(b, 1, ly.C, device=device, dtype=torch.float32)
                sc["cur_b"] = torch.empty(b, ly.HKV, ly.C, device=device, dtype=beta_n.dtype)
                sc["cur_valid"] = torch.ones(b, 1, ly.C, device=device, dtype=torch.bool)
                sc["cur_len"] = 0
            elif sc["cur_kp"] is None:
                sc["cur_kp"] = torch.empty(b, ly.HKV, ly.C, ly.HD, device=device, dtype=k_p.dtype)
                sc["cur_v"] = torch.empty_like(sc["cur_kp"])

            len_before = sc["cur_len"]
            new_len = len_before + t_s
            sc["cur_kr"][:, :, len_before:new_len] = k_r
            sc["cur_kp"][:, :, len_before:new_len] = k_p
            sc["cur_v"][:, :, len_before:new_len] = v_n
            sc["cur_pos"][:, :, len_before:new_len] = pos_n
            sc["cur_b"][:, :, len_before:new_len] = beta_n
            if valid_mask is not None:
                sc["cur_valid"][:, :, len_before:new_len] = valid_mask.unsqueeze(1)
                if not bool(valid_mask.all()):
                    sc["cur_has_pad"] = True
            else:
                sc["cur_valid"][:, :, len_before:new_len] = True
            sc["cur_len"] = new_len

            cur_kp_v = sc["cur_kp"][:, :, :new_len]
            cur_v_v = sc["cur_v"][:, :, :new_len]
            cur_val_v = sc["cur_valid"][:, :, :new_len]
            if sc["prev_kp"] is not None:
                loc_k = torch.cat([sc["prev_kp"], cur_kp_v], dim=2)
                loc_v = torch.cat([sc["prev_v"], cur_v_v], dim=2)
                loc_valid = torch.cat([sc["prev_valid"], cur_val_v], dim=2)
                prev_len = sc["prev_kp"].shape[2]
            else:
                loc_k, loc_v, loc_valid, prev_len = cur_kp_v, cur_v_v, cur_val_v, 0

            qg = q_p.view(b, ly.HKV, ly.G, t_s, ly.HD)
            s_loc = torch.matmul(qg.float(), loc_k.float().unsqueeze(2).transpose(-1, -2)) * scale
            has_pad = sc.get("prev_has_pad", False) or sc.get("cur_has_pad", False)
            if t_s > 1 or has_pad:
                qi = torch.arange(t_s, device=device)[:, None]
                kj = torch.arange(loc_k.shape[2], device=device)[None, :]
                cm = (kj <= (prev_len + len_before + qi)).view(1, 1, 1, t_s, -1)
                if has_pad:
                    q_ok = valid_mask if valid_mask is not None else torch.ones(b, t_s, device=device, dtype=torch.bool)
                    self_pos = (kj == (prev_len + len_before + qi)).view(1, 1, 1, t_s, -1)
                    cm = (cm & loc_valid.view(b, 1, 1, 1, -1)) | (self_pos & (~q_ok).view(b, 1, 1, t_s, 1))
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

            sp.append(s_loc)
            vp.append(loc_v)
            att_off = sc["sink_k"].shape[2] if sc["sink_k"] is not None else 0
            att_slice = (att_off, att_off + ly.S) if sc["ak"] is not None else None
            y, att_p = ly._attend_with_cold(
                qg,
                sp,
                vp,
                sc.get("cold_k_blk"),
                sc.get("cold_v_blk"),
                sc.get("cold_mask"),
                sc.get("cold_kmin"),
                sc.get("cold_kmax"),
                att_slice=att_slice,
            )

            if sc["ak"] is not None and att_p is not None:
                if valid_mask is not None:
                    att_p = att_p * valid_mask.view(b, 1, 1, t_s, 1).float()
                att_sum = att_p.mean(dim=2).sum(dim=2).reshape(b * ly.HKV, ly.S)
                if sc.get("util_acc") is None:
                    sc["util_acc"] = att_sum
                else:
                    sc["util_acc"] = sc["util_acc"] + att_sum

            if states[li] is not None:
                q_rn = F.normalize(q_r.view(b, ly.HKV, ly.G, t_s, ly.HD).float(), dim=-1)
                ym = torch.matmul(q_rn, states[li].unsqueeze(2)) * res_scale
                ym_n = ly.res_norm(ym.reshape(b, ly.H, t_s, ly.HD))
                y = y + (eta_n * ym_n).to(y.dtype)

            x = x + ly.o_proj(y.transpose(1, 2).reshape(b, t_s, ly.H * ly.HD))

            if new_len >= ly.C:
                self._consolidate_layer(li, ly, sc, states, b, device, inv_freq, gam, unified=False)
        else:
            h = blk.norm1(x)
            q, k, v = ly.head_qkv(h)
            q, k = ly.rope(q, pos_start=pos_start), ly.rope(k, pos_start=pos_start)
            n_prev = kv_caches[li].n
            pk, pv = kv_caches[li].append(k, v)
            if t_s > 1:
                qi = torch.arange(t_s, device=device)[:, None]
                kj = torch.arange(n_prev + t_s, device=device)[None, :]
                cmask = (kj <= (n_prev + qi)).view(1, 1, t_s, n_prev + t_s)
                try:
                    y = F.scaled_dot_product_attention(
                        q, pk.to(q.dtype), pv.to(q.dtype), attn_mask=cmask, enable_gqa=(ly.G > 1)
                    )
                except TypeError:
                    y = F.scaled_dot_product_attention(
                        q, repeat_kv(pk.to(q.dtype), ly.G), repeat_kv(pv.to(q.dtype), ly.G), attn_mask=cmask
                    )
            else:
                try:
                    y = F.scaled_dot_product_attention(q, pk.to(q.dtype), pv.to(q.dtype), enable_gqa=(ly.G > 1))
                except TypeError:
                    y = F.scaled_dot_product_attention(
                        q, repeat_kv(pk.to(q.dtype), ly.G), repeat_kv(pv.to(q.dtype), ly.G)
                    )
            x = x + ly.o_proj(y.transpose(1, 2).reshape(b, t_s, ly.H * ly.HD))

        return x + blk.mlp(blk.norm2(x))

    def _can_use_graph(
        self, tokens: torch.Tensor, valid_mask, kv_caches, dtype: torch.dtype, cold_caps: tuple[int, ...]
    ) -> bool:
        if not (tokens.is_cuda and tokens.shape[1] == 1 and valid_mask is None and self.use_cuda_graph):
            return False
        if not (self._all_bolt3 and self._shared_rope and len(kv_caches) == len(self.model.blocks)):
            return False
        if any(blk.attn.is_cold_armed and blk.attn.max_cold_blocks > 0 and blk.attn.cold_topk > 0 for blk in self.model.blocks):
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
                if cap > 0
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
            ly: Bolt3Attn = blk.attn
            cold_cap = runner.cold_caps[li]
            expected_sig = (
                rid,
                id(states[li]),
                id(sc.get("buf_k")),
                id(sc.get("sink_k")),
                id(sc.get("ak")),
                id(sc.get("prev_kp")),
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
            if runner.cur_kr[li] is not sc["cur_kr"]:
                runner.cur_kr[li].copy_(sc["cur_kr"])
                runner.cur_pos[li].copy_(sc["cur_pos"])
                runner.cur_b[li].copy_(sc["cur_b"])
                sc["cur_kr"] = runner.cur_kr[li]
                sc["cur_pos"] = runner.cur_pos[li]
                sc["cur_b"] = runner.cur_b[li]

            w0 = ly.n_sinks + ly.S
            cur_half = int(sc.get("cur_half", 0))
            p_lo = w0 + (1 - cur_half) * ly.C
            c_lo = w0 + cur_half * ly.C
            if sc.get("has_prev", False):
                sc["prev_kp"] = sc["buf_k"][:, :, p_lo : p_lo + ly.C]
                sc["prev_v"] = sc["buf_v"][:, :, p_lo : p_lo + ly.C]
            sc["cur_kp"] = sc["buf_k"][:, :, c_lo : c_lo + ly.C]
            sc["cur_v"] = sc["buf_v"][:, :, c_lo : c_lo + ly.C]
            sc["_synced_ids"] = (
                id(sc["buf_k"]),
                id(sc.get("sink_k")),
                id(sc.get("prev_kp")),
                id(sc.get("ak")),
                id(sc.get("cur_kp")),
                cold_cap,
                id(sc.get("cold_k_blk")),
            )
            sc["_runner_id"] = (
                rid,
                id(states[li]),
                id(sc["buf_k"]),
                id(sc.get("sink_k")),
                id(sc.get("ak")),
                id(sc.get("prev_kp")),
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
                    self._cold_cap_for(blk.attn, kv_caches[li]) if isinstance(kv_caches[li], dict) else 0
                    for li, blk in enumerate(m.blocks)
                )
                if len(kv_caches) == len(m.blocks)
                else ()
            )
            if self._can_use_graph(tokens, valid_mask, kv_caches, dt, cold_caps):
                len0 = int(kv_caches[0].get("cur_len", 0))
                half0 = int(kv_caches[0].get("cur_half", 0))
                if all(c.get("cur_len", 0) == len0 and c.get("cur_half", 0) == half0 for c in kv_caches):
                    runner = self._get_graph_runner(b, tokens.device, dt, cold_caps)
                    self._bind_graph_state(self, runner, states, kv_caches, b, tokens.device, dt)
                    ly0: Bolt3Attn = m.blocks[0].attn
                    slot_buf = ly0.n_sinks + ly0.S + half0 * ly0.C + len0
                    out_h = runner.replay(tokens, pos_start, pos_ids, slot_buf, len0)
                    new_len = len0 + 1
                    for sc in kv_caches:
                        sc["cur_len"] = new_len
                    if new_len >= ly0.C:
                        inv_freq = ly0.rope.inv_freq.to(tokens.device)
                        for li, blk in enumerate(m.blocks):
                            ly = blk.attn
                            gam, _ = self._get_gam_scale(ly, tokens.device)
                            self._consolidate_layer(li, ly, kv_caches[li], states, b, tokens.device, inv_freq, gam, unified=True)
                    return out_h, states, kv_caches

            with torch.autocast("cuda", torch.bfloat16, enabled=use_bf16):
                x = m.embed_tokens(tokens)
                rope_cs = None
                if self._shared_rope:
                    rope_cs = m.blocks[0].attn.rope.cos_sin(pos_start, 1, tokens.device, x.dtype, pos_ids=pos_ids)
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
                if isinstance(c, dict):
                    cur_len = c.get("cur_len", 0)
                    break
            take = min(t_tot - s, self.C - cur_len)
            p_slice = pos_ids[:, s : s + take] if pos_ids is not None else None
            v_slice = valid_mask[:, s : s + take] if valid_mask is not None else None
            with torch.autocast("cuda", torch.bfloat16, enabled=use_bf16):
                x = m.embed_tokens(tokens[:, s : s + take])
                rope_cs = None
                if self._shared_rope:
                    rope_cs = m.blocks[0].attn.rope.cos_sin(pos, take, tokens.device, x.dtype, pos_ids=p_slice)
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
                gam = blk.attn.gamma_delta().view(1, -1, 1, 1)
                start_chunks = start_pos // blk.attn.C
                end_chunks = pos // blk.attn.C
                n_res_steps = max(0, max(0, end_chunks - 2) - max(0, start_chunks - 2))
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
                end_chunks = end_p // ly.C
                next_chunks = next_pos // ly.C
                n_res_steps = max(0, max(0, next_chunks - 2) - max(0, end_chunks - 2))
                states[li] = states[li] - (gam ** n_res_steps) * delta
        return states
