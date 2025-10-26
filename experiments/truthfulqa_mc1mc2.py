#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TruthfulQA MC1/MC2 
"""

from collections import defaultdict
import os
import math
import time
import logging
import random
import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def secs(elapsed: float) -> str:
    return f"{elapsed:.2f}s"


def setup_logger(log_file: Optional[str]) -> logging.Logger:
    logger = logging.getLogger("truthfulqa_range")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def ensure_trailing_space(text: str) -> str:
    return text if text.endswith((" ", "\n")) else text + " "


def build_prompt(question: str, emo_prefix: str = "", head_repeat: int = 1) -> str:
    if emo_prefix and head_repeat > 0:
        prefix = " ".join([emo_prefix.strip()] * max(1, head_repeat)).strip()
        if prefix:
            return f"{prefix}\n\nQ: {question.strip()}\nA: "
    return f"Q: {question.strip()}\nA: "


def encode(tok, text: str, device: str) -> torch.Tensor:
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def _parse_pair_floats(value: Union[str, Tuple[float, float]], default=(0.0, 200.0)) -> Tuple[float, float]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
        return (min(lo, hi), max(lo, hi))
    try:
        lo_s, hi_s = str(value).split(",")
        lo, hi = float(lo_s), float(hi_s)
        return (min(lo, hi), max(lo, hi))
    except Exception:
        return default


def logsumexp_list(xs: List[float]) -> float:
    if not xs:
        return float("-inf")
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def find_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    raise RuntimeError("Unsupported model structure: cannot find decoder layers.")


def weighted_mean(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w.reshape(-1, 1).astype(np.float32)
    return (X * w).sum(0) / (w.sum(0) + 1e-8)


def weighted_pca(X: np.ndarray, w: np.ndarray, r: int = 2, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = weighted_mean(X, w)
    Xc = X - mu
    std = Xc.std(0, keepdims=True) + eps
    Xw = (Xc / std) * np.sqrt(w.reshape(-1, 1))
    U, S, Vt = np.linalg.svd(Xw, full_matrices=False)
    V = Vt[:r].astype(np.float32)
    for i in range(V.shape[0]):
        V[i] /= (np.linalg.norm(V[i]) + 1e-9)
    return V, mu, std


def length_adaptive_k(plen: int) -> int:
    return int(np.clip(round(0.10 * max(1, plen)), 3, 8))


EP_TEXTS = [
    "Give a concise, factual answer.",
    "Avoid repeating myths; respond concisely with verified facts.",
]

MODEL_PRESETS = {
    "llama": {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "prompt_style": "plain",
        "n_probe": 200,
        "n_dev": 416,
        "max_examples": 216,
        "layer_start": 0,
        "layer_stop": 31,
        "prompt_tail_k": 3,
        "include_prompt": True,
        "include_pred": True,
        "pred_start_rel": 1,
        "pred_stop_rel": 2,
        "z_target": 0.85,
        "alpha_clip": "0,200",
        "residual_scale": -1.0,
        "residual_auto_flip": True,
        "log_file": "runs/truthfulqa_layer_range_llama.log",
        "save_report": "runs/truthfulqa_layer_range_llama.json",
    },
    "mistral": {
        "model": "mistralai/Mistral-7B-v0.3",
        "prompt_style": "plain",
        "n_probe": 200,
        "n_dev": 216,
        "max_examples": 216,
        "layer_start": 15,
        "layer_stop": 31,
        "prompt_tail_k": 1,
        "include_prompt": True,
        "include_pred": True,
        "pred_start_rel": 1,
        "pred_stop_rel": 3,
        "z_target": 0.85,
        "alpha_clip": "2,2",
        "residual_scale": 1.0,
        "residual_auto_flip": True,
        "log_file": "runs/truthfulqa_layer_range_mistral.log",
        "save_report": "runs/truthfulqa_layer_range_mistral.json",
    },
    "qwen": {
        "model": "Qwen/Qwen2-7B-Instruct",
        "prompt_style": "plain",
        "n_probe": 200,
        "n_dev": 217,
        "max_examples": 217,
        "layer_start": 7,
        "layer_stop": 14,
        "prompt_tail_k": 1,
        "include_prompt": True,
        "include_pred": True,
        "pred_start_rel": 1,
        "pred_stop_rel": 2,
        "z_target": 0.85,
        "alpha_clip": "0,200",
        "residual_scale": 0.5,
        "residual_auto_flip": True,
        "log_file": "runs/truthfulqa_layer_range_qwen.log",
        "save_report": "runs/truthfulqa_layer_range_qwen.json",
    },
}


def _sum_vecs(vs: List[torch.Tensor]) -> torch.Tensor:
    if isinstance(vs, list):
        if len(vs) == 1:
            return vs[0]
        return torch.stack(vs, dim=0).sum(0)
    return vs


def _safe_unit(vec: torch.Tensor, eps: float = 1e-6) -> Optional[torch.Tensor]:
    norm = float(vec.norm().item())
    if norm < eps:
        return None
    return vec / norm


def _probe_unit_vectors(layer_vecs: Dict[int, List[torch.Tensor]]) -> Dict[int, torch.Tensor]:
    units: Dict[int, torch.Tensor] = {}
    for li, vlist in layer_vecs.items():
        vec = _sum_vecs(vlist).float()
        unit = _safe_unit(vec)
        if unit is not None:
            units[li] = unit
    return units


def _orth_residual(sample_vec: torch.Tensor, probe_unit: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    unit_sample = _safe_unit(sample_vec.float())
    if unit_sample is None:
        return None
    if probe_unit is None:
        return unit_sample
    proj = float(torch.dot(unit_sample, probe_unit).item())
    residual = unit_sample - proj * probe_unit
    return _safe_unit(residual)


def _maybe_empty_cache(device: str):
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def _register_steer_hooks(
    model,
    layer_vecs: Dict[int, Union[torch.Tensor, List[torch.Tensor]]],
    alpha: float,
    mask_weights: Dict[int, torch.Tensor],
):
    handles = []
    layers = find_decoder_layers(model)

    for li, vecs in layer_vecs.items():
        if li >= len(layers):
            continue
        vec = _sum_vecs(vecs)
        if vec is None:
            continue
        vec = vec.detach()

        def _hook(module, inputs, output, layer_index=li, steer_vec=vec):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden is None:
                return output
            inj = (alpha * steer_vec).to(hidden.device, hidden.dtype).view(1, 1, -1)
            if layer_index in mask_weights:
                mw = mask_weights[layer_index].to(hidden.device, hidden.dtype)
                new_hidden = hidden + mw * inj
            else:
                new_hidden = hidden + inj
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        handles.append(layers[li].register_forward_hook(_hook))
    return handles


@torch.inference_mode()
def batched_ll_for_candidates(
    tok,
    model,
    prompt_ids: torch.Tensor,
    cand_texts: List[str],
    device: str,
    steer: Optional[Dict] = None,
) -> Tuple[List[float], List[int]]:
    if not cand_texts:
        return [], []

    pad_id = tok.pad_token_id or (tok.eos_token_id if tok.eos_token_id is not None else 0)
    p_len = int(prompt_ids.size(1))
    cand_ids_list = [encode(tok, c.strip(), device) for c in cand_texts]
    max_c = max(ci.size(1) for ci in cand_ids_list)
    batch_size = len(cand_ids_list)

    seq_len = p_len + max_c
    device_t = prompt_ids.device
    input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device=device_t)
    attn_mask = torch.zeros_like(input_ids)

    input_ids[:, :p_len] = prompt_ids.expand(batch_size, -1)
    attn_mask[:, :p_len] = 1

    lengths: List[int] = []
    for i, ci in enumerate(cand_ids_list):
        L_i = int(ci.size(1))
        lengths.append(L_i)
        input_ids[i, p_len:p_len + L_i] = ci[0]
        attn_mask[i, :p_len + L_i] = 1

    handles = []
    if steer and steer.get("layer_vecs"):
        layer_vecs = steer["layer_vecs"]
        alpha_val = float(steer.get("alpha", 0.0))
        blueprint = steer.get("frozen_positions", [])
        seq_total = input_ids.size(1)
        mask_weights: Dict[int, torch.Tensor] = {
            li: torch.zeros((batch_size, seq_total, 1), dtype=torch.float32, device=device_t)
            for li in layer_vecs.keys()
        }

        for item in blueprint:
            li = int(item.get("li", -1))
            region = str(item.get("region", "prompt"))
            rel = int(item.get("rel", 0))
            weight = float(item.get("w", 0.0))
            if li not in mask_weights or weight == 0.0:
                continue
            if region == "prompt":
                t = (p_len - 1) - rel
                if 0 <= t < p_len:
                    mask_weights[li][:, t, 0] += weight
            elif region == "pred":
                for bi, L_i in enumerate(lengths):
                    t = (p_len - 1) + rel
                    if (p_len - 1) <= t <= (p_len + L_i - 2):
                        mask_weights[li][bi, t, 0] += weight
        handles = _register_steer_hooks(model, layer_vecs, alpha_val, mask_weights)

    out = model(input_ids=input_ids, attention_mask=attn_mask)
    for h in handles:
        h.remove()

    logprobs = torch.log_softmax(out.logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]

    lls: List[float] = []
    for i, L_i in enumerate(lengths):
        start = p_len
        end = p_len + L_i
        lp_slice = logprobs[i, start - 1:end - 1, :]
        tg_slice = targets[i, start - 1:end - 1].unsqueeze(-1)
        lls.append(float(lp_slice.gather(-1, tg_slice).sum().item()))
    return lls, lengths


def eval_single_truthfulqa_sample(ex, tok, model, device, steering=None) -> Tuple[float, float]:
    question = ex["question"]
    prompt = ensure_trailing_space(build_prompt(question, "", 0))
    prompt_ids = encode(tok, prompt, device)

    choices_mc1 = ex["mc1_targets"]["choices"]
    labels_mc1 = ex["mc1_targets"]["labels"]
    ll1, _ = batched_ll_for_candidates(tok, model, prompt_ids, choices_mc1, device, steer=steering)
    pred_idx = int(max(range(len(ll1)), key=lambda i: ll1[i])) if ll1 else -1
    gold_idx = int(labels_mc1.index(1)) if 1 in labels_mc1 else -1
    mc1 = 1.0 if (gold_idx >= 0 and pred_idx == gold_idx) else 0.0

    choices_mc2 = ex["mc2_targets"]["choices"]
    labels_mc2 = ex["mc2_targets"]["labels"]
    ll2, _ = batched_ll_for_candidates(tok, model, prompt_ids, choices_mc2, device, steer=steering)
    if ll2:
        denom = logsumexp_list(ll2)
        probs = [math.exp(x - denom) for x in ll2]
        mc2 = float(sum(p for p, y in zip(probs, labels_mc2) if y == 1))
    else:
        mc2 = 0.0
    return mc1, mc2


@torch.no_grad()
def extract_vectors_soft_gate(
    ds_probe,
    tok,
    model,
    device: str,
    emo_pool: Optional[List[str]] = None,
    head_repeat: int = 1,
    gate_obj: str = "mc2",
    gate_tau: float = 0.05,
    cv_folds: int = 2,
    top_k: Optional[Union[int, float]] = None,
) -> Dict[int, List[torch.Tensor]]:
    layers = find_decoder_layers(model)
    L = len(layers)
    H = model.config.hidden_size

    def _resolve_all_ep_texts(pool: Optional[List[str]]) -> List[str]:
        if not pool or (len(pool) == 1 and str(pool[0]).upper() == "ALL"):
            return EP_TEXTS
        return [str(x) for x in pool]

    ep_texts = _resolve_all_ep_texts(emo_pool)
    P = len(ep_texts)

    def _coerce_top_k(val: Optional[Union[int, float]], total: int) -> int:
        if val is None:
            return total
        if isinstance(val, float):
            if val <= 0:
                return total
            if val < 1:
                return min(total, int(math.ceil(val * total)))
            return min(total, int(val))
        try:
            k_int = int(val)
        except Exception:
            return total
        return min(total, max(1, k_int))

    K = _coerce_top_k(top_k, P)
    idx_all = np.arange(len(ds_probe))
    folds = [idx_all]
    if cv_folds >= 2 and len(idx_all) >= cv_folds:
        rng = np.random.default_rng(0)
        folds = np.array_split(rng.permutation(idx_all), cv_folds)

    diffs_tail = {l: [] for l in range(L)}
    diffs_end = {l: [] for l in range(L)}

    def objective_from_lls(lls: List[float], labels: List[int], obj="mc2", lam=0.3):
        if not lls:
            return 0.0
        denom = logsumexp_list(lls)
        probs = [math.exp(x - denom) for x in lls]
        mc2_true = sum(p for p, yy in zip(probs, labels) if yy == 1)
        if obj == "mc2":
            return mc2_true
        true_lls = [ll for ll, yy in zip(lls, labels) if yy == 1]
        false_lls = [ll for ll, yy in zip(lls, labels) if yy == 0]
        margin = (max(true_lls) - max(false_lls)) if (true_lls and false_lls) else 0.0
        return mc2_true + lam * (1.0 if margin > 0 else 0.0)

    for fold in folds:
        for idx in fold:
            ex = ds_probe[int(idx)]
            question = ex["question"]

            ids_plain = encode(tok, ensure_trailing_space(build_prompt(question, "", 0)), device)
            out_plain = model(input_ids=ids_plain, output_hidden_states=True)
            hs_plain = out_plain.hidden_states
            plen = ids_plain.size(1)
            k_tail = length_adaptive_k(plen)

            choices_mc2 = ex["mc2_targets"]["choices"]
            labels_mc2 = ex["mc2_targets"]["labels"]
            ll_plain, _ = batched_ll_for_candidates(tok, model, ids_plain, choices_mc2, device, steer=None)

            hs_plain_tail = [
                hs_plain[l + 1][0, plen - k_tail:plen, :].detach().float().cpu().numpy().mean(axis=0)
                for l in range(L)
            ]
            hs_plain_end = [
                hs_plain[l + 1][0, plen - 1, :].detach().float().cpu().numpy()
                for l in range(L)
            ]

            gains = []
            diffs_p_tail = []
            diffs_p_end = []
            for ep_text in ep_texts:
                ids_ep = encode(tok, ensure_trailing_space(build_prompt(question, ep_text, head_repeat)), device)
                out_ep = model(input_ids=ids_ep, output_hidden_states=True)
                hs_ep = out_ep.hidden_states
                elen = ids_ep.size(1)
                k_tail_ep = length_adaptive_k(elen)

                ll_ep, _ = batched_ll_for_candidates(tok, model, ids_ep, choices_mc2, device, steer=None)
                J_plain = objective_from_lls(ll_plain, labels_mc2, obj=gate_obj, lam=0.3)
                J_ep = objective_from_lls(ll_ep, labels_mc2, obj=gate_obj, lam=0.3)
                gains.append(J_ep - J_plain)

                tail = []
                end = []
                for l in range(L):
                    h_plain_tail = hs_plain_tail[l]
                    h_plain_end = hs_plain_end[l]
                    h_ep_tail = (
                        hs_ep[l + 1][0, elen - k_tail_ep:elen, :].detach().float().cpu().numpy().mean(axis=0)
                    )
                    h_ep_end = hs_ep[l + 1][0, elen - 1, :].detach().float().cpu().numpy()
                    tail.append(h_ep_tail - h_plain_tail)
                    end.append(h_ep_end - h_plain_end)
                diffs_p_tail.append(tail)
                diffs_p_end.append(end)

            gains = np.asarray(gains, dtype=np.float32)
            if K >= P:
                sel_idx = np.arange(P, dtype=np.int64)
            else:
                sel_idx = np.argpartition(gains, -K)[-K:]
                sel_idx = sel_idx[np.argsort(gains[sel_idx])[::-1]]

            for k in sel_idx.tolist():
                for l in range(L):
                    diffs_tail[l].append(np.asarray(diffs_p_tail[k][l], dtype=np.float32))
                    diffs_end[l].append(np.asarray(diffs_p_end[k][l], dtype=np.float32))

    layer_vecs: Dict[int, List[torch.Tensor]] = {}
    for l in range(L):
        Xt = np.stack(diffs_tail[l], axis=0) if diffs_tail[l] else np.zeros((1, H), np.float32)
        Xe = np.stack(diffs_end[l], axis=0) if diffs_end[l] else np.zeros((1, H), np.float32)
        X  = np.concatenate([Xt, Xe], axis=0)
        w = np.ones((X.shape[0],), dtype=np.float32)
        V, _, _ = weighted_pca(X, w, r=2)
        layer_vecs[l] = [torch.from_numpy(V[i]) for i in range(V.shape[0])]

        
    return layer_vecs


def weighted_pca_np(X: np.ndarray, r: int = 1, eps: float = 1e-6) -> np.ndarray:
    X = X.astype(np.float32)
    Xc = X - X.mean(0, keepdims=True)
    std = Xc.std(0, keepdims=True) + eps
    Xn = Xc / std
    U, _, Vt = np.linalg.svd(Xn, full_matrices=False)
    V = Vt[:r].astype(np.float32)
    for i in range(V.shape[0]):
        V[i] /= (np.linalg.norm(V[i]) + 1e-9)
    return V


@torch.no_grad()
def extract_single_sample_vecs(question: str, tok, model, device: str) -> Dict[int, torch.Tensor]:
    layers = find_decoder_layers(model)
    L = len(layers); H = model.config.hidden_size

    prompt_plain = ensure_trailing_space(build_prompt(question, "", 0))
    ids_plain = encode(tok, prompt_plain, device)
    out_plain = model(input_ids=ids_plain, output_hidden_states=True)
    hs_plain = out_plain.hidden_states
    plen = int(ids_plain.size(1))
    k_tail = length_adaptive_k(plen)

    plain_tail = []
    plain_end = []
    for l in range(L):
        h_plain = hs_plain[l+1][0]
        tail = h_plain[plen-k_tail:plen, :].detach().float().cpu().numpy().mean(axis=0)
        end = h_plain[plen-1, :].detach().float().cpu().numpy()
        plain_tail.append(tail)
        plain_end.append(end)

    del out_plain, hs_plain, ids_plain
    _maybe_empty_cache(device)

    diffs_tail = {l: [] for l in range(L)}
    diffs_end = {l: [] for l in range(L)}

    for ep_text in EP_TEXTS:
        ep_prompt = ensure_trailing_space(build_prompt(question, ep_text, 1))
        ids_ep = encode(tok, ep_prompt, device)
        out_ep = model(input_ids=ids_ep, output_hidden_states=True)
        hs_ep = out_ep.hidden_states
        elen = int(ids_ep.size(1))
        k_tail_ep = length_adaptive_k(elen)

        for l in range(L):
            h_ep = hs_ep[l+1][0]
            tail = h_ep[elen-k_tail_ep:elen, :].detach().float().cpu().numpy().mean(axis=0)
            end = h_ep[elen-1, :].detach().float().cpu().numpy()
            diffs_tail[l].append((tail - plain_tail[l]).astype(np.float32))
            diffs_end[l].append((end - plain_end[l]).astype(np.float32))

        del out_ep, hs_ep, ids_ep
        _maybe_empty_cache(device)

    layer_vecs: Dict[int, torch.Tensor] = {}
    for l in range(L):
        Xt = np.stack(diffs_tail[l], axis=0) if diffs_tail[l] else np.zeros((1, H), np.float32)
        Xe = np.stack(diffs_end[l], axis=0) if diffs_end[l] else np.zeros((1, H), np.float32)
        X = np.concatenate([Xt, Xe], axis=0)
        V = weighted_pca_np(X, r=1)
        layer_vecs[l] = torch.from_numpy(V[0])
    return layer_vecs


def _alphas_prompt_for_layers(hs_plain, p_len: int, layer_vecs_cur: Dict[int, List[torch.Tensor]],
                              rels: List[int], z_target: float,
                              alpha_clip: Tuple[float, float]) -> Tuple[Dict[Tuple[int, int], float], int]:
    results: Dict[Tuple[int, int], float] = {}
    attempts = 0
    dtype = hs_plain[1].dtype
    lo, hi = alpha_clip
    s = max(1e-6, min(1.0 - 1e-6, float(z_target)))
    for li, vlist in layer_vecs_cur.items():
        v = _sum_vecs(vlist).to(hs_plain[li+1].device, dtype)
        c = float(v.norm().item()) + 1e-9
        if c <= 0:
            continue
        u = v / c
        for rel in rels:
            t_abs = (p_len - 1) - rel
            if not (0 <= t_abs < p_len):
                continue
            attempts += 1
            h = hs_plain[li+1][0, t_abs, :].to(dtype)
            a = float(torch.dot(h, u).item())
            hn = float(h.norm().item()) + 1e-9
            Bn = math.sqrt(max(hn * hn - a * a, 0.0))
            T = (Bn * s) / math.sqrt(max(1.0 - s * s, 1e-9))
            alpha1 = max(0.0, (-a) / max(c, 1e-9))
            alpha2 = (T - a) / max(c, 1e-9)
            alpha = float(max(lo, min(hi, max(alpha1, alpha2))))
            if alpha > 0.0:
                results[(li, rel)] = alpha
    return results, attempts


@torch.no_grad()
def _alphas_pred_for_layers(tok, model, device, layer_vecs_cur: Dict[int, List[torch.Tensor]],
                             prompt_ids: torch.Tensor, cand_texts: List[str],
                             z_rank: float, z_inj: float,
                             pred_start: int, pred_stop: int,
                             alpha_clip: Tuple[float, float]) -> Tuple[Dict[Tuple[int, int], float], int]:
    results: Dict[Tuple[int, int], float] = {}
    attempts = 0
    if not cand_texts or pred_stop < pred_start:
        return results, attempts

    p_len = int(prompt_ids.size(1))
    cand_ids_list = [encode(tok, c.strip(), device) for c in cand_texts]
    if not cand_ids_list:
        return results, attempts

    max_c = max(ci.size(1) for ci in cand_ids_list)
    B = len(cand_ids_list)
    pad_id = tok.pad_token_id or (tok.eos_token_id if tok.eos_token_id is not None else 0)
    S = p_len + max_c
    input_ids = torch.full((B, S), pad_id, dtype=torch.long, device=prompt_ids.device)
    attn_mask = torch.zeros_like(input_ids)
    input_ids[:, :p_len] = prompt_ids.expand(B, -1)
    attn_mask[:, :p_len] = 1

    lengths: List[int] = []
    for i, ci in enumerate(cand_ids_list):
        L_i = int(ci.size(1))
        lengths.append(L_i)
        input_ids[i, p_len:p_len+L_i] = ci[0]
        attn_mask[i, :p_len+L_i] = 1

    out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
    hs_list = out.hidden_states
    dtype = hs_list[1].dtype

    u_by_layer: Dict[int, torch.Tensor] = {}
    c_by_layer: Dict[int, float] = {}
    for l, vlist in layer_vecs_cur.items():
        v = _sum_vecs(vlist).to(device=device, dtype=dtype)
        cn = float(v.norm().item()) + 1e-9
        if cn <= 0:
            continue
        u_by_layer[l] = v / cn
        c_by_layer[l] = cn

    s_r = max(1e-6, min(1.0 - 1e-6, float(z_rank)))
    s_i = max(1e-6, min(1.0 - 1e-6, float(z_inj)))
    lo, hi = alpha_clip
    eps = 1e-9

    rmax_available = max(lengths) - 1
    if rmax_available < 0:
        return results, attempts

    start = max(0, pred_start)
    stop = min(pred_stop, rmax_available)
    if stop < start:
        return results, attempts

    agg_inj: Dict[Tuple[int, int], List[float]] = defaultdict(list)

    for li in sorted(layer_vecs_cur.keys()):
        u = u_by_layer.get(li)
        c = c_by_layer.get(li, 0.0)
        if u is None or c <= 0:
            continue
        H = hs_list[li+1]
        for r in range(start, stop + 1):
            for bi, L_i in enumerate(lengths):
                if r > L_i - 1:
                    continue
                t = (p_len - 1) + r
                h = H[bi, t, :].to(dtype)
                a = float(torch.dot(h, u).item())
                hn = float(h.norm().item()) + 1e-9
                Bn = math.sqrt(max(hn * hn - a * a, 0.0))

                attempts += 1

                T_i = (Bn * s_i) / math.sqrt(max(1.0 - s_i * s_i, eps))
                alpha1_i = max(0.0, (-a) / max(c, eps))
                alpha2_i = (T_i - a) / max(c, eps)
                alpha_i = float(max(lo, min(hi, max(alpha1_i, alpha2_i))))
                if alpha_i > 0.0:
                    agg_inj[(li, r)].append(alpha_i)

    for key, vals in agg_inj.items():
        if not vals:
            continue
        results[key] = float(np.mean(vals))
    return results, attempts


def main():
    import argparse

    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument(
        "--preset",
        choices=sorted(MODEL_PRESETS.keys()),
        default="llama",
        help="Model-specific preset that restores the paper hyper-parameters.",
    )
    preset_args, remaining = preset_parser.parse_known_args()
    preset_cfg = MODEL_PRESETS[preset_args.preset]

    ap = argparse.ArgumentParser(parents=[preset_parser])
    ap.add_argument("--model", default=preset_cfg["model"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf_token_env", default="HF_TOKEN")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--n_probe", type=int, default=preset_cfg["n_probe"])
    ap.add_argument("--n_dev", type=int, default=preset_cfg["n_dev"])
    ap.add_argument("--max-examples", type=int, default=preset_cfg["max_examples"])
    ap.add_argument("--layer-start", type=int, default=preset_cfg["layer_start"])
    ap.add_argument("--layer-stop", type=int, default=preset_cfg["layer_stop"])
    ap.add_argument("--prompt-tail-k", type=int, default=preset_cfg["prompt_tail_k"],
                    help="Prompt tail tokens (0 => adaptive length_adaptive_k)")
    ap.add_argument("--prompt-style", choices=["auto", "chat", "plain"], default=preset_cfg["prompt_style"],
                    help="Prompt formatting: auto-detect chat/instruction vs plain Q/A style.")
    ap.add_argument("--system-prompt", type=str, default="",
                    help="System prompt used when --prompt-style resolves to chat mode.")
    ap.add_argument("--include-prompt", action=argparse.BooleanOptionalAction,
                    default=preset_cfg["include_prompt"])
    ap.add_argument("--include-pred", action=argparse.BooleanOptionalAction,
                    default=preset_cfg["include_pred"])
    ap.add_argument("--pred-start-rel", type=int, default=preset_cfg["pred_start_rel"],
                    help="Predicted token relative start (skip 0 to avoid prompt overlap)")
    ap.add_argument("--pred-stop-rel", type=int, default=preset_cfg["pred_stop_rel"],
                    help="Predicted token relative stop (inclusive)")
    ap.add_argument("--z_target", type=float, default=preset_cfg["z_target"])
    ap.add_argument("--alpha_clip", type=str, default=preset_cfg["alpha_clip"])
    ap.add_argument("--residual-scale", type=float, default=preset_cfg["residual_scale"])
    ap.add_argument("--residual-auto-flip", action=argparse.BooleanOptionalAction,
                    default=preset_cfg["residual_auto_flip"])
    ap.add_argument("--residual-log-cos", action="store_true")
    ap.add_argument("--log-file", type=str, default=preset_cfg["log_file"])
    ap.add_argument("--save-report", type=str, default=preset_cfg["save_report"])
    args = ap.parse_args(remaining)

    logger = setup_logger(args.log_file)
    logger.info("[preset] %s", args.preset)
    logger.info("[seed] %d", args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    token = os.getenv(args.hf_token_env, None)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, token=token, local_files_only=args.local_files_only)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=token,
        torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
        device_map="auto",
        local_files_only=args.local_files_only,
    )
    model.eval(); model.config.use_cache = False
    logger.info("[model] loaded in %s", secs(time.time() - t0))

    prompt_style = args.prompt_style.lower()
    has_chat_template = bool(getattr(tok, "chat_template", None))
    prompt_style_resolved = prompt_style
    if prompt_style == "auto":
        prompt_style_resolved = "chat" if (has_chat_template or "instruct" in args.model.lower()) else "plain"
    if prompt_style_resolved == "chat" and not hasattr(tok, "apply_chat_template"):
        logger.warning("Tokenizer lacks chat template API; falling back to plain prompt style.")
        prompt_style_resolved = "plain"
    system_prompt = args.system_prompt.strip() if prompt_style_resolved == "chat" else ""
    logger.info("[prompt] style=%s system=%s", prompt_style_resolved,
                "set" if system_prompt else "none")

    def _build_prompt_override(q: str, emo_prefix: str = "", head_repeat: int = 1) -> str:
        prefix_text = ""
        if emo_prefix and head_repeat > 0:
            prefix_text = " ".join([emo_prefix.strip()] * max(1, head_repeat)).strip()
        user_sections: List[str] = []
        if prefix_text:
            user_sections.append(prefix_text)
        user_sections.append(q.strip())
        user_text = "\n\n".join([s for s in user_sections if s])

        if prompt_style_resolved == "chat":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_text})
            rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            rendered = rendered.rstrip()
            if not rendered.endswith((" ", "\n")):
                rendered += " "
            return rendered

        prefix_block = f"{prefix_text}\n\n" if prefix_text else ""
        return f"{prefix_block}Q: {q.strip()}\nA: "

    globals()["build_prompt"] = _build_prompt_override

    ds_mc = load_dataset("truthful_qa", "multiple_choice", trust_remote_code=False)["validation"]
    N_total = len(ds_mc)
    rng = np.random.default_rng(args.seed)
    all_idx = np.arange(N_total); rng.shuffle(all_idx)

    n_probe = min(args.n_probe, N_total)
    remain_after_probe = max(0, N_total - n_probe)
    n_dev = remain_after_probe if args.n_dev < 0 else min(args.n_dev, remain_after_probe)

    probe_idx = all_idx[:n_probe].tolist()
    dev_idx = all_idx[n_probe:n_probe + n_dev].tolist()

    ds_probe = ds_mc.select(probe_idx)
    ds_dev = ds_mc.select(dev_idx)
    logger.info("TruthfulQA split: total=%d PROBE=%d DEV=%d", N_total, len(ds_probe), len(ds_dev))

    t0 = time.time()
    layer_vecs = extract_vectors_soft_gate(
        ds_probe, tok, model, device,
        emo_pool=None, head_repeat=1,
        gate_obj="mc2", gate_tau=0.05, cv_folds=2,
        top_k=None,
    )
    logger.info("[subspace] extracted in %s (layers=%d)", secs(time.time() - t0), len(layer_vecs))
    probe_units = _probe_unit_vectors(layer_vecs)

    examples: List[Dict] = []
    max_examples = len(ds_dev) if args.max_examples < 0 else min(len(ds_dev), args.max_examples)
    for i in range(max_examples):
        ex = ds_dev[i]
        examples.append({"ex": ex, "question": ex["question"]})

    if not examples:
        logger.info("No DEV examples to process; exit.")
        return

    logger.info("[baseline] evaluating %d samples...", len(examples))
    for item in examples:
        mc1, mc2 = eval_single_truthfulqa_sample(item["ex"], tok, model, device, steering=None)
        item["mc1_base"] = mc1
        item["mc2_base"] = mc2

    baseline_mc1 = sum(it["mc1_base"] for it in examples) / len(examples)
    baseline_mc2 = sum(it["mc2_base"] for it in examples) / len(examples)
    logger.info("[baseline] MC1=%.4f MC2=%.4f", baseline_mc1, baseline_mc2)

    logger.info("[sample-subspace] extracting per-example residual vectors...")
    for idx, item in enumerate(examples):
        residual_vecs = extract_single_sample_vecs(item["question"], tok, model, device)
        residual_clean: Dict[int, torch.Tensor] = {}
        for li, vec in residual_vecs.items():
            res = _orth_residual(vec, probe_units.get(li))
            if res is not None:
                residual_clean[li] = res
        item["residual_vecs"] = residual_clean
        if (idx + 1) % 25 == 0 or (idx + 1) == len(examples):
            logger.info("  processed %d/%d", idx + 1, len(examples))
    _maybe_empty_cache(device)

    lo_clip, hi_clip = _parse_pair_floats(args.alpha_clip, default=(0.0, 200.0))
    residual_scale = float(args.residual_scale)
    logger.info("[residual] scale=%.3f auto_flip=%s log_cos=%s",
                residual_scale, args.residual_auto_flip, args.residual_log_cos)

    Lmax = len(find_decoder_layers(model)) - 1
    Lstart = max(0, int(args.layer_start))
    Lstop = min(Lmax, int(args.layer_stop))
    range_layers = [li for li in sorted(layer_vecs.keys()) if Lstart <= li <= Lstop]
    if not range_layers:
        logger.warning("No layers in the specified range [%d, %d].", Lstart, Lstop)
        return

    include_prompt = bool(args.include_prompt)
    include_pred = bool(args.include_pred)
    pred_start_rel = int(args.pred_start_rel)
    pred_stop_rel = int(args.pred_stop_rel)
    if pred_stop_rel < pred_start_rel:
        include_pred = False

    global_mc1_sum = global_mc2_sum = 0.0
    mix_mc1_sum = mix_mc2_sum = 0.0
    alpha_global_sum = alpha_mix_sum = 0.0
    attempts_global = attempts_mix = 0
    global_nonzero = mix_nonzero = 0
    mix_usage = mix_flip = mix_neg = 0

    for ex_idx, item in enumerate(examples):
        question = item["question"]
        residuals = item["residual_vecs"]

        layer_vecs_global = {li: layer_vecs[li] for li in range_layers}
        layer_vecs_mix: Dict[int, List[torch.Tensor]] = {}

        # prepare residual-adjusted vectors per layer
        for li in range_layers:
            base_list = list(layer_vecs[li])
            residual_vec = residuals.get(li)
            if residual_vec is None:
                layer_vecs_mix[li] = base_list
                continue
            mix_usage += 1
            probe_unit = probe_units.get(li)
            cos_orig = None
            if probe_unit is not None:
                denom = float(residual_vec.norm().item() * probe_unit.norm().item()) + 1e-9
                if denom > 0.0:
                    cos_orig = float(torch.dot(residual_vec, probe_unit).item() / denom)
            if cos_orig is not None and cos_orig < 0.0:
                mix_neg += 1
            residual_vec_use = residual_vec
            flipped = False
            if args.residual_auto_flip and cos_orig is not None and cos_orig < 0.0:
                residual_vec_use = -residual_vec_use
                flipped = True
                mix_flip += 1
            if args.residual_log_cos:
                logger.info(
                    "[residual-cos] layer=%02d sample=%03d cos_orig=%s flipped=%s",
                    li, ex_idx,
                    f"{cos_orig:.4f}" if cos_orig is not None else "nan",
                    flipped,
                )
            layer_vecs_mix[li] = base_list + [residual_scale * residual_vec_use]

        for li in range_layers:
            if li not in layer_vecs_mix:
                layer_vecs_mix[li] = list(layer_vecs[li])

        prompt = ensure_trailing_space(build_prompt(question, "", 0))
        prompt_ids = encode(tok, prompt, device)
        out_plain = model(input_ids=prompt_ids, output_hidden_states=True)
        hs_plain = out_plain.hidden_states
        p_len = int(prompt_ids.size(1))

        # Prompt alphas -------------------------------------------------
        prompt_rels: List[int] = []
        if include_prompt:
            if args.prompt_tail_k <= 0:
                k_prompt = min(length_adaptive_k(p_len), p_len)
            else:
                k_prompt = min(int(args.prompt_tail_k), p_len)
            prompt_rels = list(range(0, k_prompt))

        prompt_alphas_global: Dict[Tuple[int, int], float] = {}
        prompt_alphas_mix: Dict[Tuple[int, int], float] = {}
        if prompt_rels:
            prompt_alphas_global, att_g = _alphas_prompt_for_layers(
                hs_plain, p_len, layer_vecs_global, prompt_rels,
                float(args.z_target), (lo_clip, hi_clip))
            prompt_alphas_mix, att_m = _alphas_prompt_for_layers(
                hs_plain, p_len, layer_vecs_mix, prompt_rels,
                float(args.z_target), (lo_clip, hi_clip))
            attempts_global += att_g
            attempts_mix += att_m

        # Pred alphas ---------------------------------------------------
        pred_alphas_global: Dict[Tuple[int, int], float] = {}
        pred_alphas_mix: Dict[Tuple[int, int], float] = {}
        if include_pred:
            cand_texts = item["ex"]["mc2_targets"]["choices"]
            pred_alphas_global, att_g = _alphas_pred_for_layers(
                tok, model, device, layer_vecs_global,
                prompt_ids, cand_texts,
                float(args.z_target), float(args.z_target),
                pred_start_rel, pred_stop_rel,
                (lo_clip, hi_clip),
            )
            pred_alphas_mix, att_m = _alphas_pred_for_layers(
                tok, model, device, layer_vecs_mix,
                prompt_ids, cand_texts,
                float(args.z_target), float(args.z_target),
                pred_start_rel, pred_stop_rel,
                (lo_clip, hi_clip),
            )
            attempts_global += att_g
            attempts_mix += att_m

        blueprint_global: List[Dict] = []
        blueprint_mix: List[Dict] = []

        for (li, rel), alpha in prompt_alphas_global.items():
            alpha_global_sum += alpha
            if alpha > 0.0:
                global_nonzero += 1
                blueprint_global.append({"li": li, "region": "prompt", "rel": rel, "w": float(alpha)})
        for (li, rel), alpha in prompt_alphas_mix.items():
            alpha_mix_sum += alpha
            if alpha > 0.0:
                mix_nonzero += 1
                blueprint_mix.append({"li": li, "region": "prompt", "rel": rel, "w": float(alpha)})

        for (li, rel), alpha in pred_alphas_global.items():
            alpha_global_sum += alpha
            if alpha > 0.0:
                global_nonzero += 1
                blueprint_global.append({"li": li, "region": "pred", "rel": rel, "w": float(alpha)})
        for (li, rel), alpha in pred_alphas_mix.items():
            alpha_mix_sum += alpha
            if alpha > 0.0:
                mix_nonzero += 1
                blueprint_mix.append({"li": li, "region": "pred", "rel": rel, "w": float(alpha)})

        steering_global = {
            "layer_vecs": layer_vecs_global,
            "alpha": 1.0,
            "pos_mode": "fixed_dev",
            "frozen_positions": blueprint_global,
            "energy_share": True,
        }
        mc1_g, mc2_g = eval_single_truthfulqa_sample(item["ex"], tok, model, device, steering=steering_global)
        global_mc1_sum += mc1_g
        global_mc2_sum += mc2_g

        steering_mix = {
            "layer_vecs": layer_vecs_mix,
            "alpha": 1.0,
            "pos_mode": "fixed_dev",
            "frozen_positions": blueprint_mix,
            "energy_share": True,
        }
        mc1_m, mc2_m = eval_single_truthfulqa_sample(item["ex"], tok, model, device, steering=steering_mix)
        mix_mc1_sum += mc1_m
        mix_mc2_sum += mc2_m

    total_samples = len(examples)
    global_mc1 = global_mc1_sum / total_samples
    global_mc2 = global_mc2_sum / total_samples
    mix_mc1 = mix_mc1_sum / total_samples
    mix_mc2 = mix_mc2_sum / total_samples

    positions_global = max(1, attempts_global)
    positions_mix = max(1, attempts_mix)

    row_global = {
        "mc1": round(global_mc1, 6),
        "delta_mc1": round(global_mc1 - baseline_mc1, 6),
        "mc2": round(global_mc2, 6),
        "delta_mc2": round(global_mc2 - baseline_mc2, 6),
        "alpha_mean": round(alpha_global_sum / positions_global, 6),
        "alpha_nonzero_rate": round(global_nonzero / positions_global, 6),
    }

    row_mix = {
        "mc1": round(mix_mc1, 6),
        "delta_mc1": round(mix_mc1 - baseline_mc1, 6),
        "mc2": round(mix_mc2, 6),
        "delta_mc2": round(mix_mc2 - baseline_mc2, 6),
        "alpha_mean": round(alpha_mix_sum / positions_mix, 6),
        "alpha_nonzero_rate": round(mix_nonzero / positions_mix, 6),
        "residual_usage_rate": round(mix_usage / positions_mix, 6),
        "residual_flip_rate": round(mix_flip / max(1, mix_usage), 6) if mix_usage else 0.0,
        "residual_neg_rate": round(mix_neg / max(1, mix_usage), 6) if mix_usage else 0.0,
    }

    range_name = f"{range_layers[0]}-{range_layers[-1]}" if len(range_layers) > 1 else str(range_layers[0])
    logger.info(
        "[range %s] global MC1=%.4f Δ=%.4f MC2=%.4f Δ=%.4f ᾱ=%.3f nonzero=%.3f | +res MC1=%.4f Δ=%.4f MC2=%.4f Δ=%.4f ᾱ=%.3f nonzero=%.3f use=%.3f flip=%.3f neg=%.3f",
        range_name,
        row_global["mc1"], row_global["delta_mc1"], row_global["mc2"], row_global["delta_mc2"],
        row_global["alpha_mean"], row_global["alpha_nonzero_rate"],
        row_mix["mc1"], row_mix["delta_mc1"], row_mix["mc2"], row_mix["delta_mc2"],
        row_mix["alpha_mean"], row_mix["alpha_nonzero_rate"],
        row_mix["residual_usage_rate"], row_mix["residual_flip_rate"], row_mix["residual_neg_rate"],
    )

    print("\n=== TruthfulQA prompt multi-token multi-layer injection ===")
    print(f"Baseline: MC1={baseline_mc1:.4f} MC2={baseline_mc2:.4f}  (N={total_samples})\n")
    print(f"Range {range_name}: global MC1={row_global['mc1']:.4f} Δ={row_global['delta_mc1']:+.4f} MC2={row_global['mc2']:.4f} Δ={row_global['delta_mc2']:+.4f} ᾱ={row_global['alpha_mean']:.3f} nonzero={row_global['alpha_nonzero_rate']:.3f}")
    print(f"              +res MC1={row_mix['mc1']:.4f} Δ={row_mix['delta_mc1']:+.4f} MC2={row_mix['mc2']:.4f} Δ={row_mix['delta_mc2']:+.4f} ᾱ={row_mix['alpha_mean']:.3f} nonzero={row_mix['alpha_nonzero_rate']:.3f} use={row_mix['residual_usage_rate']:.3f} flip={row_mix['residual_flip_rate']:.3f} neg={row_mix['residual_neg_rate']:.3f}")

    if args.save_report:
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump({
                "baseline": {"mc1": baseline_mc1, "mc2": baseline_mc2, "N": total_samples},
                "range": range_name,
                "global": row_global,
                "mix": row_mix,
                "config": {
                    "model": args.model,
                    "z_target": float(args.z_target),
                    "alpha_clip": [lo_clip, hi_clip],
                    "residual_scale": residual_scale,
                    "residual_auto_flip": bool(args.residual_auto_flip),
                    "layer_start": Lstart,
                    "layer_stop": Lstop,
                    "prompt_tail_k": int(args.prompt_tail_k),
                    "include_prompt": include_prompt,
                    "include_pred": include_pred,
                    "pred_start_rel": pred_start_rel,
                    "pred_stop_rel": pred_stop_rel,
                    "prompt_style": prompt_style_resolved,
                    "system_prompt": system_prompt,
                },
            }, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report to: {args.save_report}")


if __name__ == "__main__":
    main()
