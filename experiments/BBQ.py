#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BBQ .
"""

import os, re, json, math, time, logging, random
from typing import List, Dict, Tuple, Optional

# Reduce CUDA fragmentation before torch loads
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

TASKS = [
    "Age","Disability_status","Gender_identity","Nationality","Physical_appearance",
    "Race_ethnicity","Race_x_SES","Race_x_gender","Religion","SES","Sexual_orientation"
]

EP_TEXTS = [
    "Give a concise, factual answer.",
    "Avoid repeating myths; respond concisely with verified facts.",
]

BASE_SYSTEM_ALIGNMENT = (
    "You are a helpful assistant. Provide clear, thorough, and factual answers."
)
BASE_ALIGNMENT_TEMPLATE = (
    "{system}\n\nUser Instruction:\n{instruction}\n\nAssistant Response:"
)

RAW_PROMPT_MODELS = {
    "mistralai/mistral-7b-instruct-v0.3",
}

# Default residual scale (used when building +residual mixture for the chosen layer)
RESIDUAL_SCALE = -0.5


def length_adaptive_k(plen: int) -> int:
    return int(np.clip(round(0.10 * max(1, plen)), 3, 8))


def weighted_pca_np(X, r=2, eps=1e-6):
    X = X.astype(np.float32)
    Xc = X - X.mean(0, keepdims=True)
    std = Xc.std(0, keepdims=True) + eps
    Xn = Xc / std
    U, S, Vt = np.linalg.svd(Xn, full_matrices=False)
    V = Vt[:r].astype(np.float32)
    for i in range(V.shape[0]):
        V[i] /= (np.linalg.norm(V[i]) + 1e-9)
    return V


def setup_logger(log_file: Optional[str]) -> logging.Logger:
    logger = logging.getLogger("bbq_layer_scan_no_greedy")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt); logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


def build_arc_prompt(context: str, question: str, options: List[str]) -> str:
    parts = []
    if context and context.strip():
        parts.append(context.strip())
    parts.append(question.strip())
    parts.append(f"(a) {options[0]}\n(b) {options[1]}\n(c) {options[2]}")
    parts.append("Answer with the option text only. Answer:")
    return "\n".join(parts)


def find_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    raise RuntimeError("Unsupported model structure.")


def _can_use_chat_template(tok) -> bool:
    template = getattr(tok, "chat_template", None)
    return callable(getattr(tok, "apply_chat_template", None)) and bool(template)


def _get_model_identifier(model_or_name) -> str:
    if isinstance(model_or_name, str):
        return model_or_name.lower()
    config = getattr(model_or_name, "config", None)
    name = ""
    if config is not None:
        name = getattr(config, "_name_or_path", "") or getattr(config, "name_or_path", "")
    if not name:
        name = getattr(model_or_name, "name_or_path", "")
    return str(name).lower()


def _is_instruct_model(model) -> bool:
    name = _get_model_identifier(model)
    if not name:
        return False
    instruct_keywords = ("inst", "instruct", "chat", "assistant", "aligned", "tuned")
    return any(keyword in name for keyword in instruct_keywords)


def format_prompt_for_model(prompt: str, model) -> str:
    model_name = _get_model_identifier(model)
    if model_name in RAW_PROMPT_MODELS:
        return prompt
    if _is_instruct_model(model):
        return prompt
    instruction = prompt.strip()
    return BASE_ALIGNMENT_TEMPLATE.format(system=BASE_SYSTEM_ALIGNMENT, instruction=instruction)


def _sum_vecs(vs: List[torch.Tensor]) -> torch.Tensor:
    if isinstance(vs, list):
        if len(vs) == 1: return vs[0]
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
        vec = _sum_vecs(vlist)
        unit = _safe_unit(vec.float())
        if unit is not None:
            units[li] = unit
    return units


def _orth_residual(sample_vec: Optional[torch.Tensor], probe_unit: Optional[torch.Tensor],
                   eps: float = 1e-6) -> Optional[torch.Tensor]:
    if sample_vec is None:
        return None
    sample_unit = _safe_unit(sample_vec.float(), eps=eps)
    if sample_unit is None:
        return None
    if probe_unit is None:
        return sample_unit
    proj = float(torch.dot(sample_unit, probe_unit).item())
    residual = sample_unit - proj * probe_unit
    return _safe_unit(residual, eps=eps)


def load_all_configs():
    return concatenate_datasets([load_dataset("heegyu/bbq", cfg, split="test") for cfg in TASKS])

@torch.no_grad()
def extract_vectors_soft_gate_bbq(ds_probe, tok, model, device: torch.device,
                                  head_repeat: int = 1, cv_folds: int = 2) -> Dict[int, List[torch.Tensor]]:
    layers = find_decoder_layers(model); L = len(layers); H = model.config.hidden_size

    idx_all = np.arange(len(ds_probe))
    folds = [idx_all] if cv_folds < 2 else np.array_split(np.random.default_rng(0).permutation(idx_all), cv_folds)
    diffs_tail = {l: [] for l in range(L)}; diffs_end = {l: [] for l in range(L)}

    for f in range(len(folds)):
        for ii in folds[f]:
            ex = ds_probe[int(ii)]
            context = ex.get("context",""); question = ex["question"]
            options  = [ex["ans0"], ex["ans1"], ex["ans2"]]
            plain_prompt = build_arc_prompt(context, question, options)

            # plain
            if _can_use_chat_template(tok):
                plain_ids = tok.apply_chat_template([{"role":"user","content":plain_prompt}],
                                                    add_generation_prompt=True, return_tensors="pt").to(device)
            else:
                aligned_prompt = format_prompt_for_model(plain_prompt, model)
                plain_ids = tok.encode(aligned_prompt, return_tensors="pt").to(device)
            out_plain = model(input_ids=plain_ids, output_hidden_states=True)
            hs_plain  = out_plain.hidden_states; plen = plain_ids.size(1)
            k_tail = length_adaptive_k(plen)
            plain_tail = [hs_plain[l+1][0, plen-k_tail:plen, :].detach().float().cpu().numpy().mean(axis=0) for l in range(L)]
            plain_end  = [hs_plain[l+1][0, plen-1, :].detach().float().cpu().numpy() for l in range(L)]
            del out_plain, hs_plain, plain_ids
            torch.cuda.empty_cache()

            for ep_text in EP_TEXTS:
                ep_prompt = (ep_text.strip() + "\n\n" + plain_prompt).strip()
                if _can_use_chat_template(tok):
                    ep_ids = tok.apply_chat_template([{"role":"user","content":ep_prompt}],
                                                     add_generation_prompt=True, return_tensors="pt").to(device)
                else:
                    aligned_ep_prompt = format_prompt_for_model(ep_prompt, model)
                    ep_ids = tok.encode(aligned_ep_prompt, return_tensors="pt").to(device)
                out_ep = model(input_ids=ep_ids, output_hidden_states=True)
                hs_ep  = out_ep.hidden_states; elen = ep_ids.size(1)
                k_tail_ep = length_adaptive_k(elen)
                for l in range(L):
                    h_ep_tail = hs_ep[l+1][0, elen-k_tail_ep:elen, :].detach().float().cpu().numpy().mean(axis=0)
                    h_ep_end  = hs_ep[l+1][0, elen-1, :].detach().float().cpu().numpy()
                    diffs_tail[l].append((h_ep_tail - plain_tail[l]).astype(np.float32))
                    diffs_end[l].append((h_ep_end  - plain_end[l]).astype(np.float32))
                del out_ep, hs_ep, ep_ids
                torch.cuda.empty_cache()

    layer_vecs: Dict[int, List[torch.Tensor]] = {}
    for l in range(L):
        Xt = np.stack(diffs_tail[l],0) if diffs_tail[l] else np.zeros((1,H),np.float32)
        Xe = np.stack(diffs_end[l],0)  if diffs_end[l]  else np.zeros((1,H),np.float32)
        X  = np.concatenate([Xt,Xe],0)
        V  = weighted_pca_np(X, r=2)
        layer_vecs[l] = [torch.from_numpy(V[i]) for i in range(V.shape[0])]
    return layer_vecs

@torch.no_grad()
def extract_single_sample_vecs(prompt_ids_cpu: torch.Tensor, plain_prompt: str, tok, model,
                               device: torch.device) -> Dict[int, torch.Tensor]:
    layers = find_decoder_layers(model); L = len(layers); H = model.config.hidden_size

    prompt_ids = prompt_ids_cpu.to(device)
    out_plain = model(input_ids=prompt_ids, output_hidden_states=True)
    hs_plain = out_plain.hidden_states
    plen = int(prompt_ids.size(1))
    k_tail = length_adaptive_k(plen)

    plain_tail = []
    plain_end = []
    for l in range(L):
        h_plain = hs_plain[l+1][0]
        tail = h_plain[plen-k_tail:plen, :].detach().float().cpu().numpy().mean(axis=0)
        end = h_plain[plen-1, :].detach().float().cpu().numpy()
        plain_tail.append(tail)
        plain_end.append(end)
    del out_plain, hs_plain, prompt_ids
    torch.cuda.empty_cache()

    diffs_tail = {l: [] for l in range(L)}
    diffs_end = {l: [] for l in range(L)}

    for ep_text in EP_TEXTS:
        ep_prompt = (ep_text.strip() + "\n\n" + plain_prompt).strip()
        if _can_use_chat_template(tok):
            ep_ids = tok.apply_chat_template([{"role": "user", "content": ep_prompt}],
                                             add_generation_prompt=True, return_tensors="pt").to(device)
        else:
            aligned_ep_prompt = format_prompt_for_model(ep_prompt, model)
            ep_ids = tok.encode(aligned_ep_prompt, return_tensors="pt").to(device)
        out_ep = model(input_ids=ep_ids, output_hidden_states=True)
        hs_ep = out_ep.hidden_states
        elen = int(ep_ids.size(1))
        k_tail_ep = length_adaptive_k(elen)

        for l in range(L):
            h_ep = hs_ep[l+1][0]
            ep_tail = h_ep[elen-k_tail_ep:elen, :].detach().float().cpu().numpy().mean(axis=0)
            ep_end = h_ep[elen-1, :].detach().float().cpu().numpy()
            diffs_tail[l].append((ep_tail - plain_tail[l]).astype(np.float32))
            diffs_end[l].append((ep_end - plain_end[l]).astype(np.float32))

        del out_ep, hs_ep, ep_ids
        torch.cuda.empty_cache()

    layer_vecs: Dict[int, torch.Tensor] = {}
    for l in range(L):
        Xt = np.stack(diffs_tail[l], 0) if diffs_tail[l] else np.zeros((1, H), np.float32)
        Xe = np.stack(diffs_end[l], 0) if diffs_end[l] else np.zeros((1, H), np.float32)
        X = np.concatenate([Xt, Xe], 0)
        V = weighted_pca_np(X, r=1)
        vec = torch.from_numpy(V[0])
        layer_vecs[l] = vec

    return layer_vecs

@torch.no_grad()
def alpha_for_prompt_last_token(model, layer_vecs, prompt_ids_cpu: torch.Tensor,
                                li: int, z_target: float, alpha_clip: Tuple[float,float], device) -> float:
    # move prompt to GPU only here
    prompt_ids = prompt_ids_cpu.to(device)
    out = model(input_ids=prompt_ids, output_hidden_states=True)
    hs  = out.hidden_states[li+1][0]  # [S,H]
    t_abs = int(prompt_ids.size(1) - 1)
    h = hs[t_abs, :]

    vlist = layer_vecs.get(li, None)
    if vlist is None: return 0.0
    v = (_sum_vecs(vlist) if isinstance(vlist, list) else vlist).to(h.device, h.dtype)

    c = float(v.norm().item()) + 1e-9
    u = v / c
    a  = float((h @ u).item())
    hn = float(h.norm().item()) + 1e-9
    Bn = math.sqrt(max(hn*hn - a*a, 0.0))
    s  = max(1e-6, min(1.0-1e-6, float(z_target)))
    T  = (Bn * s) / math.sqrt(max(1.0 - s*s, 1e-9))
    lo, hi = float(alpha_clip[0]), float(alpha_clip[1])
    alpha1 = max(0.0, (-a)/max(c,1e-9))
    alpha2 = (T - a)/max(c,1e-9)
    return float(max(lo, min(hi, max(alpha1, alpha2))))

# ——— Minimal steering hook for a single prompt position ———
def _register_prompt_hook(model, li: int, t_abs: int, dvec: torch.Tensor, alpha: float):
    layers = find_decoder_layers(model)
    vs = (alpha * dvec).detach()
    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output  # [B,S,H]
        B,S,H = hs.shape
        if not (0 <= t_abs < S): return output
        add = vs.view(1,1,-1).to(hs.device, hs.dtype)
        idx = torch.zeros((B,S,1), dtype=hs.dtype, device=hs.device)
        idx[:, t_abs, 0] = 1.0
        new_hs = hs + idx * add
        if isinstance(output, tuple): return (new_hs,) + output[1:]
        return new_hs
    return layers[li].register_forward_hook(hook)

@torch.inference_mode()
def _compute_lls_optionwise(tok, model, prompt_ids_cpu: torch.Tensor, cand_texts: List[str],
                            device, layer_vecs=None, blueprint=None, norm="avg") -> List[float]:
    """
    Compute LLs for each candidate separately (B=1). If blueprint is provided,
    it must be a list with a single {"li": int, "region": "prompt", "rel": 0, "w": float}.
    """
    pids = prompt_ids_cpu.to(device)
    pad_id = tok.pad_token_id or (tok.eos_token_id if tok.eos_token_id is not None else 0)
    p_len = int(pids.size(1))
    t_abs = p_len - 1

    # optional hook (single position at prompt end)
    handle = None
    if blueprint and layer_vecs:
        li = int(blueprint[0]["li"]); rel = int(blueprint[0]["rel"]); w = float(blueprint[0]["w"])
        if rel != 0:
            raise ValueError("This script only supports rel=0 (prompt last token).")
        vlist = layer_vecs.get(li, None)
        if vlist is not None and w >= 0.0:  # always inject; w can be 0 (no-op)
            dvec = (_sum_vecs(vlist) if isinstance(vlist, list) else vlist).to(pids.device, torch.float32)
            handle = _register_prompt_hook(model, li, t_abs, dvec, w)

    lls = []
    for text in cand_texts:
        ci = tok(text.strip(), return_tensors="pt", add_special_tokens=False).input_ids.to(device)  # [1,Lc]
        L_i = int(ci.size(1))
        S   = p_len + L_i
        input_ids = torch.full((1, S), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros_like(input_ids)
        input_ids[:, :p_len] = pids; attn_mask[:, :p_len] = 1
        input_ids[:, p_len:p_len+L_i] = ci[0]; attn_mask[:, :p_len+L_i] = 1

        out = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits[:, p_len-1:p_len+L_i-1, :]          # [1,L_i,V]
        logp = F.log_softmax(logits, dim=-1)
        tgt  = input_ids[:, p_len:p_len+L_i].unsqueeze(-1)      # [1,L_i,1]
        ll_total = float(logp.gather(-1, tgt).squeeze(-1).sum().item())
        lls.append(ll_total / L_i if (norm == "avg" and L_i > 0) else ll_total)

        del ci, input_ids, attn_mask, out, logits, logp, tgt
        torch.cuda.empty_cache()

    if handle is not None:
        handle.remove()
    return lls


def _apply_fixed_defaults_for_meta_llama(args) -> bool:
    """If model is Meta-Llama-3-8B-Instruct, hard-override a set of defaults in-code.
    Returns True if overrides applied."""
    m = _get_model_identifier(args.model)
    if ("meta-llama-3-8b-instruct" in m) or (m.strip() == "meta-llama/meta-llama-3-8b-instruct"):
        # Fixed defaults requested by user
        args.seed = 0
        args.n_probe = 200
        args.n_dev = 216
        args.layer_start = 18
        args.layer_stop = 25
        args.z_target = 0.85
        args.alpha_clip = "0,200"
        args.dtype = "bfloat16"
        args.device = "auto"
        # Provide a reasonable single-layer default (center of range) for convenience
        if getattr(args, "inject_layer", None) is None:
            args.inject_layer = 20
        return True
    return False


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["auto","float16","bfloat16","float32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-examples", type=int, default=-1)
    ap.add_argument("--context-filter", type=str, default="all", choices=["all","ambiguous","disambiguated"])
    ap.add_argument("--length-norm", type=str, default="avg", choices=["sum","avg"])
    ap.add_argument("--n_probe", type=int, default=200)
    ap.add_argument("--n_dev",   type=int, default=216)
    ap.add_argument("--layer_start", type=int, default=18)
    ap.add_argument("--layer_stop",  type=int, default=25)
    ap.add_argument("--inject-layer", type=int, default=None, help="Run ONLY this layer; overrides start/stop if set.")
    ap.add_argument("--z_target", type=float, default=0.85)
    ap.add_argument("--alpha_clip", type=str, default="0,200")
    ap.add_argument("--log-file", type=str, default=None)
    ap.add_argument("--save-report", "--save_report", type=str, default="")
    ap.add_argument("--residual-scale", type=float, default=RESIDUAL_SCALE)
    ap.add_argument("--residual-auto-flip", action="store_true")
    ap.add_argument("--residual-log-cos", action="store_true")
    args = ap.parse_args()

    # Apply fixed defaults for Meta-Llama-3-8B-Instruct (hard-coded override)
    fixed = _apply_fixed_defaults_for_meta_llama(args)

    logger = setup_logger(args.log_file)
    residual_scale = float(args.residual_scale)
    if args.residual_log_cos:
        logger.setLevel(logging.DEBUG)
    logger.info("[residual] scale=%.3f auto_flip=%s log_cos=%s", residual_scale, args.residual_auto_flip, args.residual_log_cos)

    if fixed:
        logger.info("[fixed-defaults] Applied Meta-Llama-3-8B-Instruct defaults: seed=%d, n_probe=%d, n_dev=%d, layers=%d..%d, z_target=%.2f, alpha_clip=%s, dtype=%s, device=%s, inject_layer=%s",
                    args.seed, args.n_probe, args.n_dev, args.layer_start, args.layer_stop,
                    args.z_target, args.alpha_clip, args.dtype, args.device, str(args.inject_layer))

    # seeds
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True; cudnn.benchmark = False
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    logger.info(f"[seed] {args.seed}")

    # data
    data_all = load_all_configs().shuffle(seed=args.seed)
    N_total = len(data_all)
    n_probe = min(max(0, args.n_probe), N_total)
    n_dev   = min(max(0, args.n_dev),   max(0, N_total - n_probe))
    ds_probe = data_all.select(range(0, n_probe))
    ds_dev   = data_all.select(range(n_probe, n_probe + n_dev))
    logger.info(f"[split] total={N_total} PROBE={len(ds_probe)} DEV={len(ds_dev)}")

    # model
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token  # harmless special-tokens warning
    dtype_map = {"auto": None, "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=(torch_dtype if torch_dtype is not None else None),
        device_map=("auto" if args.device == "auto" else None)
    )
    device = model.device
    model.eval()
    model.config.use_cache = False
    logger.info("[model] %s loaded on %s", args.model, device)

    # subspace extraction (global, across layers)
    t0 = time.time()
    layer_vecs = extract_vectors_soft_gate_bbq(ds_probe, tok, model, device, head_repeat=1, cv_folds=2)
    torch.cuda.empty_cache()
    logger.info(f"[subspace] extracted in {time.time()-t0:.2f}s | layers={len(layer_vecs)}")
    probe_units = _probe_unit_vectors(layer_vecs)

    # prepare DEV examples on CPU
    examples = []
    for i, ex in enumerate(ds_dev):
        if 0 <= args.max_examples <= i: break
        cc = (ex.get("context_condition") or "").lower()
        if args.context_filter == "ambiguous" and "ambiguous" not in cc: continue
        if args.context_filter == "disambiguated" and "disambiguated" not in cc: continue
        context = ex.get("context",""); question = ex["question"]
        options = [ex["ans0"], ex["ans1"], ex["ans2"]]; gold = int(ex["label"])
        prompt = build_arc_prompt(context, question, options)
        if _can_use_chat_template(tok):
            prompt_ids_cpu = tok.apply_chat_template([{"role":"user","content":prompt}],
                                                     add_generation_prompt=True, return_tensors="pt")
        else:
            aligned_prompt = format_prompt_for_model(prompt, model)
            prompt_ids_cpu = tok.encode(aligned_prompt, return_tensors="pt")
        cand_texts = [" " + o for o in options]
        examples.append({
            "prompt_ids_cpu": prompt_ids_cpu,
            "cand_texts": cand_texts,
            "gold": gold,
            "plain_prompt": prompt,
        })

    if not examples:
        logger.info("No DEV examples after filtering; exit.")
        return

    # baseline (no steering), option-wise
    total = 0; correct = 0
    logger.info("[baseline] computing (option-wise)...")
    for ex in examples:
        lls = _compute_lls_optionwise(tok, model, ex["prompt_ids_cpu"], ex["cand_texts"],
                                      device, layer_vecs=None, blueprint=None, norm=args.length_norm)
        if not lls: continue
        gold = ex["gold"]
        pred = int(max(range(len(lls)), key=lambda j: lls[j]))
        total += 1; correct += (1 if pred == gold else 0)
    baseline_acc = correct / max(1, total)
    logger.info("[baseline] N=%d | acc=%.4f", total, baseline_acc)

    logger.info("[sample-subspace] extracting per-example orth residual vectors (%d examples)...", len(examples))
    for idx, ex in enumerate(examples):
        sample_vecs = extract_single_sample_vecs(ex["prompt_ids_cpu"], ex["plain_prompt"], tok, model, device)
        residual_vecs = {}
        for li, sample_vec in sample_vecs.items():
            residual_vec = _orth_residual(sample_vec, probe_units.get(li))
            if residual_vec is not None:
                residual_vecs[li] = residual_vec
        ex["residual_vecs"] = residual_vecs
        if (idx + 1) % 25 == 0 or (idx + 1) == len(examples):
            logger.info("  processed %d/%d", idx + 1, len(examples))
    torch.cuda.empty_cache()

    # Decide which layer(s) to evaluate
    Lmax = len(find_decoder_layers(model)) - 1
    Lstart = max(0, int(args.layer_start)); Lstop = min(Lmax, int(args.layer_stop))

    if args.inject_layer is not None:
        li_fixed = max(0, min(Lmax, int(args.inject_layer)))
        layer_indices = [li_fixed] if li_fixed in layer_vecs else []
    else:
        layer_indices = [li for li in range(Lstart, Lstop+1) if li in layer_vecs]
    range_desc = (str(layer_indices[0]) if len(layer_indices) == 1 else f"{Lstart}..{Lstop}")

    report = []
    total_examples = max(1, total)

    for li in layer_indices:
        global_correct = 0
        global_alpha_sum = 0.0
        global_nonzero = 0
        mix_correct = 0
        mix_alpha_sum = 0.0
        mix_nonzero = 0
        mix_residual_used = 0
        mix_flip_count = 0
        mix_cos_total = 0
        mix_cos_neg = 0

        probe_unit = probe_units.get(li)
        probe_unit_norm = float(probe_unit.norm().item()) if probe_unit is not None else 0.0

        for ex_idx, ex in enumerate(examples):
            prompt_ids_cpu = ex["prompt_ids_cpu"]; cand_texts = ex["cand_texts"]; gold = ex["gold"]

            # global probe only (SINGLE-LAYER injection at li)
            a_global = alpha_for_prompt_last_token(
                model, layer_vecs, prompt_ids_cpu, li=li,
                z_target=float(args.z_target), alpha_clip=tuple(float(x) for x in str(args.alpha_clip).split(",")), device=device
            )
            global_alpha_sum += a_global
            if a_global > 0.0:
                global_nonzero += 1

            blueprint_global = [{"li": li, "region": "prompt", "rel": 0, "w": float(a_global)}]
            lls_global = _compute_lls_optionwise(
                tok, model, prompt_ids_cpu, cand_texts,
                device, layer_vecs=layer_vecs, blueprint=blueprint_global, norm=args.length_norm
            )

            pred_global = None
            if lls_global:
                pred_global = int(max(range(len(lls_global)), key=lambda j: lls_global[j]))
                if pred_global == gold:
                    global_correct += 1

            # optional: +orth residual (still SINGLE-LAYER li)
            residual_vec = ex.get("residual_vecs", {}).get(li)
            if residual_vec is None:
                mix_alpha_sum += a_global
                if a_global > 0.0:
                    mix_nonzero += 1
                if pred_global is not None and pred_global == gold:
                    mix_correct += 1
                continue

            mix_residual_used += 1
            combined_vlist = list(layer_vecs.get(li, []))

            cos_orig = None
            if probe_unit is not None and probe_unit_norm > 0.0:
                res_norm = float(residual_vec.norm().item())
                if res_norm > 0.0:
                    denom = (res_norm * probe_unit_norm) + 1e-9
                    cos_orig = float(torch.dot(residual_vec, probe_unit).item() / denom)
            if cos_orig is not None:
                mix_cos_total += 1
                if cos_orig < 0.0:
                    mix_cos_neg += 1

            residual_vec_use = residual_vec
            flipped = False
            if args.residual_auto_flip and cos_orig is not None and cos_orig < 0.0:
                residual_vec_use = -residual_vec_use
                flipped = True
                mix_flip_count += 1
                cos_logged = -cos_orig
            else:
                cos_logged = cos_orig

            if args.residual_log_cos:
                logger.info(
                    "[residual-cos] layer=%02d sample=%03d cos_orig=%s cos_final=%s flipped=%s",
                    li, ex_idx,
                    f"{cos_orig:.4f}" if cos_orig is not None else "nan",
                    f"{cos_logged:.4f}" if cos_logged is not None else "nan",
                    flipped,
                )

            combined_vlist.append(residual_scale * residual_vec_use)
            layer_vecs_mix = {li: combined_vlist}

            a_mix = alpha_for_prompt_last_token(
                model, layer_vecs_mix, prompt_ids_cpu, li=li,
                z_target=float(args.z_target), alpha_clip=tuple(float(x) for x in str(args.alpha_clip).split(",")), device=device
            )
            mix_alpha_sum += a_mix
            if a_mix > 0.0:
                mix_nonzero += 1

            blueprint_mix = [{"li": li, "region": "prompt", "rel": 0, "w": float(a_mix)}]
            lls_mix = _compute_lls_optionwise(
                tok, model, prompt_ids_cpu, cand_texts,
                device, layer_vecs=layer_vecs_mix, blueprint=blueprint_mix, norm=args.length_norm
            )
            if lls_mix:
                pred_mix = int(max(range(len(lls_mix)), key=lambda j: lls_mix[j]))
                if pred_mix == gold:
                    mix_correct += 1

        global_acc = global_correct / total_examples
        mix_acc = mix_correct / total_examples

        row_global = {
            "acc": round(global_acc, 6),
            "delta": round(global_acc - baseline_acc, 6),
            "alpha_mean": round(global_alpha_sum / total_examples, 6),
            "alpha_nonzero_rate": round(global_nonzero / total_examples, 6),
        }
        row_mix = {
            "acc": round(mix_acc, 6),
            "delta": round(mix_acc - baseline_acc, 6),
            "alpha_mean": round(mix_alpha_sum / total_examples, 6),
            "alpha_nonzero_rate": round(mix_nonzero / total_examples, 6),
            "residual_usage_rate": round(mix_residual_used / total_examples, 6),
            "residual_flip_rate": round(mix_flip_count / max(1, mix_cos_total), 6) if mix_cos_total else 0.0,
            "residual_neg_rate": round(mix_cos_neg / max(1, mix_cos_total), 6) if mix_cos_total else 0.0,
        }
        row = {
            "layer": li,
            "N": total,
            "baseline_acc": round(baseline_acc, 6),
            "global": row_global,
            "orth_residual": row_mix,
            "alpha_z_target": float(args.z_target),
            "alpha_clip": [float(x) for x in str(args.alpha_clip).split(",")],
            "residual_scale": residual_scale,
            "residual_auto_flip": bool(args.residual_auto_flip),
        }
        report.append(row)
        logger.info(
            "[layer %02d] global acc=%.4f Δ=%.4f ᾱ=%.3f nonzero=%.3f | +res acc=%.4f Δ=%.4f ᾱ=%.3f nonzero=%.3f use=%.3f flip=%.3f neg=%.3f",
            li,
            row_global["acc"], row_global["delta"], row_global["alpha_mean"], row_global["alpha_nonzero_rate"],
            row_mix["acc"], row_mix["delta"], row_mix["alpha_mean"], row_mix["alpha_nonzero_rate"],
            row_mix["residual_usage_rate"], row_mix["residual_flip_rate"], row_mix["residual_neg_rate"],
        )

    report.sort(key=lambda r: r["layer"])
    print(f"\n=== BBQ — prompt last-token scan (ALWAYS inject) — layer(s) {range_desc} ===")
    print(f"Baseline accuracy: {baseline_acc:.4f}  (N={total})\n")
    print("Per-layer results (global vs + orth residual):")
    for r in report:
        g = r["global"]; m = r["orth_residual"]
        print(
            f"Layer {r['layer']:>2d}: global acc={g['acc']:.4f} Δ={g['delta']:+.4f} ᾱ={g['alpha_mean']:.3f} nonzero={g['alpha_nonzero_rate']:.3f} | "
            f"+res acc={m['acc']:.4f} Δ={m['delta']:+.4f} ᾱ={m['alpha_mean']:.3f} nonzero={m['alpha_nonzero_rate']:.3f} use={m['residual_usage_rate']:.3f} flip={m['residual_flip_rate']:.3f} neg={m['residual_neg_rate']:.3f}"
        )

    if args.save_report:
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump({"baseline_acc": baseline_acc, "results": report}, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report to: {args.save_report}")

if __name__ == "__main__":
    main()
