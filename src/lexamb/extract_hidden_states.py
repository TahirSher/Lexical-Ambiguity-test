"""Target-subtoken hidden states for every layer (spec §5), resumable and sharded.

Layer convention (checked at runtime, not assumed): ``hidden_states`` has L+1 entries; index 0 is
the embedding output, 1..L are transformer block outputs. In HF Llama/Qwen/SmolLM decoders the
final entry already has the final norm applied -- this is verified numerically and logged.

Unit of extraction = unique lowercase word form (the prompt depends only on the word, so duplicate
rows would be identical forward passes). Row-level metadata is joined back at analysis time.
Each item is its own sequence; batching only right-pads independent sequences under an attention
mask (causal attention => padding never influences real positions).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import atomic_save_npy, atomic_to_parquet, sha256_file, sha256_text, slug, write_json
from .model_registry import ModelSpec, load_model
from .prompt_formatting import chat_template_hash, render
from .score_token_likelihood import (entropy, final_state_is_normed, find_final_norm, logit_lens_kl,
                                     word_loglik)

LOG = logging.getLogger("lexamb")
FP16_MAX = 65000.0


def cache_key(meta: dict, cfg_ext: dict, tok_hash: str) -> str:
    payload = json.dumps({
        "model": meta["model_id"], "commit": meta["model_commit"], "tok": meta["tokenizer_commit"],
        "dtype": meta["dtype"], "base_tpl": cfg_ext["base_template"], "chat_tpl": cfg_ext["chat_content_template"],
        "chat_kwargs": cfg_ext.get("chat_template_kwargs"), "chat_template_hash": tok_hash,
        "poolings": cfg_ext["poolings"], "variant": meta["variant"]}, sort_keys=True)
    return sha256_text(payload)[:12]


def model_dir(act_root: Path, model_id: str) -> Path:
    """Most recent completed extraction directory for a model (analysis-time lookup)."""
    cands = sorted(act_root.glob(f"{slug(model_id)}__*/manifest.json"), key=lambda p: p.stat().st_mtime)
    # exact match on the recorded model id (a slug prefix could belong to a different model)
    cands = [p for p in cands if json.loads(p.read_text()).get("meta", {}).get("model_id") == model_id
             and json.loads(p.read_text()).get("complete")]
    if not cands:
        raise FileNotFoundError(f"no activations for {model_id} under {act_root}")
    return cands[-1].parent


def _validate_shard(d: Path, entry: dict) -> bool:
    try:
        return all(sha256_file(d / f) == h for f, h in entry["files"].items())
    except FileNotFoundError:
        return False


def extract_model(spec: ModelSpec, words: pd.DataFrame, cfg: dict, act_root: Path, device: str,
                  resume: bool = True) -> Path:
    import torch

    ext = cfg["extraction"]
    tok, model, meta = load_model(spec, ext["revision"], ext["dtype"], device)
    thash = chat_template_hash(tok) if spec.variant == "instruct" else sha256_text(ext["base_template"])
    key = cache_key(meta, ext, thash)
    out = act_root / f"{slug(spec.id)}__{key}"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if (resume and manifest_path.exists()) else {"shards": {}}
    meta.update({"prompt_template_hash": thash, "cache_key": key,
                 "chat_template": tok.chat_template if spec.variant == "instruct" else None,
                 "chat_template_kwargs": ext.get("chat_template_kwargs") if spec.variant == "instruct" else None,
                 "base_template": ext["base_template"] if spec.variant == "base" else None,
                 "layer_convention": "layer 0 = embedding output; layers 1..L = transformer block outputs"})

    # --- render every prompt first (cheap) and write the alignment audit -------------------------
    rendered = [render(tok, w, spec.variant, ext) for w in words.word_lc]
    align = pd.DataFrame([{
        "word_idx": int(i), "word": r.word, "ok": r.ok, "reason": r.reason, "n_tokens": len(r.input_ids),
        "target_idx": json.dumps(r.target_idx), "n_subtokens": len(r.target_idx),
        "target_tokens": json.dumps(tok.convert_ids_to_tokens([r.input_ids[k] for k in r.target_idx])
                                    if r.target_idx else []),
        "char_span": json.dumps(r.char_span)} for i, r in zip(words.word_idx, rendered)])
    align.to_csv(out / "alignment.csv", index=False)
    rng = np.random.default_rng(cfg["seed"])
    n_audit = min(ext["alignment_audit_n"], len(align))
    audit = align.iloc[np.sort(rng.choice(len(align), n_audit, replace=False))].copy()
    audit["rendered_prompt"] = [rendered[i].text for i in audit.index]
    audit["all_tokens"] = [json.dumps(tok.convert_ids_to_tokens(rendered[i].input_ids)) for i in audit.index]
    audit["manual_check_ok"] = ""   # to be filled by a human reviewer (spec §5.5)
    audit.to_csv(out / "alignment_manual_audit.csv", index=False)
    failures = align[~align.ok]
    failures.to_csv(out / "extraction_failures.csv", index=False)
    if len(failures):
        LOG.warning("%s: %d/%d words failed alignment (excluded; see extraction_failures.csv)",
                    spec.id, len(failures), len(align))
    example = rendered[0]
    meta["example_prompt"] = example.text
    meta["example_input_ids"] = example.input_ids

    norm = find_final_norm(model) if ext.get("logit_lens", True) else None
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    L = meta["n_layers"]
    final_normed = None
    shard_size = ext["shard_size"]
    t0 = time.time()
    for s_start in range(0, len(words), shard_size):
        s_idx = list(range(s_start, min(s_start + shard_size, len(words))))
        sname = f"shard_{s_start // shard_size:04d}"
        if resume and sname in manifest["shards"] and _validate_shard(out, manifest["shards"][sname]):
            LOG.info("%s %s: valid shard exists, skipping", spec.id, sname)
            continue
        acts = {p: np.full((len(s_idx), L + 1, meta["hidden_size"]), np.nan, dtype=np.float32)
                for p in ext["poolings"]}
        rows = []
        for b_start in range(0, len(s_idx), ext["batch_size"]):
            batch = [i for i in s_idx[b_start:b_start + ext["batch_size"]] if rendered[i].ok]
            if not batch:
                continue
            maxlen = max(len(rendered[i].input_ids) for i in batch)
            ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
            att = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for j, i in enumerate(batch):
                seq = rendered[i].input_ids
                ids[j, :len(seq)] = torch.tensor(seq)
                att[j, :len(seq)] = 1
            with torch.inference_mode():
                o = model(input_ids=ids.to(device), attention_mask=att.to(device),
                          output_hidden_states=True, use_cache=False)
            hs = o.hidden_states
            if len(hs) != L + 1:
                raise RuntimeError(f"{spec.id}: expected {L + 1} hidden states, got {len(hs)}")
            for j, i in enumerate(batch):
                r = rendered[i]
                tidx = torch.tensor(r.target_idx, device=hs[0].device)
                stack_all = torch.stack([h[j, tidx].float() for h in hs])          # (L+1, n_sub, d)
                last = stack_all[:, -1]
                if "last" in acts:
                    acts["last"][i - s_start] = last.cpu().numpy()
                if "mean" in acts:
                    acts["mean"][i - s_start] = stack_all.mean(1).cpu().numpy()
                logits = o.logits[j, :len(r.input_ids)]
                logprobs = torch.log_softmax(logits.float(), dim=-1)
                ll_sum, ll_mean, n_sc = word_loglik(logprobs, r.input_ids, r.target_idx)
                row = {"word_idx": int(words.word_idx.iloc[i]), "word": r.word, "status": "ok",
                       "n_subtokens": len(r.target_idx), "target_last": r.last, "seq_len": len(r.input_ids),
                       "loglik_sum": ll_sum, "loglik_mean": ll_mean, "surprisal_mean": -ll_mean,
                       "next_entropy": entropy(logits[r.last])}
                if norm is not None:
                    if final_normed is None:
                        final_normed = final_state_is_normed(model, hs[-1][j, r.last], logits[r.last])
                        meta["final_hidden_state_is_normed"] = final_normed
                        LOG.info("%s: final hidden state already normed = %s", spec.id, final_normed)
                    row["logit_lens_kl"] = json.dumps([round(float(v), 6) for v in
                                                       logit_lens_kl(model, norm, last.to(hs[0].dtype), final_normed)])
                rows.append(row)
        for i in s_idx:
            if not rendered[i].ok:
                rows.append({"word_idx": int(words.word_idx.iloc[i]), "word": rendered[i].word,
                             "status": f"alignment_failed: {rendered[i].reason}"})
        files = {}
        for p, a in acts.items():
            finite = a[np.isfinite(a)]
            overflow = finite.size and np.abs(finite).max() > FP16_MAX
            store = a.astype(np.float32 if (overflow or ext["storage_dtype"] == "float32") else np.float16)
            if overflow:
                LOG.warning("%s %s: |activation| > fp16 range, shard stored as float32", spec.id, sname)
            f = f"{sname}_{p}.npy"
            atomic_save_npy(out / f, store)
            files[f] = sha256_file(out / f)
        mf = f"{sname}_meta.parquet"
        atomic_to_parquet(pd.DataFrame(rows).sort_values("word_idx"), out / mf)
        files[mf] = sha256_file(out / mf)
        manifest["shards"][sname] = {"start": s_start, "n": len(s_idx), "files": files}
        manifest["meta"] = meta
        write_json(manifest_path, manifest)
        LOG.info("%s %s done (%d items, %.1fs elapsed)", spec.id, sname, len(s_idx), time.time() - t0)
    manifest["meta"] = meta
    manifest["n_words"] = len(words)
    manifest["complete"] = True
    write_json(manifest_path, manifest)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def load_activations(d: Path, pooling: str) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Returns (acts[n_words, L+1, d] float32, per-word meta, manifest meta)."""
    manifest = json.loads((d / "manifest.json").read_text())
    shards = sorted(manifest["shards"].items(), key=lambda kv: kv[1]["start"])
    acts = np.concatenate([np.load(d / f"{name}_{pooling}.npy").astype(np.float32) for name, _ in shards])
    meta = pd.concat([pd.read_parquet(d / f"{name}_meta.parquet") for name, _ in shards], ignore_index=True)
    meta = meta.sort_values("word_idx").reset_index(drop=True)
    return acts, meta, manifest["meta"]
