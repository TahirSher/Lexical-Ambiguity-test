"""Outcome D (exploratory only): model-rated recognition ease, instruct models only (spec §6 D).

Each word is an independent request. No label or count is shown. Instead of sampling text, the
rating is read deterministically from the next-token distribution restricted to the fixed response
vocabulary "1".."7": expected rating = sum_k k p(k) / sum_k p(k). Paraphrases are prompt-sensitivity
conditions, not independent participants; inter-paraphrase agreement is reported. Higher = EASIER.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

LOG = logging.getLogger("lexamb")


def rate_words(tok, model, words: pd.DataFrame, cfg: dict, device: str) -> pd.DataFrame:
    er = cfg["ease_rating"]
    scale = er["scale"]
    ids = []
    for k in scale:
        enc = tok.encode(str(k), add_special_tokens=False)
        if len(enc) != 1:
            raise RuntimeError(f"rating token {k!r} is not a single token for this tokenizer")
        ids.append(enc[0])
    kwargs = dict(cfg["extraction"].get("chat_template_kwargs") or {})
    rows = []
    for p_i, para in enumerate(er["paraphrases"]):
        for w_idx, w in zip(words.word_idx, words.word_lc):
            msgs = [{"role": "user", "content": para.format(word=w)}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kwargs)
            enc = tok(text, add_special_tokens=False, return_tensors="pt").to(device)
            with torch.inference_mode():
                logits = model(**enc).logits[0, -1].float()
            p = torch.softmax(logits, -1)[ids].cpu().numpy()
            mass = float(p.sum())
            rows.append({"word_idx": int(w_idx), "paraphrase": p_i, "rating_mass": mass,
                         "expected_rating": float(np.dot(scale, p) / max(mass, 1e-12))})
    df = pd.DataFrame(rows)
    wide = df.pivot(index="word_idx", columns="paraphrase", values="expected_rating")
    r = wide.corr(method="spearman").to_numpy()
    LOG.info("ease rating inter-paraphrase Spearman (mean off-diagonal) = %.3f",
             np.nanmean(r[np.triu_indices_from(r, 1)]) if len(r) > 1 else np.nan)
    out = df.groupby("word_idx").agg(ease_rating=("expected_rating", "mean"),
                                     ease_sd_paraphrase=("expected_rating", "std"),
                                     ease_mass=("rating_mass", "mean")).reset_index()
    return out
