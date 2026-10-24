"""Output-probability measures (spec §6, Outcome C) -- secondary, analysed separately.

Scoring rule (identical for every item within a model):
  * prefix  = everything before the target word in the model's neutral prompt (``Word:`` for base
    models, the native chat wrapper + ``Word:`` for instruct models) -- common to all items;
  * word log-likelihood = sum over target subtokens t of log p(x_t | x_<t);
  * length-normalised = that sum / n_subtokens  (reported with n_subtokens; tokeniser-sensitive);
  * next-token entropy at the last target subtoken (nats).
These are lexical-form/predictability measures, NOT ambiguity measures and NOT reaction times.

Logit lens (secondary settling proxy): at the last target subtoken, every layer's state is passed
through the final norm + unembedding; KL(p_final || p_layer) per layer. Its area over relative depth
is an output-space analogue of settling time (higher = later convergence).
"""
from __future__ import annotations

import numpy as np
import torch


def word_loglik(logprobs_prev: torch.Tensor, ids: list[int], target_idx: list[int]) -> tuple[float, float, int]:
    """logprobs_prev: (T, V) log-softmax over the item's own sequence (position t predicts t+1)."""
    lp = [float(logprobs_prev[t - 1, ids[t]]) for t in target_idx if t >= 1]
    if len(lp) != len(target_idx):
        return float("nan"), float("nan"), len(target_idx)
    return float(np.sum(lp)), float(np.mean(lp)), len(lp)


def entropy(logits_row: torch.Tensor) -> float:
    lp = torch.log_softmax(logits_row.float(), dim=-1)
    return float(-(lp.exp() * lp).sum())


def find_final_norm(model):
    for path in ("model.norm", "model.final_layernorm", "transformer.ln_f"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    return None


def final_state_is_normed(model, h_last: torch.Tensor, logits_last: torch.Tensor) -> bool:
    """HF decoders usually apply the final norm before appending the last hidden state; verify it."""
    head = model.get_output_embeddings()
    approx = head(h_last.to(head.weight.dtype)).float()
    return bool(torch.allclose(approx, logits_last.float(), rtol=2e-2, atol=2e-2 * logits_last.float().abs().max()))


def logit_lens_kl(model, norm, states: torch.Tensor, final_normed: bool) -> np.ndarray:
    """states: (L+1, d) at one position. Returns KL(p_L || p_l) for l = 0..L (nats)."""
    head = model.get_output_embeddings()
    x = states.to(head.weight.dtype)
    normed = norm(x)
    if final_normed:
        normed[-1] = x[-1]
    lp = torch.log_softmax(head(normed).float(), dim=-1)
    p_final = lp[-1].exp()
    kl = (p_final * (lp[-1] - lp)).sum(-1)
    return kl.clamp_min(0).cpu().numpy()
