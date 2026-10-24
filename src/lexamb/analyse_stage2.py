"""Stage 2: WordNet-derived generalisation pool (spec §7 Stage 2). NOT a replication of Rodd et al.

Inferential model per representation-derived score (each proxy is a one-number-per-word summary, so
no OLS on thousands of hidden dimensions is ever fitted):

    z(proxy) ~ z(log1p wn_n_all) + z(GCIDE etymologies) + z(log_subtlex) + z(length) + z(concreteness)

OLS with HC3 SEs, VIFs reported. For the higher-is-harder proxies the Rodd-compatible pattern is
beta(sense count) < 0 and beta(etymologies) > 0. GCIDE etymology counts are a computed proxy for
unrelated meanings, not a human norm; zero = "no etymology recorded" and is handled by policy.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import stats_utils as su
from .audit_data import EvalSet
from .analyse_rodd_factorial import SCALAR_PROXIES

LOG = logging.getLogger("lexamb")


def _variants(cfg: dict) -> list[dict]:
    prim = cfg["probes"]["primary_controls"]
    exp = cfg["probes"]["expanded_controls"]
    return [
        {"variant": "primary", "eval_set": "der_matched", "sense": "log1p_wn_n_all", "meaning": "gcide_nonzero",
         "extra": [], "controls": prim, "note": "matched range; GCIDE zeros & missing excluded"},
        {"variant": "zero_recoded_to_1", "eval_set": "der_matched", "sense": "log1p_wn_n_all",
         "meaning": "gcide_recode", "extra": [], "controls": prim, "note": "GCIDE 0 -> 1"},
        {"variant": "zero_as_category", "eval_set": "der_matched", "sense": "log1p_wn_n_all",
         "meaning": "gcide_recode", "extra": ["gcide_zero"], "controls": prim,
         "note": "GCIDE 0 -> 1 plus a 'no etymology recorded' indicator"},
        {"variant": "full_pool", "eval_set": "der_full", "sense": "log1p_wn_n_all", "meaning": "gcide_nonzero",
         "extra": [], "controls": prim, "note": "all derived words; zeros excluded"},
        {"variant": "expanded_controls", "eval_set": "der_matched", "sense": "log1p_wn_n_all",
         "meaning": "gcide_nonzero", "extra": [], "controls": exp, "note": "adds ortho_n, n_syll, n_phon"},
        {"variant": "noun_senses", "eval_set": "der_matched", "sense": "log1p_wn_n_noun",
         "meaning": "gcide_nonzero", "extra": [], "controls": prim, "note": "noun sense count (secondary construct)"},
        {"variant": "semantic_distance", "eval_set": "der_matched", "sense": "log1p_wn_n_all",
         "meaning": "sim_top2", "extra": [], "controls": prim,
         "note": "RQ4: similarity of the top-2 noun senses instead of etymologies (lower = more distant)"},
    ]


def run_stage2(model_meta: dict, prox: pd.DataFrame, eval_sets: dict[str, EvalSet], cfg: dict,
               ease: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for v in _variants(cfg):
        es = eval_sets.get(v["eval_set"])
        if es is None:
            continue
        it = es.items.merge(prox.drop(columns=["word"], errors="ignore"), on="word_idx", how="left")
        if ease is not None:
            it = it.merge(ease, on="word_idx", how="left")
        proxies = [c for c in SCALAR_PROXIES + ["ease_rating"] if c in it and it[c].notna().any()]
        preds = [v["sense"], v["meaning"]] + v["extra"] + [c for c in v["controls"] if c not in (v["sense"],)]
        for oc in proxies:
            d = it[[oc] + preds].dropna()
            if len(d) < len(preds) + 10:
                continue
            Z = np.column_stack([np.ones(len(d))] + [su_z(d[c]) for c in preds])
            est, res = su.ols_hc3(su_z(d[oc]), Z, ["intercept"] + preds)
            vifs = su.vif(Z[:, 1:])
            for k, c in enumerate(preds):
                rows.append({"model_id": model_meta["model_id"], "family": model_meta["family"],
                             "variant_model": model_meta["variant"], "size": model_meta["size"],
                             "analysis_variant": v["variant"], "eval_set": v["eval_set"], "pool": "derived",
                             "outcome": oc, "predictor": c,
                             "role": "sense_count" if c == v["sense"] else ("meanings_proxy" if c == v["meaning"]
                                                                           else "control"),
                             "n": len(d), "beta_z": est[c]["beta"], "se_hc3": est[c]["se"], "p_hc3": est[c]["p"],
                             "ci_lo": est[c]["ci_lo"], "ci_hi": est[c]["ci_hi"], "vif": vifs[k],
                             "r2_model": float(res.rsquared), "note": v["note"]})
    df = pd.DataFrame(rows)
    if len(df):
        key = ["analysis_variant", "predictor"]
        df["q_bh_proxies"] = df.groupby(key).p_hc3.transform(lambda p: su.bh_fdr(p.to_numpy()))
    return df


def su_z(x) -> np.ndarray:
    x = np.asarray(x, float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * 0
