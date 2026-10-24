"""Stage 1: Rodd et al. (2002) calibration (spec §7 Stage 1, RQ3, RQ6).

Three analyses, kept strictly separate (spec §12):

1. **Directional dissociation on a pre-registered model-derived proxy** (RQ6).
   Primary proxy = representational settling cost (RSC), motivated by attractor accounts of
   ambiguity effects (Rodd, Gaskell & Marslen-Wilson 2004, Cognitive Science; Armstrong & Plaut 2016,
   Lang. Cogn. Neurosci.) and by the view of the residual stream as iterative refinement
   (Jastrzebski et al., ICLR 2018). For word w with target state h_l at layer l:
       z_l = (h_l - mean_l) / sd_l   (per layer, per dimension, over all words; label-free; removes the
                                      rogue/massive dimensions that dominate raw cosine -- Timkey &
                                      van Schijndel, EMNLP 2021)
       d_l = 1 - cos(z_l, z_L)       ("distance from the final state" at depth l)
       RSC = integral_0^1 d(depth) d depth   (trapezoid over relative depth)
   Higher RSC = the representation reaches its final configuration later = MORE DIFFICULT.
   Expected sign: senses(many) < 0 (facilitation), ambiguity(ambiguous) > 0 (inhibition).
   Both factors are fitted in ONE model with effect coding (+-0.5) and their interaction.
   Secondary proxies: logit-lens KL area, word surprisal, next-token entropy, (ease rating).
   NONE of these is a reaction time.

2. **Time course** (novel, exploratory): the same factorial model on d_l at every layer tests the
   semantic-settling-dynamics prediction (Armstrong & Plaut 2016) that the polysemy advantage
   emerges earlier than the homonymy disadvantage: effect-weighted depth of the ambiguity effect minus
   that of the senses effect (> 0 predicted), with a cell-stratified bootstrap CI.

3. **Representational geometry of the 2x2 design**
   * factorial selectivity: out-of-fold probe scores for ambiguity and for senses regressed on both
     factors (does the ambiguity read-out also respond to senses, and vice versa?);
   * cross-condition generalisation performance, CCGP (Bernardi et al., Cell 2020): train an
     ambiguity decoder on few-sense words and test on many-sense words (and the reverse), etc. High
     CCGP means the factor is coded along a direction that is abstract w.r.t. the other factor.
"""
from __future__ import annotations

import json
import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats_utils as su
from .audit_data import EvalSet

LOG = logging.getLogger("lexamb")

SCALAR_PROXIES = ["rsc_last", "rsc_mean", "kl_auc", "surprisal_mean", "next_entropy"]
PROXY_ORIENTATION = {"rsc_last": "higher = harder (slower settling)", "rsc_mean": "higher = harder (slower settling)",
                     "kl_auc": "higher = harder (later output convergence)",
                     "surprisal_mean": "higher = less predictable word form",
                     "next_entropy": "higher = more uncertain continuation",
                     "ease_rating": "higher = EASIER (reverse of the others)"}


# ============================================================================ proxies
def settling_profile(acts: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """d_l = 1 - cos(z_l, z_L) for every word and layer; acts (n, L+1, d). Invalid rows -> NaN."""
    n, L1, _ = acts.shape
    out = np.full((n, L1), np.nan)
    zL = None
    for l in range(L1 - 1, -1, -1):
        h = acts[valid, l].astype(np.float64)
        mu, sd = su.standardise_fit(h)
        z = (h - mu) / sd
        z /= np.linalg.norm(z, axis=1, keepdims=True).clip(1e-12)
        if zL is None:
            zL = z
        out[valid, l] = 1 - np.sum(z * zL, axis=1)
    return out


def compute_proxies(acts_last: np.ndarray, acts_mean: np.ndarray | None, word_meta: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    valid = (word_meta.status == "ok").to_numpy()
    L1 = acts_last.shape[1]
    depth = np.arange(L1) / (L1 - 1)
    d_last = settling_profile(acts_last, valid)
    prox = word_meta[["word_idx", "word", "status"]].copy()
    prox["n_subtokens"] = word_meta.get("n_subtokens")
    prox["rsc_last"] = np.trapezoid(d_last, depth, axis=1)
    if acts_mean is not None:
        prox["rsc_mean"] = np.trapezoid(settling_profile(acts_mean, valid), depth, axis=1)
    for c in ("surprisal_mean", "next_entropy", "loglik_sum"):
        if c in word_meta:
            prox[c] = word_meta[c].to_numpy()
    if "logit_lens_kl" in word_meta:
        kl = np.array([json.loads(s) if isinstance(s, str) else [np.nan] * L1 for s in word_meta.logit_lens_kl])
        prox["kl_auc"] = np.trapezoid(kl, depth, axis=1)
    prox.loc[~valid, [c for c in prox.columns if c not in ("word_idx", "word", "status")]] = np.nan
    return prox, d_last


# ============================================================================ factorial models
def _effect_code(items: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    amb = np.where(items.ambiguity == "ambiguous", 0.5, -0.5)
    sen = np.where(items.senses == "many", 0.5, -0.5)
    return amb, sen


def _z(x):
    x = np.asarray(x, float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * 0


def _cell_boot_indices(cells: np.ndarray, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((B, len(cells)), dtype=int)
    for c in np.unique(cells):
        idx = np.flatnonzero(cells == c)
        out[:, idx] = rng.choice(idx, size=(B, len(idx)), replace=True)
    return out


def factorial_models(items: pd.DataFrame, outcomes: pd.DataFrame, controls: list[str], design: str,
                     B: int, seed: int, appendix: str) -> tuple[pd.DataFrame, dict]:
    """design 'factorial': z(y) ~ amb + sen + amb:sen + z(controls); 'ambiguity': z(y) ~ amb + z(controls).
    outcomes: (n, m) DataFrame aligned to items. Returns tidy coefficients + bootstrap draws."""
    cols = outcomes.columns.tolist()
    ok = outcomes.notna().all(axis=1).to_numpy() & items[controls].notna().all(axis=1).to_numpy()
    it, Y = items.loc[ok], outcomes.loc[ok].to_numpy(float)
    amb, sen = _effect_code(it)
    terms = ["intercept", "ambiguity"] + (["senses", "ambiguity:senses"] if design == "factorial" else [])
    X = [np.ones(len(it)), amb] + ([sen, amb * sen] if design == "factorial" else [])
    X += [_z(it[c]) for c in controls]
    X = np.column_stack(X)
    names = terms + controls
    Yz = np.column_stack([_z(Y[:, j]) for j in range(Y.shape[1])])
    cells = (it.ambiguity.fillna("NA").astype(str) + "|" + it.senses.fillna("NA").astype(str)).to_numpy()
    bidx = _cell_boot_indices(cells, B, seed)
    boots = np.full((B, len(names), Y.shape[1]), np.nan)
    for b in range(B):
        i = bidx[b]
        Yb = np.column_stack([_z(Y[i, j]) for j in range(Y.shape[1])])
        boots[b] = np.linalg.lstsq(X[i], Yb, rcond=None)[0]
    rows = []
    for j, oc in enumerate(cols):
        est, _ = su.ols_hc3(Yz[:, j], X, names)
        for t_i, t in enumerate(names):
            if t == "intercept" or t in controls:
                continue
            bb = boots[:, t_i, j]
            rows.append({"appendix": appendix, "outcome": oc, "term": t, "n": int(ok.sum()), "design": design,
                         "controls": "+".join(controls) or "none", "beta_z": est[t]["beta"], "se_hc3": est[t]["se"],
                         "p_hc3": est[t]["p"], "ci_lo_hc3": est[t]["ci_lo"], "ci_hi_hc3": est[t]["ci_hi"],
                         "ci_lo_boot": np.nanpercentile(bb, 2.5), "ci_hi_boot": np.nanpercentile(bb, 97.5),
                         "p_boot": float(su.boot_p_two_sided(bb[:, None])[0])})
    return pd.DataFrame(rows), {"names": names, "outcomes": cols, "boots": boots,
                                "words": it.word_lc.to_numpy()}


def dissociation_verdict(coefs: pd.DataFrame, outcome: str, orientation_sign: int = 1) -> dict:
    """Operational pass rule (spec §7). orientation_sign=+1 for higher=harder proxies, -1 for ease."""
    c = coefs[coefs.outcome == outcome].set_index("term")
    if not {"ambiguity", "senses"} <= set(c.index):
        return {"verdict": "not estimable"}
    exp_amb, exp_sen = +1 * orientation_sign, -1 * orientation_sign
    sig = lambda t, s: (c.loc[t, "ci_lo_boot"] > 0) if s > 0 else (c.loc[t, "ci_hi_boot"] < 0)
    wrong = lambda t, s: (c.loc[t, "ci_hi_boot"] < 0) if s > 0 else (c.loc[t, "ci_lo_boot"] > 0)
    a_ok, s_ok = sig("ambiguity", exp_amb), sig("senses", exp_sen)
    a_wrong, s_wrong = wrong("ambiguity", exp_amb), wrong("senses", exp_sen)
    b_a, b_s = c.loc["ambiguity", "beta_z"], c.loc["senses", "beta_z"]
    if a_ok and s_ok:
        v = "dissociation recovered (both effects in the predicted opposite directions, CIs exclude 0)"
    elif (a_ok or s_ok) and not (a_wrong or s_wrong):
        v = "partial (one factor in the predicted direction; the other not distinguishable from 0)"
    elif np.sign(b_a) == exp_amb and np.sign(b_s) == exp_sen and not (a_wrong or s_wrong):
        v = "directionally consistent but not significant"
    elif np.sign(b_a) == np.sign(b_s):
        v = "not recovered (both factors move in the same direction)"
    else:
        v = "not recovered (effect(s) in the non-predicted direction)"
    return {"verdict": v, "beta_ambiguity": b_a, "beta_senses": b_s, "ambiguity_ok": bool(a_ok),
            "senses_ok": bool(s_ok)}


# ============================================================================ CCGP
def ccgp(acts: np.ndarray, pos: np.ndarray, items: pd.DataFrame, factor: str, other: str, P: int,
         lambdas: np.ndarray, seed: int) -> pd.DataFrame:
    """Cross-condition generalisation of `factor` across the two levels of `other` (per layer)."""
    f = items[factor].to_numpy()
    o = items[other].to_numpy()
    lev_f = sorted(np.unique(f))
    yf = np.where(f == lev_f[0], 1.0, -1.0)
    rng = np.random.default_rng(seed)
    rows = []
    levels = sorted(np.unique(o))
    perms = {lv: np.column_stack([rng.permutation(np.flatnonzero(o == lv)) for _ in range(P)]) for lv in levels}
    for layer in range(acts.shape[1]):
        X = acts[pos, layer].astype(np.float64)
        obs, null = [], []
        for tr_lv in levels:
            tr = o == tr_lv
            te = ~tr
            mu, sd = su.standardise_fit(X[tr])
            Xs = (X - mu) / sd
            K = Xs @ Xs.T
            tr_idx = np.flatnonzero(tr)
            Yp = yf[perms[tr_lv]]                                   # permuted labels within condition
            Y = np.column_stack([yf[tr_idx], Yp])
            one = np.ones((len(tr_idx), 1))
            pred, *_ = su.fwl_kernel_ridge(K[np.ix_(tr, tr)], K[np.ix_(te, tr)], one, np.ones((te.sum(), 1)),
                                           Y, lambdas)
            y01 = (yf[te] > 0).astype(int)
            au = su.auroc(y01, pred.T)                               # (1+P,)
            obs.append(au[0])
            null.append(au[1:])
        o_m, n_m = np.mean(obs), np.mean(null, axis=0)
        rows.append({"factor": factor, "across": other, "layer": layer, "ccgp_auroc": o_m,
                     "ccgp_dir1": obs[0], "ccgp_dir2": obs[1], "null_mean": float(np.mean(n_m)),
                     "null_q95": float(np.quantile(n_m, 0.95)), "p_perm": float((1 + np.sum(n_m >= o_m)) / (1 + P))})
    df = pd.DataFrame(rows)
    df["q_bh_layers"] = su.bh_fdr(df.p_perm.to_numpy())
    return df


def factorial_selectivity(pred_dir: Path, stem_prefix: str, es: EvalSet) -> pd.DataFrame:
    """Out-of-fold probe scores (ambiguity probe, senses probe; no controls) on both factors."""
    rows = []
    for probe in ("ambiguity_bin", "senses_bin"):
        f = pred_dir / f"{stem_prefix}__{es.name}__{probe}__none.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        words = z["words"]
        it = es.items.set_index("word_lc").loc[words].reset_index()
        amb, sen = _effect_code(it)
        X = np.column_stack([np.ones(len(it)), amb, sen, amb * sen])
        pred = z["pred"].astype(np.float64).mean(0)                    # (L, n) repeat-averaged OOF scores
        for layer in range(pred.shape[0]):
            est, _ = su.ols_hc3(_z(pred[layer]), X, ["intercept", "ambiguity", "senses", "ambiguity:senses"])
            for t in ("ambiguity", "senses", "ambiguity:senses"):
                rows.append({"probe": probe, "layer": layer, "term": t, "n": len(it), **{k: v for k, v in est[t].items()}})
    df = pd.DataFrame(rows)
    if len(df):
        df["q_bh_layers"] = df.groupby(["probe", "term"]).p.transform(lambda p: su.bh_fdr(p.to_numpy()))
    return df


# ============================================================================ driver
def run_stage1(model_meta: dict, prox: pd.DataFrame, d_last: np.ndarray, acts_by_pool: dict, word_meta: pd.DataFrame,
               eval_sets: dict[str, EvalSet], cfg: dict, pred_dir: Path, out_dir: Path, ease: pd.DataFrame | None):
    mid = model_meta["model_id"]
    rc, pc = cfg["rodd"], cfg["probes"]
    lambdas = np.asarray(pc["ridge_lambdas_rel"], float)
    pos_map = {int(w): k for k, w in enumerate(word_meta.word_idx)}
    L1 = d_last.shape[1]
    depth = np.arange(L1) / (L1 - 1)
    coef_rows, verdicts, tc_rows, ccgp_rows, sel_rows, boots_store = [], [], [], [], [], {}
    for app in ("B", "C"):
        es = eval_sets[f"pub_{app}"]
        it = es.items
        p = it[["word_idx"]].merge(prox, on="word_idx", how="left")
        scalar = p[[c for c in SCALAR_PROXIES if c in p]].copy()
        if ease is not None:
            scalar["ease_rating"] = it[["word_idx"]].merge(ease, on="word_idx", how="left")["ease_rating"].to_numpy()
        seed = (cfg["seed"] + zlib.crc32(f"rodd{app}".encode())) % 2 ** 31
        n_ok = int(scalar.notna().all(axis=1).sum())
        if n_ok < 12 or it.groupby(["ambiguity", "senses"]).size().min() < 3:
            LOG.warning("%s Appendix %s: too few items for the factorial model (n=%d)", mid, app, n_ok)
            continue
        for ctrl_name, ctrl in (("primary", pc["primary_controls"]), ("none", [])):
            co, bt = factorial_models(it, scalar, ctrl, "factorial", rc["n_bootstrap"], seed, app)
            co["control_set"] = ctrl_name
            co["q_bh_proxies"] = co.groupby("term").p_boot.transform(lambda x: su.bh_fdr(x.to_numpy()))
            coef_rows.append(co)
            boots_store[f"{app}_{ctrl_name}_scalar"] = bt
            for oc in scalar.columns:
                v = dissociation_verdict(co, oc, -1 if oc == "ease_rating" else 1)
                verdicts.append({"appendix": app, "control_set": ctrl_name, "outcome": oc,
                                 "primary": oc == cfg["proxies"]["primary"], **v})
        # time course on d_l (layers 0..L-1; d_L == 0 by construction)
        dl = pd.DataFrame(d_last[[pos_map[int(w)] for w in it.word_idx], :L1 - 1],
                          columns=[f"d_{l}" for l in range(L1 - 1)])
        co, bt = factorial_models(it, dl, pc["primary_controls"], "factorial", rc["n_bootstrap"], seed, app)
        co["layer"] = co.outcome.str[2:].astype(int)
        co["rel_depth"] = depth[co.layer]
        co["q_bh_layers"] = co.groupby("term").p_boot.transform(lambda x: su.bh_fdr(x.to_numpy()))
        tc_rows.append(co)
        boots_store[f"{app}_timecourse"] = bt
        # Armstrong & Plaut (2016) ordering test: effect-weighted depth(ambiguity) - depth(senses)
        names = bt["names"]
        ba = bt["boots"][:, names.index("ambiguity"), :]               # (B, L)
        bs = -bt["boots"][:, names.index("senses"), :]                 # facilitation as positive
        dd = depth[:L1 - 1]
        wd = lambda e: np.sum(dd * np.clip(e, 0, None), -1) / np.clip(np.sum(np.clip(e, 0, None), -1), 1e-12, None)
        diff = wd(ba) - wd(bs)
        pa = co[co.term == "ambiguity"].sort_values("layer").beta_z.to_numpy()
        ps = -co[co.term == "senses"].sort_values("layer").beta_z.to_numpy()
        verdicts.append({"appendix": app, "control_set": "primary", "outcome": "timecourse_order",
                         "primary": False, "verdict": "exploratory: depth(ambiguity effect) - depth(senses effect)",
                         "estimate": float(wd(pa[None])[0] - wd(ps[None])[0]),
                         "ci_lo": float(np.nanpercentile(diff, 2.5)), "ci_hi": float(np.nanpercentile(diff, 97.5))})
        # CCGP + factorial selectivity, per pooling
        for pooling, acts in acts_by_pool.items():
            valid_it = it[it.word_idx.isin(word_meta.loc[word_meta.status == "ok", "word_idx"])].reset_index(drop=True)
            pos = np.array([pos_map[int(w)] for w in valid_it.word_idx])
            for factor, other in (("ambiguity", "senses"), ("senses", "ambiguity")):
                c = ccgp(acts, pos, valid_it, factor, other, rc["n_permutations"], lambdas,
                         (seed + zlib.crc32(factor.encode())) % 2 ** 31)
                c["appendix"], c["pooling"], c["rel_depth"] = app, pooling, depth[c.layer]
                ccgp_rows.append(c)
            s = factorial_selectivity(pred_dir, f"{_slug(mid)}__{pooling}", es)
            if len(s):
                s["appendix"], s["pooling"], s["rel_depth"] = app, pooling, depth[s.layer]
                sel_rows.append(s)
    # Stage 1A (Appendix A): ambiguity only -- confounded with senses, NOT the dissociation test
    it = eval_sets["pub_A"].items
    p = it[["word_idx"]].merge(prox, on="word_idx", how="left")
    scalar = p[[c for c in SCALAR_PROXIES if c in p]]
    if scalar.notna().all(axis=1).sum() >= 12:
        co, _ = factorial_models(it, scalar, pc["primary_controls"], "ambiguity", rc["n_bootstrap"],
                                 cfg["seed"], "A")
        co["control_set"] = "primary"
        co["q_bh_proxies"] = su.bh_fdr(co.p_boot.to_numpy())
        coef_rows.append(co)
    tag = lambda df: df.assign(model_id=mid, family=model_meta["family"], variant=model_meta["variant"],
                               size=model_meta["size"])
    res = {"coefs": tag(pd.concat(coef_rows, ignore_index=True)) if coef_rows else pd.DataFrame(),
           "verdicts": tag(pd.DataFrame(verdicts)),
           "timecourse": tag(pd.concat(tc_rows, ignore_index=True)) if tc_rows else pd.DataFrame(),
           "ccgp": tag(pd.concat(ccgp_rows, ignore_index=True)) if ccgp_rows else pd.DataFrame(),
           "selectivity": tag(pd.concat(sel_rows, ignore_index=True)) if sel_rows else pd.DataFrame()}
    for k, v in boots_store.items():
        np.savez_compressed(out_dir / f"rodd_boot__{_slug(mid)}__{k}.npz", names=np.array(v["names"]),
                            outcomes=np.array(v["outcomes"]), boots=v["boots"], words=v["words"])
    return res


def _slug(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "__", s)
