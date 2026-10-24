"""Base vs instruct comparisons within matched pairs only (spec §7, RQ5).

Primary comparison = predictive-performance curves, not raw activations. Both members of a pair are
evaluated on the same items with the same folds, so every bootstrap resample draws the SAME words
for both models (paired word-level bootstrap). The paired difference is reported per layer (pairs
share architecture, hence identical layer counts) and as a shift in the decodability centre of mass.
Instruct inputs use the native chat template, so differences are "under each variant's native input
format" and not a prompt-identical causal effect of instruction tuning.

Exploratory: linear CKA (Kornblith et al., ICML 2019) between base and instruct target-word states
at each layer.
"""
from __future__ import annotations

import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats_utils as su
from .common import slug

LOG = logging.getLogger("lexamb")


def _metric(kind, y, pred):
    if kind == "reg":
        return su.r2(y, pred)
    return su.auroc((y > 0).astype(int), pred)


def paired_curves(base_id: str, inst_id: str, metrics: pd.DataFrame, pred_dir: Path, cfg: dict) -> pd.DataFrame:
    B = cfg["probes"]["n_bootstrap"]
    jobs = metrics[(metrics.model_id == base_id) & (metrics.status == "ok")][
        ["pooling", "eval_set", "pool", "target", "kind", "control_set"]].drop_duplicates()
    rows = []
    for _, j in jobs.iterrows():
        stem = f"{j.pooling}__{j.eval_set}__{j.target}__{j.control_set}.npz"
        fb, fi = pred_dir / f"{slug(base_id)}__{stem}", pred_dir / f"{slug(inst_id)}__{stem}"
        if not (fb.exists() and fi.exists()):
            continue
        zb, zi = np.load(fb, allow_pickle=True), np.load(fi, allow_pickle=True)
        common = np.intersect1d(zb["words"], zi["words"])
        ib = np.searchsorted(zb["words"], common, sorter=np.argsort(zb["words"]))
        ib = np.argsort(zb["words"])[ib]
        ii = np.argsort(zi["words"])[np.searchsorted(zi["words"], common, sorter=np.argsort(zi["words"]))]
        y = zb["y"][ib]
        pb, pi = zb["pred"][:, :, ib].astype(np.float64), zi["pred"][:, :, ii].astype(np.float64)
        cb, ci = zb["cov"][:, ib].astype(np.float64), zi["cov"][:, ii].astype(np.float64)
        n, L = len(y), pb.shape[1]
        depth = np.arange(L) / (L - 1)
        m_b, m_i = np.nanmean(_metric(j.kind, y, pb), 0), np.nanmean(_metric(j.kind, y, pi), 0)
        d_b = m_b - np.nanmean(_metric(j.kind, y, cb))
        d_i = m_i - np.nanmean(_metric(j.kind, y, ci))
        rng = np.random.default_rng((cfg["seed"] + zlib.crc32(f"pair{stem}".encode())) % 2 ** 31)
        bd, bdd, bcog = [], [], []
        shift = 0.5 if j.kind == "bin" else 0.0
        for start in range(0, B, 100):
            idx = rng.integers(0, n, size=(min(100, B - start), n))
            yb = y[idx]
            ya = yb[None, None]
            mb = np.nanmean(_metric(j.kind, ya, pb[:, :, idx]), 0)       # (L, b)
            mi = np.nanmean(_metric(j.kind, ya, pi[:, :, idx]), 0)
            cbm = np.nanmean(_metric(j.kind, yb[None], cb[:, idx]), 0)  # (b,)
            cim = np.nanmean(_metric(j.kind, yb[None], ci[:, idx]), 0)
            bd.append(mi - mb)
            bdd.append((mi - cim) - (mb - cbm))
            bcog += [su.centre_of_mass_depth(depth, mi[:, k] - shift) - su.centre_of_mass_depth(depth, mb[:, k] - shift)
                     for k in range(mb.shape[1])]
        bd, bdd, bcog = np.concatenate(bd, 1).T, np.concatenate(bdd, 1).T, np.array(bcog)
        p = su.boot_p_two_sided(bd)
        q = su.bh_fdr(p)
        pdd = su.boot_p_two_sided(bdd)
        cog_b = su.centre_of_mass_depth(depth, m_b - shift)
        cog_i = su.centre_of_mass_depth(depth, m_i - shift)
        for l in range(L):
            rows.append({"base_id": base_id, "instruct_id": inst_id, **j.to_dict(), "layer": l, "rel_depth": depth[l],
                         "n": n, "metric": "r2_lin" if j.kind == "reg" else "auroc",
                         "base": m_b[l], "instruct": m_i[l], "diff": m_i[l] - m_b[l],
                         "diff_ci_lo": np.nanpercentile(bd[:, l], 2.5), "diff_ci_hi": np.nanpercentile(bd[:, l], 97.5),
                         "p_boot": p[l], "q_bh_layers": q[l],
                         "base_increment": d_b[l], "instruct_increment": d_i[l], "diff_increment": d_i[l] - d_b[l],
                         "diff_increment_ci_lo": np.nanpercentile(bdd[:, l], 2.5),
                         "diff_increment_ci_hi": np.nanpercentile(bdd[:, l], 97.5), "p_boot_increment": pdd[l],
                         "base_peak_depth": depth[int(np.nanargmax(m_b))], "instruct_peak_depth": depth[int(np.nanargmax(m_i))],
                         "cog_base": cog_b, "cog_instruct": cog_i, "cog_shift": cog_i - cog_b,
                         "cog_shift_ci_lo": np.nanpercentile(bcog, 2.5), "cog_shift_ci_hi": np.nanpercentile(bcog, 97.5)})
    df = pd.DataFrame(rows)
    if len(df):
        df["q_bh_increment_layers"] = df.groupby(["pooling", "eval_set", "target", "control_set"]).p_boot_increment.transform(
            lambda x: su.bh_fdr(x.to_numpy()))
    return df


def cka_by_layer(acts_b: np.ndarray, acts_i: np.ndarray, valid: np.ndarray, base_id: str, inst_id: str,
                 pooling: str) -> pd.DataFrame:
    L = acts_b.shape[1]
    rows = [{"base_id": base_id, "instruct_id": inst_id, "pooling": pooling, "layer": l, "rel_depth": l / (L - 1),
             "n": int(valid.sum()),
             "linear_cka": su.linear_cka(acts_b[valid, l].astype(np.float64), acts_i[valid, l].astype(np.float64))}
            for l in range(L)]
    return pd.DataFrame(rows)


def paired_rodd(base_id: str, inst_id: str, boot_dir: Path, rodd_coefs: pd.DataFrame, primary: str) -> pd.DataFrame:
    """Instruct - base difference of the factorial coefficients of the primary proxy (paired bootstrap:
    identical cell-stratified resamples, valid only when both models kept the same words)."""
    rows = []
    for app in ("B", "C"):
        fb = boot_dir / f"rodd_boot__{slug(base_id)}__{app}_primary_scalar.npz"
        fi = boot_dir / f"rodd_boot__{slug(inst_id)}__{app}_primary_scalar.npz"
        if not (fb.exists() and fi.exists()):
            continue
        zb, zi = np.load(fb, allow_pickle=True), np.load(fi, allow_pickle=True)
        if not np.array_equal(zb["words"], zi["words"]):
            LOG.warning("paired Rodd comparison skipped for %s/%s %s: item sets differ", base_id, inst_id, app)
            continue
        names, outs = list(zb["names"]), list(zb["outcomes"])
        if primary not in outs:
            continue
        o = outs.index(primary)
        for t in ("ambiguity", "senses", "ambiguity:senses"):
            k = names.index(t)
            d = zi["boots"][:, k, o] - zb["boots"][:, k, o]
            cb = rodd_coefs[(rodd_coefs.model_id == base_id) & (rodd_coefs.appendix == app) &
                            (rodd_coefs.outcome == primary) & (rodd_coefs.term == t) & (rodd_coefs.control_set == "primary")]
            ci = rodd_coefs[(rodd_coefs.model_id == inst_id) & (rodd_coefs.appendix == app) &
                            (rodd_coefs.outcome == primary) & (rodd_coefs.term == t) & (rodd_coefs.control_set == "primary")]
            if cb.empty or ci.empty:
                continue
            rows.append({"base_id": base_id, "instruct_id": inst_id, "appendix": app, "outcome": primary, "term": t,
                         "beta_base": float(cb.beta_z.iloc[0]), "beta_instruct": float(ci.beta_z.iloc[0]),
                         "diff": float(ci.beta_z.iloc[0] - cb.beta_z.iloc[0]),
                         "diff_ci_lo": np.nanpercentile(d, 2.5), "diff_ci_hi": np.nanpercentile(d, 97.5),
                         "p_boot": float(su.boot_p_two_sided(d[:, None])[0])})
    return pd.DataFrame(rows)
