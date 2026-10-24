"""Layer-wise grouped-CV probes with conditional (covariate-partialled) increments (spec §6 A/B, §7).

For every model x pooling x analysis set x target x control set and every layer:
  * repeated StratifiedGroupKFold by word (identical folds for every model -> paired comparisons);
  * all scaling, covariate regression and alpha selection inside training folds only;
  * out-of-fold predictions of (i) covariates-only baseline and (ii) covariates + hidden state;
  * metrics averaged over repeats; word-level bootstrap CIs; Freedman-Lane permutation null
    (Freedman & Lane 1983; Ojala & Garriga 2010) shared by all layers -> per-layer p, BH-FDR across
    layers and Westfall-Young max-T FWER across layers.
"""
from __future__ import annotations

import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import stats_utils as su
from .audit_data import EvalSet

LOG = logging.getLogger("lexamb")

TARGETS = {
    # target column: (kind, partner controls added to the control set)
    "log_subtlex": ("reg", ["log1p_wn_n_all"]),       # frequency beyond sense count
    "log1p_wn_n_all": ("reg", []),                    # sense count beyond frequency/form
    "wn_n_all": ("reg", []),
    "log1p_wn_n_noun": ("reg", []),
    "gcide_nonzero": ("reg", ["log1p_wn_n_all"]),     # meanings proxy beyond sense count
    "gcide_recode": ("reg", ["log1p_wn_n_all"]),
    "gcide_multi": ("bin", ["log1p_wn_n_all"]),
    "sim_top2": ("reg", ["log1p_wn_n_all"]),          # semantic distance beyond sense count (RQ4)
    "min_sim": ("reg", ["log1p_wn_n_all"]),
    "mean_sim": ("reg", ["log1p_wn_n_all"]),
    "ambiguity_bin": ("bin", ["log1p_wn_n_all"]),
    "senses_bin": ("bin", []),
    "pool_condition_bin": ("bin", []),                # rule-derived label: exploratory only
}
SOURCE = {"log_subtlex": "log_subtlex", "log1p_wn_n_all": "log1p_wn_n_all", "wn_n_all": "log1p_wn_n_all",
          "log1p_wn_n_noun": "log1p_wn_n_all"}


def controls_for(target: str, control_set: str, cfg: dict) -> list[str]:
    if control_set == "none":
        return []
    base = {"primary": cfg["probes"]["primary_controls"], "primary_tok": cfg["probes"]["primary_controls"],
            "expanded": cfg["probes"]["expanded_controls"]}[control_set]
    partner = TARGETS[target][1]
    src = SOURCE.get(target, target)
    ctrl = [c for c in base if c != src] + [p for p in partner if p != src]
    if control_set == "primary_tok":
        ctrl = ctrl + ["log_n_subtokens"]
    return list(dict.fromkeys(ctrl))


def build_jobs(cfg: dict, eval_sets: dict[str, EvalSet]) -> list[dict]:
    jobs = []
    for es_name, plan in cfg["probes"]["plan"].items():
        if es_name not in eval_sets:
            continue
        for target, csets in plan.items():
            for cs in csets:
                jobs.append({"eval_set": es_name, "target": target, "kind": TARGETS[target][0],
                             "control_set": cs, "controls": controls_for(target, cs, cfg)})
    return jobs


def make_folds(es: EvalSet, n_folds: int, n_repeats: int, seed: int) -> np.ndarray:
    """(R, n) fold ids; identical across models because they depend only on the item set and seed."""
    n = len(es.items)
    k = max(2, min(n_folds, n // 2))
    folds = np.zeros((n_repeats, n), dtype=int)
    groups = es.items["word_idx"].to_numpy()
    for r in range(n_repeats):
        _, counts = np.unique(es.strat, return_counts=True)
        strat = es.strat if counts.min() >= k else np.zeros(n, dtype=int)
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed + 1000 * r)
        for f, (_, te) in enumerate(cv.split(np.zeros(n), strat, groups)):
            folds[r, te] = f
    return folds


def _job_seed(cfg_seed: int, *parts: str) -> int:
    return (cfg_seed + zlib.crc32("|".join(parts).encode())) % (2 ** 31)


def _design(items: pd.DataFrame, controls: list[str], rows: np.ndarray, tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Intercept + controls standardised on training rows; constant columns dropped."""
    if not controls:
        one = np.ones((len(rows), 1))
        return one[tr], one[~tr]
    Z = items.loc[rows, controls].to_numpy(dtype=float)
    mu, sd = Z[tr].mean(0), Z[tr].std(0)
    keep = sd > 1e-10
    Z = (Z[:, keep] - mu[keep]) / sd[keep]
    C = np.column_stack([np.ones(len(rows)), Z])
    return C[tr], C[~tr]


def run_probes(acts: np.ndarray, word_meta: pd.DataFrame, model_meta: dict, eval_sets: dict[str, EvalSet],
               jobs: list[dict], cfg: dict, pooling: str, pred_dir: Path) -> pd.DataFrame:
    pc = cfg["probes"]
    lambdas = np.asarray(pc["ridge_lambdas_rel"], dtype=float)
    R, P, B = pc["n_repeats"], pc["n_permutations"], pc["n_bootstrap"]
    n_layers = acts.shape[1]
    depth = np.arange(n_layers) / (n_layers - 1)
    pos = {int(w): k for k, w in enumerate(word_meta.word_idx)}
    ok_words = set(word_meta.loc[word_meta.status == "ok", "word_idx"].astype(int))
    nsub = dict(zip(word_meta.word_idx.astype(int), word_meta.get("n_subtokens", pd.Series(dtype=float))))
    model_id = model_meta["model_id"]
    all_rows = []

    for es_name in dict.fromkeys(j["eval_set"] for j in jobs):
        es = eval_sets[es_name]
        items = es.items.copy()
        items["log_n_subtokens"] = np.log([nsub.get(int(w), np.nan) for w in items.word_idx])
        valid = items.word_idx.isin(ok_words).to_numpy()
        es_jobs = [j for j in jobs if j["eval_set"] == es_name]
        folds = make_folds(es, pc["n_folds"], R, cfg["seed"])
        # ---- per-job row masks, targets and Freedman-Lane permuted targets --------------------
        prepared = []
        for j in es_jobs:
            cols = [j["target"]] + j["controls"]
            mask = valid & items[cols].notna().all(axis=1).to_numpy()
            y = items.loc[mask, j["target"]].to_numpy(dtype=float)
            if j["kind"] == "bin":
                y = np.where(y > 0.5, 1.0, -1.0)
            n = int(mask.sum())
            skip = None
            if n < pc["min_n"]:
                skip = f"n={n} < min_n={pc['min_n']}"
            elif j["kind"] == "bin" and min((y > 0).sum(), (y < 0).sum()) < folds.max() + 1:
                skip = "a class has fewer items than folds"
            elif np.std(y) == 0:
                skip = "constant target"
            if skip:
                LOG.warning("%s/%s skip %s|%s|%s: %s", model_id, pooling, es_name, j["target"], j["control_set"], skip)
                all_rows.append({**_ids(model_meta, pooling, es, j), "layer": np.nan, "n": n, "status": f"skipped: {skip}"})
                continue
            rng = np.random.default_rng(_job_seed(cfg["seed"], es_name, j["target"], j["control_set"]))
            rows_idx = np.flatnonzero(mask)
            Cg = np.column_stack([np.ones(n)] + ([items.loc[mask, j["controls"]].to_numpy(dtype=float)]
                                                 if j["controls"] else []))
            fitted = Cg @ np.linalg.lstsq(Cg, y, rcond=None)[0]
            resid = y - fitted
            ystar = np.column_stack([fitted + resid[rng.permutation(n)] for _ in range(P)]) if P else np.zeros((n, 0))
            prepared.append({**j, "mask": mask, "rows": rows_idx, "y": y, "ystar": ystar, "n": n,
                             "pred": np.full((R, n_layers, n), np.nan, np.float32),
                             "prob": np.full((R, n_layers, n), np.nan, np.float32) if j["kind"] == "bin" else None,
                             "cov": np.full((R, n), np.nan, np.float32),
                             "perm_pred": np.full((n_layers, n, P), np.nan, np.float32),
                             "perm_cov": np.full((n, P), np.nan, np.float32)})
        if not prepared:
            continue
        groups: dict = {}
        for jp in prepared:
            groups.setdefault((jp["mask"].tobytes(), tuple(jp["controls"])), []).append(jp)
        es_rows = np.flatnonzero(valid)
        for layer in range(n_layers):
            X = acts[[pos[int(w)] for w in items.word_idx.iloc[es_rows]], layer].astype(np.float64)
            for r in range(R):
                for f in range(folds[r].max() + 1):
                    tr_all = folds[r, es_rows] != f
                    mu, sd = su.standardise_fit(X[tr_all])
                    Xs = (X - mu) / sd
                    Kfull = Xs @ Xs.T
                    loc = {int(i): k for k, i in enumerate(es_rows)}
                    for (_, controls), gj in groups.items():
                        rows = gj[0]["rows"]
                        kk = np.array([loc[int(i)] for i in rows])
                        tr = folds[r, rows] != f
                        if tr.all() or not tr.any():
                            continue
                        K = Kfull[np.ix_(kk, kk)]
                        C_T, C_E = _design(items, list(controls), rows, tr)
                        Ycols = [jp["y"][:, None] for jp in gj]
                        if r == 0:
                            Ycols += [jp["ystar"] for jp in gj]
                        Y = np.column_stack(Ycols)
                        pred_E, loo_T, cov_E, _ = su.fwl_kernel_ridge(K[np.ix_(tr, tr)], K[np.ix_(~tr, tr)],
                                                                     C_T, C_E, Y[tr], lambdas)
                        te = np.flatnonzero(~tr)
                        for m, jp in enumerate(gj):
                            jp["pred"][r, layer, te] = pred_E[:, m]
                            jp["cov"][r, te] = cov_E[:, m]
                            if jp["kind"] == "bin":
                                jp["prob"][r, layer, te] = su.platt(loo_T[:, m], (jp["y"][tr] > 0).astype(int),
                                                                    pred_E[:, m])
                        if r == 0 and P:
                            off = len(gj)
                            for m, jp in enumerate(gj):
                                sl = slice(off + m * P, off + (m + 1) * P)
                                jp["perm_pred"][layer, te] = pred_E[:, sl]
                                jp["perm_cov"][te] = cov_E[:, sl]
            if layer % 8 == 0 or layer == n_layers - 1:
                LOG.info("%s/%s %s layer %d/%d done", model_id, pooling, es_name, layer, n_layers - 1)
        for jp in prepared:
            all_rows += _summarise(jp, es, model_meta, pooling, depth, B, cfg, pred_dir)
    return pd.DataFrame(all_rows)


def _ids(model_meta, pooling, es, j):
    return {"model_id": model_meta["model_id"], "family": model_meta["family"], "variant": model_meta["variant"],
            "size": model_meta["size"], "paired_base": model_meta.get("paired_base"), "pooling": pooling,
            "eval_set": es.name, "pool": es.pool, "target": j["target"], "kind": j["kind"],
            "control_set": j["control_set"], "controls": "+".join(j["controls"]) or "intercept"}


def _summarise(jp, es, model_meta, pooling, depth, B, cfg, pred_dir: Path) -> list[dict]:
    y, pred, cov, n = jp["y"], jp["pred"].astype(np.float64), jp["cov"].astype(np.float64), jp["n"]
    R, L = pred.shape[0], pred.shape[1]
    words = es.items.word_lc.to_numpy()[jp["rows"]]
    stem = f"{_slug(model_meta['model_id'])}__{pooling}__{es.name}__{jp['target']}__{jp['control_set']}"
    np.savez_compressed(pred_dir / f"{stem}.npz", y=y, pred=jp["pred"], cov=jp["cov"], words=words,
                        prob=jp["prob"] if jp["prob"] is not None else np.zeros(0))
    # ---- point metrics (mean over repeats)
    r2_full = su.r2(y, pred)                                      # (R, L)
    r2_cov = su.r2(y, cov)                                        # (R,)
    out = {"r2_lin": r2_full.mean(0), "delta_r2_lin": (r2_full - r2_cov[:, None]).mean(0),
           "r2_lin_cov": np.repeat(r2_cov.mean(), L), "r2_lin_sd_repeats": r2_full.std(0)}
    if jp["kind"] == "reg":
        out["mae"] = su.mae(y, pred).mean(0)
        out["spearman"] = np.array([[su.spearman(y, pred[r, l]) for l in range(L)] for r in range(R)]).mean(0)
        primary, delta = "r2_lin", "delta_r2_lin"
    else:
        y01 = (y > 0).astype(int)
        prob = jp["prob"].astype(np.float64)
        au = su.auroc(y01, pred)
        au_cov = su.auroc(y01, cov)
        out["auroc"] = au.mean(0)
        out["auroc_cov"] = np.repeat(au_cov.mean(), L)
        out["delta_auroc"] = (au - au_cov[:, None]).mean(0)
        out["balanced_acc"] = np.array([[su.balanced_acc(y01, prob[r, l]) for l in range(L)] for r in range(R)]).mean(0)
        out["macro_f1"] = np.array([[su.macro_f1(y01, prob[r, l]) for l in range(L)] for r in range(R)]).mean(0)
        out["brier"] = np.mean((prob - y01) ** 2, axis=-1).mean(0)
        out["brier_prior"] = np.repeat(np.mean((y01.mean() - y01) ** 2), L)
        primary, delta = "auroc", "delta_auroc"
    # ---- word-level bootstrap CIs (same item resamples for every layer)
    rng = np.random.default_rng(_job_seed(cfg["seed"], "boot", es.name, jp["target"], jp["control_set"]))
    boot_p, boot_d = [], []
    for start in range(0, B, 100):
        idx = rng.integers(0, n, size=(min(100, B - start), n))
        yb = y[idx]                                                    # (b, n)
        pb = pred[:, :, idx]                                           # (R, L, b, n)
        cb = cov[:, idx]                                               # (R, b, n)
        if jp["kind"] == "reg":
            m_full = su.r2(yb[None, None], pb)                          # (R, L, b)
            m_cov = su.r2(yb[None], cb)                                 # (R, b)
        else:
            yb01 = (yb > 0).astype(int)
            m_full = su.auroc(yb01[None, None], pb)
            m_cov = su.auroc(yb01[None], cb)
        boot_p.append(np.nanmean(m_full, 0))
        boot_d.append(np.nanmean(m_full - m_cov[:, None], 0))
    boot_p = np.concatenate(boot_p, axis=1)                            # (L, B)
    boot_d = np.concatenate(boot_d, axis=1)
    # ---- permutation test of the conditional increment on repeat 0
    P = jp["ystar"].shape[1]
    obs = su.r2(y, pred[0]) - su.r2(y, cov[0])                         # (L,)
    if P:
        ys = jp["ystar"].T                                              # (P, n)
        pp = np.transpose(jp["perm_pred"], (2, 0, 1)).astype(np.float64)  # (P, L, n)
        null = su.r2(ys[:, None], pp) - su.r2(ys, jp["perm_cov"].T.astype(np.float64))[:, None]
        p_perm = su.perm_pvalue(obs, null)
        p_maxT = su.maxT_pvalue(obs, null)
    else:
        p_perm = p_maxT = np.full(L, np.nan)
    q = su.bh_fdr(p_perm)
    cog = su.centre_of_mass_depth(depth, out[primary] - (0.5 if jp["kind"] == "bin" else 0))
    exp_layer = su.expected_layer_tenney(depth, out[primary])
    rows = []
    ids = _ids(model_meta, pooling, es, jp)
    for l in range(L):
        row = {**ids, "layer": l, "rel_depth": depth[l], "n_layers_blocks": L - 1, "n": n, "status": "ok",
               "primary_metric": primary,
               f"{primary}_ci_lo": np.nanpercentile(boot_p[l], 2.5), f"{primary}_ci_hi": np.nanpercentile(boot_p[l], 97.5),
               f"{delta}_ci_lo": np.nanpercentile(boot_d[l], 2.5), f"{delta}_ci_hi": np.nanpercentile(boot_d[l], 97.5),
               "p_perm": p_perm[l], "q_bh_layers": q[l], "p_maxT_layers": p_maxT[l], "n_perm": P,
               "cog_depth": cog, "expected_layer_tenney": exp_layer}
        for k, v in out.items():
            row[k] = float(v[l])
        rows.append(row)
    return rows


def _slug(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "__", s)
