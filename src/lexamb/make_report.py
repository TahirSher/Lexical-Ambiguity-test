"""Markdown results report (spec §12). Every number is read from the analysis outputs; if an output
is missing, the section says so instead of inventing a result."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import atomic_write_text


def _md(df: pd.DataFrame, cols=None, n=40) -> str:
    if df is None or df.empty:
        return "_No results available for this section._\n"
    d = df[cols] if cols else df
    d = d.head(n)
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "|".join(["---"] * len(d.columns)) + "|\n"
    body = ""
    for _, r in d.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else f"{v:.3f}")
            else:
                cells.append(str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def _peaks(metrics, target, eval_set, control="none", col=None):
    m = metrics[(metrics.target == target) & (metrics.eval_set == eval_set) & (metrics.control_set == control)
                & (metrics.pooling == "last") & (metrics.status == "ok")]
    if m.empty:
        return pd.DataFrame()
    col = col or m.primary_metric.iloc[0]
    if control != "none":
        col = "delta_r2_lin" if col == "r2_lin" else "delta_auroc"
    out = []
    for mid, g in m.groupby("model_id"):
        g = g.sort_values("layer")
        b = g.loc[g[col].idxmax()]
        f = g.iloc[-1]
        out.append({"model": mid.split("/")[-1], "variant": b.variant, "N": int(b.n), "metric": col,
                    "embedding": g.iloc[0][col], "peak": b[col],
                    "peak 95% CI": f"[{b[f'{col}_ci_lo']:.3f}, {b[f'{col}_ci_hi']:.3f}]",
                    "peak depth": b.rel_depth, "final": f[col], "max-T p (peak)": b.p_maxT_layers,
                    "n layers q<.05": int((g.q_bh_layers < 0.05).sum()), "CoG depth": b.cog_depth})
    return pd.DataFrame(out)


def write_report(dirs: dict, audit_key: dict, metrics: pd.DataFrame, rodd: dict, stage2: pd.DataFrame,
                 pairs_df: pd.DataFrame, cka: pd.DataFrame, model_metas: list[dict], cfg: dict, dry_run: bool) -> Path:
    L = []
    L.append("# Layer-wise lexical frequency and semantic ambiguity in base and instruct LLMs — results report\n")
    if dry_run:
        L.append("> **DRY RUN.** Tiny word sample and reduced resampling: these numbers validate the pipeline's "
                 "schemas only and must not be interpreted scientifically.\n")
    have = metrics is not None and not metrics.empty
    L.append("## 1. Executive summary\n")
    if not have:
        L.append("No probe results exist yet; no conclusions are drawn.\n")
    else:
        L.append("All statements below are generated from the output tables. 'Decodable' means *linearly recoverable "
                 "from the target-word state under grouped cross-validation*; it does not show that the model uses the "
                 "feature (Belinkov 2022; Elazar et al. 2021).\n")
        for tgt, es, lab in (("log_subtlex", "der_matched", "RQ1 frequency (derived matched)"),
                             ("log_subtlex", "pub_all", "RQ1 frequency (published)"),
                             ("log1p_wn_n_all", "der_matched", "RQ2 sense count (derived matched)"),
                             ("log1p_wn_n_all", "pub_all", "RQ2 sense count (published)")):
            p = _peaks(metrics, tgt, es)
            pc = _peaks(metrics, tgt, es, "primary")
            if p.empty:
                continue
            sig = int((pc["max-T p (peak)"] < 0.05).sum()) if not pc.empty else 0
            L.append(f"- **{lab}:** peak out-of-fold R² ranged {p.peak.min():.3f}–{p.peak.max():.3f} across "
                     f"{len(p)} models (median peak depth {p['peak depth'].median():.2f}); the conditional increment "
                     f"beyond the primary controls had a max-T-corrected p < .05 at its peak layer in {sig}/{len(pc)} models.")
        v = rodd.get("verdicts")
        if v is not None and len(v):
            pv = v[v.primary.astype(bool) & (v.control_set == "primary")]
            for app in ("B", "C"):
                g = pv[pv.appendix == app]
                if len(g):
                    counts = g.verdict.value_counts().to_dict()
                    L.append(f"- **RQ6 Appendix {app}** (primary proxy `{cfg['proxies']['primary']}`): " +
                             "; ".join(f"{k}: {n}" for k, n in counts.items()) + ".")
        L.append("")
    L.append("## 2. Dataset audit and exact sample sizes\n")
    L.append(f"- Rows: {audit_key['n_rows']}; unique word forms: {audit_key['n_unique_words']}; provenance: "
             f"{audit_key['provenance']}.")
    L.append(f"- Published appendix rows: {audit_key['appendix_rows']}; 2×2 factorial rows: {audit_key['bc_factorial_rows']}.")
    L.append("- Stage 2 attrition: " + "; ".join(f"{r['filter']} → {r['n_unique_words']}" for r in audit_key["stage2_attrition"]) + ".")
    L.append("- Analysis sets (deduplicated within set; provenance never mixed): " +
             ", ".join(f"{k} N={v}" for k, v in audit_key["eval_sets"].items()) + ".\n")
    L.append("## 3. Models and prompts\n")
    L.append(_md(pd.DataFrame(model_metas), [c for c in ["model_id", "variant", "paired_base", "n_layers", "hidden_size",
                                                         "dtype", "model_commit", "final_hidden_state_is_normed"]
                                             if model_metas and c in model_metas[0]]))
    L.append("Base models receive `Word: {word}` as plain text; instruct models receive the same content inside "
             "their native chat template (`add_generation_prompt=False`, fixed `date_string` for Llama templates). "
             "States are read at the target word's last subtoken (primary) and the mean of its subtokens (sensitivity). "
             "Layer 0 is the embedding output; layers 1…L are block outputs.\n")
    for sec, tgt, es_list in (("4. Layer-wise frequency (RQ1)", "log_subtlex", ["pub_all", "der_matched", "der_full"]),
                              ("5. Layer-wise sense count (RQ2)", "log1p_wn_n_all", ["pub_all", "der_matched", "der_full"])):
        L.append(f"## {sec}\n")
        for es in es_list:
            for ctrl in ("none", "primary"):
                p = _peaks(metrics, tgt, es, ctrl) if have else pd.DataFrame()
                if p.empty:
                    continue
                L.append(f"**{es}, controls = {ctrl}** ({'R²' if ctrl == 'none' else 'ΔR² over covariates'})\n")
                L.append(_md(p))
        L.append("See `fig02_frequency_layerwise` / `fig03_sense_count_layerwise` and the conditional supplementary figures.\n")
    L.append("## 6. Rodd Stage 1 (published pool)\n")
    L.append("Appendix A confounds meanings with senses and is not the dissociation test. Appendices B and C are "
             "analysed separately in one factorial model each (effect coding, interaction included). Appendix C is "
             "the auditory stimulus set presented as text. In the human data the Appendix B ambiguity effect was "
             "small/non-significant, so a null there is not a failure.\n")
    if have:
        for es in ("pub_all", "pub_A", "pub_B", "pub_C"):
            p = _peaks(metrics, "ambiguity_bin", es)
            if len(p):
                L.append(f"**Ambiguity probe, {es}** (AUROC)\n")
                L.append(_md(p))
        for es in ("pub_B", "pub_C"):
            p = _peaks(metrics, "senses_bin", es)
            if len(p):
                L.append(f"**Senses probe, {es}** (AUROC)\n")
                L.append(_md(p))
    v = rodd.get("verdicts")
    L.append("**Dissociation rule on model-derived proxies** (higher = harder, except ease):\n")
    L.append(_md(v, [c for c in ["model_id", "appendix", "control_set", "outcome", "verdict", "beta_ambiguity",
                                 "beta_senses", "estimate", "ci_lo", "ci_hi"] if v is not None and c in v], n=200))
    c = rodd.get("ccgp")
    if c is not None and len(c):
        best = c[c.pooling == "last"].loc[lambda d: d.groupby(["model_id", "appendix", "factor"]).ccgp_auroc.idxmax()]
        L.append("**Cross-condition generalisation (CCGP) — best layer per model** (exploratory)\n")
        L.append(_md(best, ["model_id", "appendix", "factor", "layer", "rel_depth", "ccgp_auroc", "null_q95", "p_perm",
                            "q_bh_layers"], n=200))
    L.append("## 7. Stage 2 — WordNet-derived generalisation\n")
    if stage2 is not None and len(stage2):
        s = stage2[(stage2.role != "control") & (stage2.analysis_variant == "primary")]
        L.append(_md(s, ["model_id", "outcome", "predictor", "n", "beta_z", "ci_lo", "ci_hi", "p_hc3", "q_bh_proxies", "vif"], n=200))
    else:
        L.append("_No Stage 2 results._\n")
    L.append("## 8. Base vs instruct (matched pairs only)\n")
    if pairs_df is not None and len(pairs_df):
        t = pairs_df[(pairs_df.pooling == "last") & (pairs_df.control_set == "none")]
        summ = t.groupby(["base_id", "eval_set", "target"]).agg(
            n=("n", "max"), layers=("layer", "size"), layers_q05=("q_bh_layers", lambda q: int((q < 0.05).sum())),
            max_abs_diff=("diff", lambda d: float(np.nanmax(np.abs(d)))), cog_shift=("cog_shift", "first"),
            cog_lo=("cog_shift_ci_lo", "first"), cog_hi=("cog_shift_ci_hi", "first")).reset_index()
        L.append(_md(summ, n=200))
        paired = {m.get("pair") for m in cfg["models"]}
        solo = [m["id"].split("/")[-1] for m in cfg["models"] if m["variant"] == "base" and m["id"] not in paired]
        L.append(f"Base-only models (no instruct counterpart; never paired): {', '.join(solo) or 'none'}.\n")
    else:
        L.append("_No paired comparisons available._\n")
    if cka is not None and len(cka):
        L.append("Exploratory linear CKA (base vs instruct, same layer) — minimum over layers per pair:\n")
        L.append(_md(cka.groupby(["base_id", "pooling"]).linear_cka.agg(["min", "mean"]).reset_index()))
    L.append("## 9. Robustness and multiple testing\n")
    L.append("Families: BH-FDR across layers within (model, set, target, control set, pooling); Westfall–Young "
             "max-T across layers from shared permutations; BH across proxies within (model, appendix, term). "
             "Sensitivity analyses: mean-subtoken pooling, expanded controls, tokenisation control, full derived pool, "
             "GCIDE zero recoding / indicator, noun sense count, semantic distance, repeated grouped CV. See Table 8 and Fig. 7.\n")
    L.append("## 10. Limitations and reproducibility checklist\n")
    L.append("""- Probe decodability ≠ use: causal claims require interventions (e.g. amnesic probing).
- Proxies (settling cost, surprisal, entropy, ease) are model-derived; none is a human reaction time.
- Appendix C is an auditory stimulus set given to text-only models; no auditory processing is simulated.
- GCIDE etymologies are a computed meanings proxy, and WordNet candidate labels are rule-derived, not human gold labels.
- Stage 2 is generalisation on a pool that is longer, rarer and more orthographically isolated, not a replication.
- Base and instruct inputs differ by the native chat wrapper, so pair differences are not prompt-identical effects of tuning.
- Reproducibility: `run_manifest.json` (config, data hash, package versions, model commits, output hashes); seeds fixed;
  alignment audit sheets per tokenizer in `outputs/activations/*/alignment_manual_audit.csv` need human sign-off.
""")
    path = dirs["reports"] / "results_report.md"
    atomic_write_text(path, "\n".join(L))
    return path
