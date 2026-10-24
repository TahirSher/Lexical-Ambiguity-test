"""Publication tables (spec §9): CSV + Markdown + LaTeX; every table carries N and the pool."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .common import write_table

LOG = logging.getLogger("lexamb")

PREDICTORS = [
    ("Frequency", "log_subtlex", "log10(SUBTLEX-US count + 1)", "primary frequency target / control",
     "Corpus frequency, not familiarity"),
    ("Sense count", "wn_n_all", "number of WordNet 3.0 synsets, all parts of speech (probe target log1p)",
     "more senses -> facilitation (Rodd et al. 2002)", "Counts related AND unrelated senses; not a meanings count"),
    ("Noun sense count", "wn_n_noun", "WordNet noun synsets", "secondary check", "Different construct from wn_n_all"),
    ("Meanings (published)", "ambiguity", "Rodd et al. ambiguous vs unambiguous (Wordsmyth entries)",
     "ambiguous -> inhibition", "Published rows only; confounded with senses in Appendix A"),
    ("Senses (published)", "senses", "Rodd et al. few vs many senses", "many -> facilitation", "Appendix B/C only"),
    ("Meanings proxy (computed)", "gcide_etymologies", "number of GCIDE etymology fields",
     "more etymologies -> inhibition (generalisation)", "Computed, not a human norm; 0 = none recorded"),
    ("Semantic distance", "sim_top2 / min_sim / mean_sim", "WordNet similarity of noun senses",
     "lower similarity = more homonym-like", "Distance, not count; noun senses only"),
    ("Candidate class", "ambiguity_class / pool_condition", "rule-derived from WordNet counts & similarity",
     "exploratory only", "Not human-validated; the two labels share their rule"),
    ("Form controls", "length, ortho_n, n_syll, n_phon", "letters, orthographic neighbours, syllables, phonemes",
     "controls", "n_syll / n_phon partially missing"),
    ("Semantic control", "concreteness", "Brysbaert et al. (2014) ratings", "control", "2 missing"),
]


def table1(audit_key: dict, attr: pd.DataFrame, eval_sets: dict, tables_dir):
    rows = [{"item": "rows / unique word forms", "pool": "all", "n": f"{audit_key['n_rows']} / {audit_key['n_unique_words']}"}]
    rows += [{"item": f"provenance = {k}", "pool": k, "n": v} for k, v in audit_key["provenance"].items()]
    rows += [{"item": f"Appendix {k} rows", "pool": "published", "n": v} for k, v in sorted(audit_key["appendix_rows"].items())]
    rows += [{"item": f"Appendix {k} 2x2 factorial rows", "pool": "published", "n": v}
             for k, v in sorted(audit_key["bc_factorial_rows"].items())]
    rows += [{"item": f"Stage 2 filter: {r['filter']}", "pool": "derived", "n": r["n_unique_words"]} for _, r in attr.iterrows()]
    rows += [{"item": f"analysis set {k}: {v.description}", "pool": v.pool, "n": len(v.items)} for k, v in eval_sets.items()]
    write_table(pd.DataFrame(rows), tables_dir / "table1_dataset_filtering", "Table 1. Dataset and filtering summary")


def table2(model_metas: list[dict], base_only_ids: set[str], tables_dir):
    keep = ["model_id", "family", "variant", "paired_base", "size", "n_layers", "hidden_size", "vocab_size", "dtype",
            "device", "quantization", "model_commit", "tokenizer_commit", "prompt_template_hash", "final_hidden_state_is_normed"]
    df = pd.DataFrame([{k: m.get(k) for k in keep} for m in model_metas])
    if len(df):
        df["paired_base"] = [p if v == "instruct" else ("base-only (no instruct counterpart)" if mid in base_only_ids
                                                         else "base of a pair")
                             for mid, v, p in zip(df.model_id, df.variant, df.paired_base)]
    write_table(df, tables_dir / "table2_model_inventory", "Table 2. Model inventory (pairs share tokenizer and architecture)")


def table3(df: pd.DataFrame, tables_dir):
    rows = []
    for construct, col, op, direction, lim in PREDICTORS:
        first = col.split(" ")[0].split("/")[0].strip().rstrip(",")
        cov = {}
        for prov, g in df.groupby("provenance"):
            cov[prov] = f"{g[first].notna().mean() * 100:.0f}%" if first in g else "n/a"
        rows.append({"construct": construct, "column": col, "operational_definition": op, "expected_direction": direction,
                     "coverage_published": cov.get("published"), "coverage_derived": cov.get("wordnet-derived"),
                     "limitations": lim})
    write_table(pd.DataFrame(rows), tables_dir / "table3_predictor_definitions", "Table 3. Predictor definitions")


def table4(metrics: pd.DataFrame, tables_dir):
    m = metrics[(metrics.status == "ok") & (metrics.pooling == "last") & metrics.control_set.isin(["none", "primary"])]
    rows = []
    for key, g in m.groupby(["model_id", "eval_set", "pool", "target", "control_set"]):
        metric = g.primary_metric.iloc[0]
        col = metric if key[4] == "none" else ("delta_r2_lin" if metric == "r2_lin" else "delta_auroc")
        lo, hi = f"{col}_ci_lo", f"{col}_ci_hi"
        g = g.sort_values("layer")
        b = g.loc[g[col].idxmax()]
        e, f = g.iloc[0], g.iloc[-1]
        rows.append({"model": key[0].split("/")[-1], "pool": key[2], "eval_set": key[1], "target": key[3],
                     "controls": key[4], "metric": col, "n": int(b.n),
                     "embedding": e[col], "best_layer": int(b.layer), "best_rel_depth": b.rel_depth,
                     "best": b[col], "best_ci": f"[{b[lo]:.3f}, {b[hi]:.3f}]", "final": f[col],
                     "final_ci": f"[{f[lo]:.3f}, {f[hi]:.3f}]", "p_maxT_best": b.p_maxT_layers,
                     "q_bh_best": b.q_bh_layers, "cog_depth": b.cog_depth})
    df = pd.DataFrame(rows)
    write_table(df, tables_dir / "table4_layerwise_performance",
                "Table 4. Best and final layer probe performance (best layer selected post hoc; final layer also shown)")
    return df


def table5(coefs: pd.DataFrame, verdicts: pd.DataFrame, tables_dir):
    if coefs is None or coefs.empty:
        return
    c = coefs.copy()
    c["model"] = c.model_id.str.split("/").str[-1]
    cols = ["model", "appendix", "design", "control_set", "outcome", "term", "n", "beta_z", "ci_lo_boot", "ci_hi_boot",
            "p_boot", "q_bh_proxies", "se_hc3", "p_hc3"]
    write_table(c[cols], tables_dir / "table5_rodd_factorial",
                "Table 5. Rodd Stage 1 effects on model-derived proxies (published pool; z-scored outcome; "
                "effect coding; Appendix C = auditory stimulus set given as text)")
    if verdicts is not None and len(verdicts):
        v = verdicts.copy()
        v["model"] = v.model_id.str.split("/").str[-1]
        write_table(v.drop(columns=["model_id"]), tables_dir / "table5b_dissociation_verdicts",
                    "Table 5b. Operational dissociation rule per model, appendix and proxy")


def table6(stage2: pd.DataFrame, tables_dir):
    if stage2 is None or stage2.empty:
        return
    s = stage2[stage2.role != "control"].copy()
    s["model"] = s.model_id.str.split("/").str[-1]
    cols = ["model", "analysis_variant", "eval_set", "outcome", "predictor", "role", "n", "beta_z", "ci_lo", "ci_hi",
            "p_hc3", "q_bh_proxies", "vif", "note"]
    write_table(s[cols], tables_dir / "table6_stage2_generalisation",
                "Table 6. Stage 2 (WordNet-derived pool; generalisation, not replication): sense count and GCIDE "
                "meanings proxy after controls")


def table7(pairs_df: pd.DataFrame, tables_dir):
    if pairs_df is None or pairs_df.empty:
        return
    rows = []
    for key, g in pairs_df[pairs_df.pooling == "last"].groupby(["base_id", "instruct_id", "eval_set", "target", "control_set"]):
        g = g.sort_values("layer")
        f = g.iloc[-1]
        rows.append({"pair": key[0].split("/")[-1], "eval_set": key[2], "target": key[3], "controls": key[4],
                     "n": int(f.n), "metric": f.metric, "diff_final": f["diff"],
                     "diff_final_ci": f"[{f.diff_ci_lo:.3f}, {f.diff_ci_hi:.3f}]",
                     "max_abs_diff": g["diff"].abs().max(), "n_layers_q<.05": int((g.q_bh_layers < 0.05).sum()),
                     "n_layers": len(g), "peak_depth_base": f.base_peak_depth, "peak_depth_instruct": f.instruct_peak_depth,
                     "cog_shift": f.cog_shift, "cog_shift_ci": f"[{f.cog_shift_ci_lo:.3f}, {f.cog_shift_ci_hi:.3f}]",
                     "limitation": "native templates differ (chat wrapper vs plain text)"})
    write_table(pd.DataFrame(rows), tables_dir / "table7_base_instruct",
                "Table 7. Paired base vs instruct comparisons (instruct - base; paired word-level bootstrap; BH across layers)")


def table8(metrics: pd.DataFrame, tables_dir):
    m = metrics[(metrics.status == "ok") & metrics.target.isin(["log_subtlex", "log1p_wn_n_all", "gcide_nonzero",
                                                                  "gcide_recode", "ambiguity_bin"])]
    if m.empty:
        return
    col = np.where(m.kind == "reg", m.r2_lin, m.get("auroc", np.nan))
    m = m.assign(score=col)
    best = m.groupby(["model_id", "eval_set", "target", "control_set", "pooling"]).score.max().reset_index()
    best["variant"] = best.eval_set + " | " + best.control_set + " | " + best.pooling
    piv = best.pivot_table(index=["model_id", "target"], columns="variant", values="score").reset_index()
    piv["model_id"] = piv.model_id.str.split("/").str[-1]
    write_table(piv, tables_dir / "table8_robustness",
                "Table 8. Robustness: peak score (R2 or AUROC) by analysis set | control set | pooling")


def tableS(metrics: pd.DataFrame, tables_dir, attr_missing: pd.DataFrame | None = None):
    metrics.to_csv(tables_dir / "tableS1_full_layer_metrics.csv", index=False)
    if attr_missing is not None:
        write_table(attr_missing, tables_dir / "tableS2_missingness", "Table S2. Missingness by variable and provenance")
