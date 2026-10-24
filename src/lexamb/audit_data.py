"""Dataset validation, predefined analysis subsets and the audit tables (spec §3)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .common import write_table

LOG = logging.getLogger("lexamb")

STIMULUS_COLS = ["word", "provenance", "source_set", "pool_condition", "appendix", "experiment",
                 "modality", "ambiguity", "senses"]
FORM_COLS = ["subtlex", "log_subtlex", "length", "n_syll", "n_phon", "ortho_n"]
SENSE_COLS = ["gcide_entries", "gcide_etymologies", "wn_n_all", "wn_n_noun", "wn_n_verb", "wn_n_adj",
              "wn_n_adv", "wn_n_pos", "wn_domains_all", "wn_domains_noun", "sim_top2", "min_sim",
              "mean_sim", "ambiguity_class"]
SEMANTIC_COLS = ["concreteness", "conc_sd", "imageability", "familiarity", "meaningfulness", "aoa"]
REQUIRED_COLS = STIMULUS_COLS + FORM_COLS + SENSE_COLS + SEMANTIC_COLS

# Columns that must NEVER reach a prompt (spec §2.12). Used by the leakage unit test.
LABEL_COLS = ["ambiguity", "senses", "pool_condition", "ambiguity_class", "gcide_etymologies",
              "gcide_entries", "wn_n_all", "sim_top2", "min_sim", "mean_sim"]


class DataValidationError(RuntimeError):
    pass


def load_and_validate(csv_path: str | Path, cfg_data: dict) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    problems = []
    exp_shape = (cfg_data["expected_rows"], cfg_data["expected_cols"])
    if df.shape != exp_shape:
        problems.append(f"shape {df.shape} != expected {exp_shape}")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        problems.append(f"missing required columns: {missing}")
    prov = df["provenance"].value_counts().to_dict()
    for k, v in cfg_data["expected_provenance"].items():
        if prov.get(k) != v:
            problems.append(f"provenance[{k}] = {prov.get(k)} != {v}")
    uniq = df.groupby("provenance")["word"].apply(lambda s: s.str.lower().nunique()).to_dict()
    for k, v in cfg_data["expected_unique_words"].items():
        if uniq.get(k) != v:
            problems.append(f"unique words[{k}] = {uniq.get(k)} != {v}")
    bc = df[(df.provenance == "published") & df.appendix.isin(["B", "C"]) & df.senses.notna()]
    bc_counts = bc.appendix.value_counts().to_dict()
    for k, v in cfg_data["expected_bc_rows"].items():
        if bc_counts.get(k) != v:
            problems.append(f"Appendix {k} factorial rows = {bc_counts.get(k)} != {v}")
    # word-level variables must be identical across duplicated word forms, else dedup is unsafe
    word_level = ["ambiguity", "log_subtlex", "wn_n_all", "gcide_etymologies", "length", "concreteness",
                  "sim_top2", "min_sim", "mean_sim", "wn_n_noun"]
    key = df.word.str.lower()
    for c in word_level:
        n = df.groupby(key)[c].nunique(dropna=False)
        if (n > 1).any():
            problems.append(f"column {c} differs across duplicate rows of {list(n[n > 1].index[:5])}")
    if problems:
        raise DataValidationError("Dataset validation failed:\n  - " + "\n  - ".join(problems))
    df = df.copy()
    df.insert(0, "row_id", np.arange(len(df)))
    df["word_lc"] = key
    LOG.info("dataset validated: %s rows, %s unique word forms", len(df), df.word_lc.nunique())
    return df


def unique_words(df: pd.DataFrame) -> pd.DataFrame:
    """One row per lowercase word form (the unit of extraction: the prompt depends only on the word)."""
    u = df.drop_duplicates("word_lc", keep="first").reset_index(drop=True)
    u["word_idx"] = np.arange(len(u))
    return u


def add_derived_targets(items: pd.DataFrame) -> pd.DataFrame:
    """Derived analysis variables. Labels stay NaN where undefined; nothing is imputed."""
    t = items.copy()
    t["log1p_wn_n_all"] = np.log1p(t["wn_n_all"])
    t["log1p_wn_n_noun"] = np.log1p(t["wn_n_noun"])
    g = t["gcide_etymologies"]
    t["gcide_nonzero"] = g.where(g > 0)                         # primary: zeros & missing excluded
    t["gcide_recode"] = g.where(g.isna(), np.maximum(g, 1))     # sensitivity: zero -> 1
    t["gcide_zero"] = (g == 0).astype(float).where(g.notna())   # "no etymology recorded" indicator
    t["gcide_multi"] = (t["gcide_nonzero"] >= 2).astype(float).where(t["gcide_nonzero"].notna())
    t["ambiguity_bin"] = t["ambiguity"].map({"ambiguous": 1.0, "unambiguous": 0.0})
    t["senses_bin"] = t["senses"].map({"many": 1.0, "few": 0.0})
    t["pool_condition_bin"] = t["pool_condition"].map({"homonym-candidate": 1.0, "polyseme-candidate": 0.0})
    return t


@dataclass
class EvalSet:
    name: str
    pool: str           # "published" | "derived"
    description: str
    items: pd.DataFrame  # deduplicated by word_lc; carries word_idx into the activation tensor
    strat: np.ndarray    # stratification labels for grouped CV
    notes: list = field(default_factory=list)


def _strat_quantiles(x: pd.Series, q: int = 5) -> np.ndarray:
    ranks = x.rank(method="first")
    return pd.qcut(ranks, q=min(q, max(1, len(x) // 10) or 1), labels=False).to_numpy()


def build_eval_sets(df: pd.DataFrame, cfg: dict) -> dict[str, EvalSet]:
    """Predefined subsets (spec §3). Each set is deduplicated *within* itself by word form; the
    provenance groups are never mixed (spec §2.10)."""
    words = unique_words(df)[["word_lc", "word_idx"]]
    tgt = add_derived_targets(df)
    pub = tgt[tgt.provenance == "published"]
    der = tgt[tgt.provenance == "wordnet-derived"]
    s2 = cfg["stage2"]

    def mk(name, pool, desc, sub, strat_col=None):
        notes = []
        n_rows = len(sub)
        sub = sub.drop_duplicates("word_lc", keep="first").merge(words, on="word_lc", how="left")
        if len(sub) != n_rows:
            notes.append(f"deduplicated {n_rows} rows -> {len(sub)} unique words")
        if strat_col == "cell":
            strat = (sub["ambiguity"].fillna("NA").astype(str) + "|" + sub["senses"].fillna("NA").astype(str)).astype("category").cat.codes.to_numpy()
        elif strat_col is not None:
            strat = sub[strat_col].astype("category").cat.codes.to_numpy()
        else:
            strat = _strat_quantiles(sub["log_subtlex"].fillna(sub["log_subtlex"].median()))
        return EvalSet(name, pool, desc, sub.reset_index(drop=True), strat, notes)

    sets = {
        "pub_all": mk("pub_all", "published", "All published Rodd et al. (2002) words (A+B+C, deduplicated)",
                      pub, "ambiguity"),
        "pub_A": mk("pub_A", "published", "Appendix A (Experiment 1, visual); ambiguity confounded with senses",
                    pub[pub.appendix == "A"], "ambiguity"),
        "pub_B": mk("pub_B", "published", "Appendix B (Experiment 2, visual) 2x2 ambiguity x senses",
                    pub[(pub.appendix == "B") & pub.senses.notna()], "cell"),
        "pub_C": mk("pub_C", "published",
                    "Appendix C (Experiment 3, auditory stimulus set presented as TEXT) 2x2 ambiguity x senses",
                    pub[(pub.appendix == "C") & pub.senses.notna()], "cell"),
        "pub_BC": mk("pub_BC", "published", "Appendix B and C pooled (deduplicated; sensitivity only)",
                     pub[pub.appendix.isin(["B", "C"]) & pub.senses.notna()], "cell"),
    }
    matched = der[der.length.between(*s2["length_range"]) & der.log_subtlex.between(*s2["log_subtlex_range"])]
    sets["der_matched"] = mk("der_matched", "derived",
                             f"WordNet-derived, length {s2['length_range']} and log_subtlex {s2['log_subtlex_range']}",
                             matched)
    sets["der_full"] = mk("der_full", "derived", "All WordNet-derived words (sensitivity)", der)
    for s in sets.values():
        LOG.info("eval set %s: N=%d (%s) %s", s.name, len(s.items), s.pool, "; ".join(s.notes))
    return sets


def stage2_attrition(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s2 = cfg["stage2"]
    d = df[df.provenance == "wordnet-derived"]
    steps = [("WordNet-derived pool", d)]
    d1 = d[d.length.between(*s2["length_range"])]
    steps.append((f"length in {s2['length_range']}", d1))
    d2 = d1[d1.log_subtlex.between(*s2["log_subtlex_range"])]
    steps.append((f"log_subtlex in {s2['log_subtlex_range']}", d2))
    d3 = d2[d2.gcide_etymologies.notna()]
    steps.append(("gcide_etymologies observed (zeros kept; recode-to-1 sensitivity set)", d3))
    d4 = d3[d3.gcide_etymologies > 0]
    steps.append(("gcide_etymologies > 0 (PRIMARY Stage 2 set)", d4))
    d5 = d4[d4.concreteness.notna()]
    steps.append(("+ concreteness observed (primary-control complete cases)", d5))
    return pd.DataFrame([{"step": i, "filter": name, "n_rows": len(x), "n_unique_words": x.word_lc.nunique()}
                         for i, (name, x) in enumerate(steps)])


def write_audit(df: pd.DataFrame, cfg: dict, out: dict[str, Path], eval_sets: dict[str, EvalSet]) -> dict:
    a = out["audit"]
    # overall counts
    summary = []
    for keys, g in df.groupby(["provenance", "source_set"], dropna=False):
        summary.append({"provenance": keys[0], "source_set": keys[1], "n_rows": len(g),
                        "n_unique_words": g.word_lc.nunique(),
                        "appendix": ",".join(sorted(g.appendix.dropna().unique())),
                        "experiment": ",".join(str(int(x)) for x in sorted(g.experiment.dropna().unique())),
                        "modality": ",".join(sorted(g.modality.dropna().unique())) or "n/a"})
    summary = pd.DataFrame(summary)
    write_table(summary, a / "audit_counts", "Rows and unique word forms by provenance and source set")

    miss = []
    for prov, g in df.groupby("provenance"):
        for c in REQUIRED_COLS:
            miss.append({"provenance": prov, "column": c, "n": len(g), "n_missing": int(g[c].isna().sum()),
                         "pct_missing": float(g[c].isna().mean() * 100)})
    miss = pd.DataFrame(miss)
    write_table(miss, a / "audit_missingness", "Missingness per variable and provenance (Table S2)")

    dup = df[df.duplicated("word_lc", keep=False)].groupby("word_lc").agg(
        n_rows=("row_id", "size"), appendices=("appendix", lambda s: ",".join(sorted(s.dropna().astype(str)))),
        provenance=("provenance", lambda s: ",".join(sorted(set(s))))).reset_index()
    write_table(dup, a / "audit_duplicates", "Duplicated word forms (all within the published pool)")

    pub = df[df.provenance == "published"]
    cells = pd.crosstab([pub.appendix, pub.modality, pub.ambiguity], pub.senses.fillna("n/a")).reset_index()
    cells.columns = [str(c) for c in cells.columns]
    write_table(cells, a / "audit_design_cells", "Published design cells (appendix x modality x ambiguity x senses)")

    attr = stage2_attrition(df, cfg)
    write_table(attr, a / "audit_stage2_attrition", "Stage 2 filter attrition (WordNet-derived pool)")

    es = pd.DataFrame([{"eval_set": s.name, "pool": s.pool, "n_unique_words": len(s.items),
                        "description": s.description, "notes": "; ".join(s.notes)} for s in eval_sets.values()])
    write_table(es, a / "audit_eval_sets", "Analysis sets used for probing (deduplicated within set)")

    # predictor correlations and VIF (per pool)
    preds = ["log_subtlex", "length", "concreteness", "ortho_n", "n_syll", "n_phon", "wn_n_all",
             "gcide_etymologies", "sim_top2", "min_sim", "mean_sim"]
    t = add_derived_targets(df)
    for prov, g in t.groupby("provenance"):
        g = g.drop_duplicates("word_lc")
        g[preds].corr(method="spearman").to_csv(a / f"audit_spearman_{prov}.csv")
    vif = predictor_vif(t, ["log_subtlex", "length", "concreteness", "log1p_wn_n_all", "gcide_nonzero"])
    write_table(vif, a / "audit_vif", "Variance inflation factors of the Stage 2 primary predictors")

    key = {
        "n_rows": len(df), "n_unique_words": int(df.word_lc.nunique()),
        "provenance": df.provenance.value_counts().to_dict(),
        "appendix_rows": pub.appendix.value_counts().to_dict(),
        "bc_factorial_rows": {k: int(v) for k, v in
                              pub[pub.senses.notna()].appendix.value_counts().items()},
        "stage2_attrition": attr.to_dict(orient="records"),
        "eval_sets": {s.name: len(s.items) for s in eval_sets.values()},
    }
    return key


def predictor_vif(t: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for prov, g in t.groupby("provenance"):
        x = g.drop_duplicates("word_lc")[cols].dropna()
        if len(x) < len(cols) + 2:
            continue
        z = ((x - x.mean()) / x.std(ddof=0)).to_numpy()
        for j, c in enumerate(cols):
            others = np.delete(z, j, axis=1)
            X = np.column_stack([np.ones(len(z)), others])
            beta, *_ = np.linalg.lstsq(X, z[:, j], rcond=None)
            r2 = 1 - np.sum((z[:, j] - X @ beta) ** 2) / np.sum((z[:, j] - z[:, j].mean()) ** 2)
            rows.append({"provenance": prov, "predictor": c, "n": len(x), "vif": 1 / max(1 - r2, 1e-12)})
    return pd.DataFrame(rows)
