#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Layer-wise lexical frequency and semantic ambiguity in base vs instruction-tuned LLMs
====================================================================================
ONE self-contained, end-to-end script: audit -> extract -> analyse -> compare -> report.

Research questions (see the specification / docs/RESEARCH_RATIONALE.md)
  RQ1 frequency (log_subtlex) across layers          RQ4 semantic distance vs sense count
  RQ2 sense count (wn_n_all) beyond frequency        RQ5 matched base vs instruct pairs
  RQ3 published meanings factor / GCIDE proxy        RQ6 Rodd et al. (2002) directional dissociation

Design summary
  * Neutral prompts only ("Word: {word}"; instruct models: native chat template with that content);
    labels are joined to the representations only after extraction (no leakage).
  * Hidden state at the target word's last subtoken (primary) / mean of subtokens (sensitivity),
    every layer (0 = embeddings, 1..L = blocks), one independent sequence per word.
  * Probes: dual-form ridge, alpha by exact leave-one-out, covariates unpenalised and partialled out
    by Frisch-Waugh-Lovell inside each fold (conditional probing); repeated StratifiedGroupKFold by
    word with identical folds for all models; Freedman-Lane permutation null shared by all layers
    (per-layer p, BH-FDR and Westfall-Young max-T across layers); word-level bootstrap CIs.
  * Stage 1 (published): Appendix B and C each in ONE effect-coded 2x2 model on the pre-registered
    representational settling cost (RSC; higher = harder); layer-wise time course; cross-condition
    generalisation (CCGP); factorial selectivity of the probes. Appendix A separately (confounded).
  * Stage 2 (WordNet-derived, generalisation only): proxy ~ sense count + GCIDE proxy + controls,
    with GCIDE-zero, full-pool, expanded-control, noun-sense and semantic-distance sensitivities.
  * Base vs instruct: paired word-level bootstrap per layer, centre-of-mass shift, CKA (exploratory).
  * Figures (PDF/SVG/PNG-300dpi + plotted data CSV), tables (CSV/Markdown/LaTeX), Markdown report,
    run_manifest.json with config, data/script hashes, package versions and model commits.

Usage
  python lexical_ambiguity_pipeline.py --self-test                    # offline checks (no download)
  python lexical_ambiguity_pipeline.py --stage audit                  # validate the CSV
  python lexical_ambiguity_pipeline.py --dry-run                      # 10 words, SmolLM2-360M, all stages
  python lexical_ambiguity_pipeline.py --stage all --resume --device cuda
  python lexical_ambiguity_pipeline.py --stage extract --models Qwen/Qwen2.5-3B Qwen/Qwen2.5-3B-Instruct
  python lexical_ambiguity_pipeline.py --write-config my.yaml         # dump the embedded config, edit,
  python lexical_ambiguity_pipeline.py --config my.yaml --stage all   # ... and run with it

Requirements: torch, transformers>=4.53, tokenizers, numpy>=2, pandas, scipy, scikit-learn,
statsmodels, matplotlib, pyarrow, pyyaml. Llama-3.x repositories are gated: accept the licence on
Hugging Face and run `huggingface-cli login` (or set HF_TOKEN) first.

Interpretation guard-rails: a probe peak means "most linearly decodable under this protocol", not
"where the model resolves ambiguity"; no proxy is a human reaction time; Appendix C is the auditory
stimulus SET presented as text; GCIDE etymologies and WordNet candidate labels are computed, not human
norms; Stage 2 is generalisation, not replication; base/instruct inputs differ by the chat wrapper.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import logging
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from scipy import stats as sps  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

__version__ = "1.0.0"
ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("lexamb")


class _LazyTorch:
    """torch is imported on first use, so `--stage audit` / report work without PyTorch installed."""

    def __getattr__(self, name):
        import torch as _torch
        return getattr(_torch, name)


torch = _LazyTorch()

# ==================================================================================================
# Embedded experiment configuration: the single source of truth for every analysis decision.
# Dump it with --write-config, edit, and pass it back with --config.
# ==================================================================================================
DEFAULT_CONFIG_YAML = r"""
# =============================================================================
# Layer-wise lexical frequency & semantic ambiguity in base vs instruct LLMs
# Single source of truth for every analysis decision (spec §10).
# Nothing analytic is hard-coded elsewhere in the script; every stage reads from this block.
# =============================================================================

data:
  csv: ambiguity_items_combined.csv
  expected_rows: 2035
  expected_cols: 35
  expected_provenance: {published: 402, wordnet-derived: 1633}
  expected_unique_words: {published: 309, wordnet-derived: 1633}
  expected_bc_rows: {B: 128, C: 92}

output_dir: outputs
seed: 20260101

models:
  # id: HF hub id; family; variant; pair: base partner (instruct only); size label
  - {id: HuggingFaceTB/SmolLM2-360M,          family: smollm, variant: base,     size: 360M}
  - {id: HuggingFaceTB/SmolLM3-3B-Base,       family: smollm, variant: base,     size: 3B}
  - {id: Qwen/Qwen2.5-1.5B,                   family: qwen,   variant: base,     size: 1.5B}
  - {id: Qwen/Qwen2.5-3B,                     family: qwen,   variant: base,     size: 3B}
  - {id: Qwen/Qwen3-4B-Base,                  family: qwen,   variant: base,     size: 4B}
  - {id: Qwen/Qwen3-8B-Base,                  family: qwen,   variant: base,     size: 8B}
  - {id: meta-llama/Llama-3.2-1B,             family: llama,  variant: base,     size: 1B}
  - {id: meta-llama/Llama-3.2-3B,             family: llama,  variant: base,     size: 3B}
  - {id: meta-llama/Llama-3.1-8B,             family: llama,  variant: base,     size: 8B}
  - {id: HuggingFaceTB/SmolLM2-360M-Instruct, family: smollm, variant: instruct, size: 360M, pair: HuggingFaceTB/SmolLM2-360M}
  - {id: Qwen/Qwen2.5-1.5B-Instruct,          family: qwen,   variant: instruct, size: 1.5B, pair: Qwen/Qwen2.5-1.5B}
  - {id: Qwen/Qwen2.5-3B-Instruct,            family: qwen,   variant: instruct, size: 3B,   pair: Qwen/Qwen2.5-3B}
  - {id: meta-llama/Llama-3.2-1B-Instruct,    family: llama,  variant: instruct, size: 1B,   pair: meta-llama/Llama-3.2-1B}
  - {id: meta-llama/Llama-3.2-3B-Instruct,    family: llama,  variant: instruct, size: 3B,   pair: meta-llama/Llama-3.2-3B}
  - {id: meta-llama/Llama-3.1-8B-Instruct,    family: llama,  variant: instruct, size: 8B,   pair: meta-llama/Llama-3.1-8B}

extraction:
  revision: main               # pin a commit hash for the final run; the resolved hash is always logged
  dtype: auto                  # auto -> bfloat16 on CUDA (if supported) else float16 on CUDA, float32 on CPU
  quantization: none           # keep "none"; quantising one member of a pair breaks comparability (spec §4)
  batch_size: 16               # independent items, right-padded + attention mask; no cross-item text
  shard_size: 256              # unique words per atomic shard (resumable)
  storage_dtype: float16       # raw activations on disk; all statistics are computed in float32/64
  base_template: "Word: {word}"
  chat_content_template: "Word: {word}"
  # Llama-3.x chat templates inject today's date unless date_string is given -> fix it for reproducibility
  chat_template_kwargs: {date_string: "26 Jul 2024"}
  poolings: [last, mean]       # last target subtoken = primary; mean of target subtokens = sensitivity
  alignment_audit_n: 50        # words per tokenizer written to the manual alignment audit sheet
  logit_lens: true             # secondary output-space settling proxy

proxies:
  # Pre-registered PRIMARY model-derived proxy for the Rodd directional check (RQ6):
  #   representational settling cost (RSC) = area under 1 - cos(z_l, z_L) over relative depth,
  #   z_l = per-layer, per-dimension standardised target state. Orientation: HIGHER = MORE DIFFICULT.
  # Expected (attractor/semantic-settling account): many senses -> lower RSC; ambiguous -> higher RSC.
  primary: rsc_last
  orientation: higher_is_harder
  timecourse_layers_rel: [0.0, 0.25, 0.5, 0.75]   # preregistered depths for Figure 4 point estimates

ease_rating:                   # Outcome D, exploratory only, instruct models only
  enabled: false
  scale: [1, 2, 3, 4, 5, 6, 7]
  paraphrases:
    - "How easy is it to recognise the English word \"{word}\"? Answer with one number from 1 (very hard) to 7 (very easy)."
    - "Rate how quickly a reader would recognise \"{word}\" as an English word, from 1 (very slowly) to 7 (very quickly). Reply with a single digit."
    - "On a 1-7 scale (1 = very difficult, 7 = very easy), how easy is the word \"{word}\" to identify? Give only the number."

probes:
  n_folds: 5
  n_repeats: 3                 # repeated grouped CV (different fold seeds)
  ridge_lambdas_rel: [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
  n_permutations: 200          # label-permutation null (alpha re-selected inside every permutation)
  n_bootstrap: 1000            # grouped (word-level) bootstrap for metric CIs
  min_n: 20                    # skip (and log) any probe with fewer items
  primary_controls: [log_subtlex, length, concreteness]
  expanded_controls: [log_subtlex, length, concreteness, ortho_n, n_syll, n_phon]
  # Control sets: none = intercept only (baseline = training mean / class prior);
  #   primary = primary_controls (+ partner variable, e.g. log1p(wn_n_all) when probing frequency/GCIDE/
  #   similarity/ambiguity, so the increment is information beyond sense count);
  #   expanded = expanded_controls (+ partner) on their own complete cases (sensitivity);
  #   primary_tok = primary + log(number of target subtokens) (tokenisation sensitivity).
  plan:
    pub_all:
      log_subtlex: [none, primary, expanded, primary_tok]
      log1p_wn_n_all: [none, primary, expanded, primary_tok]
      wn_n_all: [none]
      log1p_wn_n_noun: [none]
      gcide_nonzero: [none, primary]
      gcide_recode: [none, primary]
      sim_top2: [none, primary]
      min_sim: [none, primary]
      mean_sim: [none, primary]
      ambiguity_bin: [none, primary, expanded, primary_tok]
    pub_A:
      ambiguity_bin: [none, primary]
    pub_B:
      ambiguity_bin: [none, primary]
      senses_bin: [none, primary]
    pub_C:
      ambiguity_bin: [none, primary]
      senses_bin: [none, primary]
    pub_BC:
      ambiguity_bin: [none]
      senses_bin: [none]
    der_matched:
      log_subtlex: [none, primary, expanded, primary_tok]
      log1p_wn_n_all: [none, primary, expanded, primary_tok]
      wn_n_all: [none]
      log1p_wn_n_noun: [none]
      gcide_nonzero: [none, primary, primary_tok]
      gcide_recode: [none, primary]
      gcide_multi: [none]
      sim_top2: [none, primary]
      min_sim: [none, primary]
      mean_sim: [none, primary]
      pool_condition_bin: [none]
    der_full:
      log_subtlex: [none, primary]
      log1p_wn_n_all: [none, primary]
      gcide_nonzero: [none, primary]
      gcide_recode: [none]
      sim_top2: [none, primary]
      pool_condition_bin: [none]

stage2:
  length_range: [4, 6]
  log_subtlex_range: [2.0, 4.0]
  gcide_zero_policy_primary: exclude     # exclude zeros & missing (primary); sensitivity: recode zero -> 1

rodd:
  n_bootstrap: 2000            # cell-stratified bootstrap for factorial coefficients
  n_permutations: 500          # CCGP within-condition label permutations
  coding: effect               # +0.5/-0.5 coding so main effects are averaged over the other factor

fdr:
  alpha: 0.05
  # BH family: one (model, analysis, eval_set, target, pooling, control_set) across all layers
  family: per_model_target_evalset_across_layers

dry_run:
  n_words: 10
  models: [HuggingFaceTB/SmolLM2-360M]
"""


# ==================================================================================================
# Shared utilities: configuration, logging, hashing, atomic IO, table writers   [section: common]
# ==================================================================================================
# Shared utilities: configuration, logging, hashing, atomic IO, table writers.




OUTPUT_SUBDIRS = ("audit", "activations", "predictions", "metrics", "tables",
                  "figures", "figures_png", "reports")


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Embedded DEFAULT_CONFIG_YAML, or a YAML file with the same schema when a path is given."""
    if path is None:
        cfg = yaml.safe_load(DEFAULT_CONFIG_YAML)
    else:
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
    cfg = copy.deepcopy(cfg)
    for key, value in (overrides or {}).items():
        if value is not None:
            cfg[key] = value
    return cfg


def setup_logging(log_dir: str | Path, name: str = "run") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    class _JsonFormatter(logging.Formatter):
        def format(self, record):
            return json.dumps({"t": self.formatTime(record), "lvl": record.levelname,
                               "mod": record.module, "msg": record.getMessage()})

    root = logging.getLogger("lexamb")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fh = logging.FileHandler(Path(log_dir) / f"{name}.jsonl")
    fh.setFormatter(_JsonFormatter())
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(fh)
    root.addHandler(sh)


def out_dirs(output_dir: str | Path) -> dict[str, Path]:
    base = Path(output_dir)
    dirs = {name: base / name for name in OUTPUT_SUBDIRS}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    dirs["root"] = base
    return dirs


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_id)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npy.tmp")
    with os.fdopen(fd, "wb") as fh:          # file handle: np.save must not append ".npy" to the temp name
        np.save(fh, array)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def atomic_to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    os.close(fd)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    atomic_write_text(Path(path), json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ----------------------------------------------------------------------------- tables
def _latex_escape(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    return "".join(rep.get(ch, ch) for ch in str(s))


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.2e}"
        return f"{v:.3f}"
    return str(v)


def write_table(df: pd.DataFrame, stem: Path, caption: str = "") -> None:
    """Write CSV + Markdown + LaTeX (booktabs) versions of a table."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    cols = list(df.columns)
    md = [f"**{caption}**\n" if caption else ""]
    md.append("| " + " | ".join(map(str, cols)) + " |")
    md.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        md.append("| " + " | ".join(_fmt(row[c]).replace("|", "/") for c in cols) + " |")
    atomic_write_text(stem.with_suffix(".md"), "\n".join(md) + "\n")
    tex = [r"\begin{table}[ht]", r"\centering", r"\small"]
    if caption:
        tex.append(r"\caption{" + _latex_escape(caption) + "}")
    tex.append(r"\begin{tabular}{" + "l" * len(cols) + "}")
    tex.append(r"\toprule")
    tex.append(" & ".join(_latex_escape(c) for c in cols) + r" \\")
    tex.append(r"\midrule")
    for _, row in df.iterrows():
        tex.append(" & ".join(_latex_escape(_fmt(row[c])) for c in cols) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    atomic_write_text(stem.with_suffix(".tex"), "\n".join(tex) + "\n")


# ==================================================================================================
# Dataset validation, predefined analysis subsets, audit tables (spec §3)   [section: audit_data]
# ==================================================================================================
# Dataset validation, predefined analysis subsets and the audit tables (spec §3).





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


# ==================================================================================================
# Model inventory, base/instruct pairing, loading (spec §4)   [section: model_registry]
# ==================================================================================================
# Model inventory, base/instruct pairing and loading (spec §4).




@dataclass(frozen=True)
class ModelSpec:
    id: str
    family: str
    variant: str          # "base" | "instruct"
    size: str
    pair: str | None = None

    @property
    def short(self) -> str:
        return self.id.split("/")[-1]


def registry(cfg: dict) -> list[ModelSpec]:
    specs = [ModelSpec(m["id"], m["family"], m["variant"], str(m["size"]), m.get("pair")) for m in cfg["models"]]
    ids = {s.id for s in specs}
    for s in specs:
        if s.variant == "instruct" and s.pair not in ids:
            raise ValueError(f"instruct model {s.id} pairs with {s.pair}, which is not in the inventory")
        if s.variant == "base" and s.pair:
            raise ValueError(f"base model {s.id} must not declare a pair")
    return specs


def pairs(specs: list[ModelSpec]) -> list[tuple[ModelSpec, ModelSpec]]:
    by_id = {s.id: s for s in specs}
    return [(by_id[s.pair], s) for s in specs if s.variant == "instruct"]


def base_only(specs: list[ModelSpec]) -> list[ModelSpec]:
    paired = {s.pair for s in specs if s.variant == "instruct"}
    return [s for s in specs if s.variant == "base" and s.id not in paired]


def select(specs: list[ModelSpec], ids: list[str] | None) -> list[ModelSpec]:
    if not ids:
        return specs
    by_id = {s.id: s for s in specs}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise ValueError(f"models not in the configured inventory: {unknown}")
    return [by_id[i] for i in ids]


def resolve_dtype(name: str, device: str):
    import torch
    if name == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def load_model(spec: ModelSpec, revision: str, dtype_name: str, device: str):
    """Load tokenizer + causal LM in eval mode. Gated (Llama) repos raise a helpful error."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import transformers

    dtype = resolve_dtype(dtype_name, device)
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    try:
        tok = AutoTokenizer.from_pretrained(spec.id, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(spec.id, revision=revision, **{dtype_kw: dtype})
    except OSError as err:
        msg = str(err)
        if "gated" in msg.lower() or "401" in msg or "403" in msg or "access" in msg.lower():
            raise RuntimeError(
                f"Cannot download {spec.id}: the repository is gated. Accept the licence on its Hugging Face "
                f"page and authenticate with `huggingface-cli login` (or set HF_TOKEN), then rerun with --resume."
            ) from err
        raise
    if not getattr(tok, "is_fast", False):
        raise RuntimeError(f"{spec.id}: a fast tokenizer is required for character offsets")
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    cfg = model.config
    meta = {
        "model_id": spec.id, "family": spec.family, "variant": spec.variant, "size": spec.size,
        "paired_base": spec.pair, "requested_revision": revision,
        "model_commit": getattr(cfg, "_commit_hash", None),
        "tokenizer_commit": tok.init_kwargs.get("_commit_hash"),
        "n_layers": int(cfg.num_hidden_layers), "hidden_size": int(cfg.hidden_size),
        "vocab_size": int(getattr(cfg, "vocab_size", len(tok))), "tokenizer_len": len(tok),
        "dtype": str(dtype).replace("torch.", ""), "device": device, "quantization": "none",
        "architecture": (cfg.architectures or ["?"])[0],
    }
    LOG.info("loaded %s (%s layers, d=%s, %s on %s)", spec.id, meta["n_layers"], meta["hidden_size"],
             meta["dtype"], device)
    return tok, model, meta


# ==================================================================================================
# Neutral prompts and target-subtoken alignment (spec §5)   [section: prompt_formatting]
# ==================================================================================================
# Neutral prompt rendering and target-subtoken alignment (spec §5).
#
# Base models see ``Word: {word}`` as plain text; instruct models see the same content inside their
# native chat template (add_generation_prompt=False). No label, count or construct word ever enters a
# prompt. The hidden state is read at the target word's own subtokens, located by character offsets.
#




@dataclass
class RenderedPrompt:
    word: str
    text: str
    input_ids: list[int]
    target_idx: list[int]                 # indices of the target word's subtokens
    char_span: tuple[int, int]
    ok: bool
    reason: str = ""
    template_hash: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def last(self) -> int:
        return self.target_idx[-1]


def chat_template_hash(tokenizer) -> str:
    return sha256_text(tokenizer.chat_template or "") if getattr(tokenizer, "chat_template", None) else ""


def _ids_from_chat_template(tokenizer, messages, kwargs) -> list[int]:
    out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, **kwargs)
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return list(map(int, out))


def render(tokenizer, word: str, variant: str, cfg_ext: dict) -> RenderedPrompt:
    if variant == "base":
        template = cfg_ext["base_template"]
        text = template.format(word=word)
        start = len(template.split("{word}")[0])
        enc = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        ids, offsets = list(enc["input_ids"]), list(enc["offset_mapping"])
        thash = sha256_text(template)
        extra = {}
    else:
        content_tpl = cfg_ext["chat_content_template"]
        content = content_tpl.format(word=word)
        messages = [{"role": "user", "content": content}]
        kwargs = dict(cfg_ext.get("chat_template_kwargs") or {})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **kwargs)
        pos = text.rfind(content)
        if pos < 0:
            return RenderedPrompt(word, text, [], [], (-1, -1), False, "content not found in rendered template")
        start = pos + len(content_tpl.split("{word}")[0])
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = list(enc["input_ids"]), list(enc["offset_mapping"])
        ref = _ids_from_chat_template(tokenizer, messages, kwargs)
        thash = chat_template_hash(tokenizer)
        extra = {"template_ids_match": ids == ref}
        if ids != ref:
            return RenderedPrompt(word, text, ref, [], (start, start + len(word)), False,
                                  "apply_chat_template ids differ from offset tokenisation", thash, extra)
    end = start + len(word)
    span = (start, end)
    if text[start:end] != word:
        return RenderedPrompt(word, text, ids, [], span, False, "character span mismatch", thash, extra)
    tgt = [k for k, (s, e) in enumerate(offsets) if e > s and s < end and e > start]
    if not tgt:
        return RenderedPrompt(word, text, ids, [], span, False, "no token overlaps the target span", thash, extra)
    s0, _ = offsets[tgt[0]]
    _, e1 = offsets[tgt[-1]]
    if text[s0:start].strip() != "":
        return RenderedPrompt(word, text, ids, tgt, span, False, "first target token merges preceding text",
                              thash, extra)
    if e1 > end and text[end:e1].strip() != "":
        return RenderedPrompt(word, text, ids, tgt, span, False, "last target token merges following text",
                              thash, extra)
    if tgt != list(range(tgt[0], tgt[-1] + 1)):
        return RenderedPrompt(word, text, ids, tgt, span, False, "target tokens not contiguous", thash, extra)
    decoded = tokenizer.decode([ids[k] for k in tgt]).strip()
    if decoded != word:
        return RenderedPrompt(word, text, ids, tgt, span, False, f"decoded target {decoded!r} != word",
                              thash, extra)
    return RenderedPrompt(word, text, ids, tgt, span, True, "", thash, extra)


# ==================================================================================================
# Output-probability measures, Outcome C (spec §6)   [section: score_token_likelihood]
# ==================================================================================================
# Output-probability measures (spec §6, Outcome C) -- secondary, analysed separately.
#
# Scoring rule (identical for every item within a model):
#   * prefix  = everything before the target word in the model's neutral prompt (``Word:`` for base
#     models, the native chat wrapper + ``Word:`` for instruct models) -- common to all items;
#   * word log-likelihood = sum over target subtokens t of log p(x_t | x_<t);
#   * length-normalised = that sum / n_subtokens  (reported with n_subtokens; tokeniser-sensitive);
#   * next-token entropy at the last target subtoken (nats).
# These are lexical-form/predictability measures, NOT ambiguity measures and NOT reaction times.
#
# Logit lens (secondary settling proxy): at the last target subtoken, every layer's state is passed
# through the final norm + unembedding; KL(p_final || p_layer) per layer. Its area over relative depth
# is an output-space analogue of settling time (higher = later convergence).
#



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


# ==================================================================================================
# Resumable, sharded hidden-state extraction (spec §5)   [section: extract_hidden_states]
# ==================================================================================================
# Target-subtoken hidden states for every layer (spec §5), resumable and sharded.
#
# Layer convention (checked at runtime, not assumed): ``hidden_states`` has L+1 entries; index 0 is
# the embedding output, 1..L are transformer block outputs. In HF Llama/Qwen/SmolLM decoders the
# final entry already has the final norm applied -- this is verified numerically and logged.
#
# Unit of extraction = unique lowercase word form (the prompt depends only on the word, so duplicate
# rows would be identical forward passes). Row-level metadata is joined back at analysis time.
# Each item is its own sequence; batching only right-pads independent sequences under an attention
# mask (causal attention => padding never influences real positions).
#




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


# ==================================================================================================
# Outcome D: exploratory ease rating (spec §6)   [section: ease_rating]
# ==================================================================================================
# Outcome D (exploratory only): model-rated recognition ease, instruct models only (spec §6 D).
#
# Each word is an independent request. No label or count is shown. Instead of sampling text, the
# rating is read deterministically from the next-token distribution restricted to the fixed response
# vocabulary "1".."7": expected rating = sum_k k p(k) / sum_k p(k). Paraphrases are prompt-sensitivity
# conditions, not independent participants; inter-paraphrase agreement is reported. Higher = EASIER.
#





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


# ==================================================================================================
# Statistical machinery   [section: stats_utils]
# ==================================================================================================
# Statistical machinery (named stats_utils to avoid shadowing the stdlib ``statistics`` module).
#
# Probe estimator
# ---------------
# Ridge regression of a target y on a standardised layer state x with *unpenalised* covariates C
# (intercept always included):
#
#     min_{b, g} ||y - C b - X g||^2 + alpha ||g||^2
#
# By the Frisch-Waugh-Lovell theorem this equals ridge of M_C y on M_C X (M_C = I - P_C), so the
# covariates are fully partialled out and only the hidden state is shrunk. We solve it in the dual
# (kernel) form, K = X X^T, which is exact and fast when n << d (here n <= 1.6k, d up to 4096).
#
# alpha is chosen per target by exact leave-one-out error, using the closed-form hat diagonal
# h_ii = [P_C]_ii + sum_j U_ij^2 s_j/(s_j+alpha) (Golub, Heath & Wahba 1979; Hastie et al. ESL §7.10).
# The grid is relative to the mean kernel eigenvalue, making it scale-invariant across layers/models.
#
# With C = [1] this is an ordinary ridge probe whose covariate-only baseline is the training mean.
# With C = controls it is a *conditional probe* (Hewitt et al., EMNLP 2021): the increment
# R^2(C + h) - R^2(C) estimates information in the layer beyond the lexical covariates.
#



# ============================================================================ kernel ridge
def fwl_kernel_ridge(K_TT: np.ndarray, K_ET: np.ndarray, C_T: np.ndarray, C_E: np.ndarray,
                     Y_T: np.ndarray, lambdas_rel: np.ndarray):
    """Fit on training rows T, predict test rows E, for many target columns at once.

    Returns pred_E (nE, m), loo_T (nT, m) leave-one-out fitted values, cov_E (nE, m) covariate-only
    test predictions, chosen alpha index (m,).
    """
    nT = K_TT.shape[0]
    Q, _ = np.linalg.qr(C_T)
    pinvC = np.linalg.pinv(C_T)                       # (k, nT)
    b_y = pinvC @ Y_T                                 # covariate coefficients (OLS)
    Yr = Y_T - C_T @ b_y                              # M_C y
    KQ = K_TT @ Q
    Kr = K_TT - Q @ KQ.T - KQ @ Q.T + Q @ (Q.T @ KQ) @ Q.T  # M K M
    Kr = (Kr + Kr.T) / 2
    s, U = np.linalg.eigh(Kr)
    s = np.clip(s, 0, None)
    scale = max(s.sum() / max(nT - Q.shape[1], 1), 1e-12)
    alphas = np.asarray(lambdas_rel) * scale
    UtY = U.T @ Yr                                    # (nT, m)
    hP = np.sum(Q ** 2, axis=1)                       # diag(P_C)
    U2 = U ** 2
    best_err = np.full(Y_T.shape[1], np.inf)
    best_idx = np.zeros(Y_T.shape[1], dtype=int)
    best_loo = np.zeros_like(Y_T)
    for a_i, a in enumerate(alphas):
        f = s / (s + a)
        h = hP + U2 @ f
        fit_r = U @ (f[:, None] * UtY)
        loo_res = (Yr - fit_r) / np.clip(1 - h, 1e-8, None)[:, None]
        err = np.mean(loo_res ** 2, axis=0)
        better = err < best_err
        if better.any():
            best_err[better] = err[better]
            best_idx[better] = a_i
            best_loo[:, better] = (Y_T - loo_res)[:, better]
    # dual coefficients with each column's own alpha
    inv = 1.0 / (s[:, None] + alphas[best_idx][None, :])
    A = U @ (inv * UtY)                                # (nT, m)
    # test kernel against residualised training states: (K_ET - C_E pinvC K_TT) M
    K_Er = K_ET - C_E @ (pinvC @ K_TT)
    K_Er = K_Er - (K_Er @ Q) @ Q.T
    cov_E = C_E @ b_y
    pred_E = cov_E + K_Er @ A
    return pred_E, best_loo, cov_E, best_idx


def standardise_fit(X: np.ndarray):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def platt(train_scores: np.ndarray, train_y01: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """Platt (1999) scaling fitted on held-out (LOO) training decision values only."""
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(train_y01)) < 2:
        return np.full(len(test_scores), train_y01.mean())
    lr = LogisticRegression(C=1e4, max_iter=1000)
    lr.fit(train_scores.reshape(-1, 1), train_y01)
    return lr.predict_proba(test_scores.reshape(-1, 1))[:, 1]


# ============================================================================ metrics
def r2(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Out-of-sample R^2 (can be negative) along the last axis."""
    ss_res = np.sum((y - p) ** 2, axis=-1)
    ss_tot = np.sum((y - y.mean(axis=-1, keepdims=True)) ** 2, axis=-1)
    return 1 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)


def mae(y, p):
    return np.mean(np.abs(y - p), axis=-1)


def spearman(y, p):
    if np.std(p) == 0 or np.std(y) == 0:
        return np.nan
    return sps.spearmanr(y, p).statistic


def auroc(y01: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Rank-based AUROC (Mann-Whitney), vectorised over leading axes of s; y01 broadcastable."""
    ranks = sps.rankdata(s, axis=-1)
    pos = (y01 == 1)
    n1 = pos.sum(axis=-1)
    n0 = (~pos).sum(axis=-1)
    rs = np.sum(ranks * pos, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rs - n1 * (n1 + 1) / 2) / (n1 * n0)


def balanced_acc(y01, prob):
    pred = prob >= 0.5
    tpr = np.mean(pred[y01 == 1]) if np.any(y01 == 1) else np.nan
    tnr = np.mean(~pred[y01 == 0]) if np.any(y01 == 0) else np.nan
    return (tpr + tnr) / 2


def macro_f1(y01, prob):
    from sklearn.metrics import f1_score
    return f1_score(y01.astype(int), (prob >= 0.5).astype(int), average="macro", zero_division=0)


def brier(y01, prob):
    return np.mean((prob - y01) ** 2)


# ============================================================================ inference
def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg (1995) adjusted p-values; NaNs are left out of the family."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    if m == 0:
        return q
    ps = p[ok]
    order = np.argsort(ps)
    ranked = ps[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def perm_pvalue(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """One-sided permutation p, (1 + #{null >= obs}) / (1 + P) (Phipson & Smyth 2010). null: (P, ...)."""
    return (1 + np.sum(null >= obs[None], axis=0)) / (1 + null.shape[0])


def maxT_pvalue(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Westfall-Young max-statistic FWER-adjusted p across layers (same permutations for all layers)."""
    mx = np.nanmax(null, axis=1)
    return (1 + np.sum(mx[:, None] >= obs[None, :], axis=0)) / (1 + null.shape[0])


def boot_p_two_sided(boot: np.ndarray) -> np.ndarray:
    """Two-sided bootstrap p for H0: parameter = 0 (percentile inversion)."""
    lo = np.mean(boot <= 0, axis=0)
    hi = np.mean(boot >= 0, axis=0)
    return np.clip(2 * np.minimum(lo, hi), 1.0 / boot.shape[0], 1.0)


def ols_hc3(y: np.ndarray, X: np.ndarray, names: list[str]):
    """OLS with HC3 heteroscedasticity-consistent SEs (MacKinnon & White 1985)."""
    import statsmodels.api as sm
    res = sm.OLS(y, X).fit(cov_type="HC3")
    ci = res.conf_int()
    return {n: {"beta": float(res.params[i]), "se": float(res.bse[i]), "p": float(res.pvalues[i]),
                "ci_lo": float(ci[i, 0]), "ci_hi": float(ci[i, 1])} for i, n in enumerate(names)}, res


def vif(X: np.ndarray) -> np.ndarray:
    out = []
    for j in range(X.shape[1]):
        others = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])
        beta, *_ = np.linalg.lstsq(others, X[:, j], rcond=None)
        res = X[:, j] - others @ beta
        r2_ = 1 - res.var() / max(X[:, j].var(), 1e-12)
        out.append(1 / max(1 - r2_, 1e-12))
    return np.array(out)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al., ICML 2019) via centred Gram matrices."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    Kx, Ky = X @ X.T, Y @ Y.T
    hsic = np.sum(Kx * Ky)
    return float(hsic / np.sqrt(np.sum(Kx * Kx) * np.sum(Ky * Ky)))


def centre_of_mass_depth(depth: np.ndarray, score: np.ndarray) -> float:
    """Decodability-weighted mean relative depth, sum d*max(s,0) / sum max(s,0)."""
    w = np.clip(score, 0, None)
    return float(np.sum(depth * w) / np.sum(w)) if np.sum(w) > 0 else np.nan


def expected_layer_tenney(depth: np.ndarray, score: np.ndarray) -> float:
    """Tenney et al. (ACL 2019) expected layer from differential scores; increments clipped at 0
    so that the estimate stays inside [0, 1] when later layers lose information."""
    inc = np.clip(np.diff(score), 0, None)
    return float(np.sum(depth[1:] * inc) / np.sum(inc)) if np.sum(inc) > 0 else np.nan


# ==================================================================================================
# Layer-wise grouped-CV conditional probes (spec §6 A/B, §7)   [section: train_layerwise_probes]
# ==================================================================================================
# Layer-wise grouped-CV probes with conditional (covariate-partialled) increments (spec §6 A/B, §7).
#
# For every model x pooling x analysis set x target x control set and every layer:
#   * repeated StratifiedGroupKFold by word (identical folds for every model -> paired comparisons);
#   * all scaling, covariate regression and alpha selection inside training folds only;
#   * out-of-fold predictions of (i) covariates-only baseline and (ii) covariates + hidden state;
#   * metrics averaged over repeats; word-level bootstrap CIs; Freedman-Lane permutation null
#     (Freedman & Lane 1983; Ojala & Garriga 2010) shared by all layers -> per-layer p, BH-FDR across
#     layers and Westfall-Young max-T FWER across layers.
#





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
                    mu, sd = standardise_fit(X[tr_all])
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
                        pred_E, loo_T, cov_E, _ = fwl_kernel_ridge(K[np.ix_(tr, tr)], K[np.ix_(~tr, tr)],
                                                                     C_T, C_E, Y[tr], lambdas)
                        te = np.flatnonzero(~tr)
                        for m, jp in enumerate(gj):
                            jp["pred"][r, layer, te] = pred_E[:, m]
                            jp["cov"][r, te] = cov_E[:, m]
                            if jp["kind"] == "bin":
                                jp["prob"][r, layer, te] = platt(loo_T[:, m], (jp["y"][tr] > 0).astype(int),
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
    stem = f"{slug(model_meta['model_id'])}__{pooling}__{es.name}__{jp['target']}__{jp['control_set']}"
    np.savez_compressed(pred_dir / f"{stem}.npz", y=y, pred=jp["pred"], cov=jp["cov"], words=words,
                        prob=jp["prob"] if jp["prob"] is not None else np.zeros(0))
    # ---- point metrics (mean over repeats)
    r2_full = r2(y, pred)                                      # (R, L)
    r2_cov = r2(y, cov)                                        # (R,)
    out = {"r2_lin": r2_full.mean(0), "delta_r2_lin": (r2_full - r2_cov[:, None]).mean(0),
           "r2_lin_cov": np.repeat(r2_cov.mean(), L), "r2_lin_sd_repeats": r2_full.std(0)}
    if jp["kind"] == "reg":
        out["mae"] = mae(y, pred).mean(0)
        out["spearman"] = np.array([[spearman(y, pred[r, l]) for l in range(L)] for r in range(R)]).mean(0)
        primary, delta = "r2_lin", "delta_r2_lin"
    else:
        y01 = (y > 0).astype(int)
        prob = jp["prob"].astype(np.float64)
        au = auroc(y01, pred)
        au_cov = auroc(y01, cov)
        out["auroc"] = au.mean(0)
        out["auroc_cov"] = np.repeat(au_cov.mean(), L)
        out["delta_auroc"] = (au - au_cov[:, None]).mean(0)
        out["balanced_acc"] = np.array([[balanced_acc(y01, prob[r, l]) for l in range(L)] for r in range(R)]).mean(0)
        out["macro_f1"] = np.array([[macro_f1(y01, prob[r, l]) for l in range(L)] for r in range(R)]).mean(0)
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
            m_full = r2(yb[None, None], pb)                          # (R, L, b)
            m_cov = r2(yb[None], cb)                                 # (R, b)
        else:
            yb01 = (yb > 0).astype(int)
            m_full = auroc(yb01[None, None], pb)
            m_cov = auroc(yb01[None], cb)
        boot_p.append(np.nanmean(m_full, 0))
        boot_d.append(np.nanmean(m_full - m_cov[:, None], 0))
    boot_p = np.concatenate(boot_p, axis=1)                            # (L, B)
    boot_d = np.concatenate(boot_d, axis=1)
    # ---- permutation test of the conditional increment on repeat 0
    P = jp["ystar"].shape[1]
    obs = r2(y, pred[0]) - r2(y, cov[0])                         # (L,)
    if P:
        ys = jp["ystar"].T                                              # (P, n)
        pp = np.transpose(jp["perm_pred"], (2, 0, 1)).astype(np.float64)  # (P, L, n)
        null = r2(ys[:, None], pp) - r2(ys, jp["perm_cov"].T.astype(np.float64))[:, None]
        p_perm = perm_pvalue(obs, null)
        p_maxT = maxT_pvalue(obs, null)
    else:
        p_perm = p_maxT = np.full(L, np.nan)
    q = bh_fdr(p_perm)
    cog = centre_of_mass_depth(depth, out[primary] - (0.5 if jp["kind"] == "bin" else 0))
    exp_layer = expected_layer_tenney(depth, out[primary])
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


# ==================================================================================================
# Stage 1: Rodd et al. (2002) calibration (RQ3, RQ6)   [section: analyse_rodd_factorial]
# ==================================================================================================
# Stage 1: Rodd et al. (2002) calibration (spec §7 Stage 1, RQ3, RQ6).
#
# Three analyses, kept strictly separate (spec §12):
#
# 1. **Directional dissociation on a pre-registered model-derived proxy** (RQ6).
#    Primary proxy = representational settling cost (RSC), motivated by attractor accounts of
#    ambiguity effects (Rodd, Gaskell & Marslen-Wilson 2004, Cognitive Science; Armstrong & Plaut 2016,
#    Lang. Cogn. Neurosci.) and by the view of the residual stream as iterative refinement
#    (Jastrzebski et al., ICLR 2018). For word w with target state h_l at layer l:
#        z_l = (h_l - mean_l) / sd_l   (per layer, per dimension, over all words; label-free; removes the
#                                       rogue/massive dimensions that dominate raw cosine -- Timkey &
#                                       van Schijndel, EMNLP 2021)
#        d_l = 1 - cos(z_l, z_L)       ("distance from the final state" at depth l)
#        RSC = integral_0^1 d(depth) d depth   (trapezoid over relative depth)
#    Higher RSC = the representation reaches its final configuration later = MORE DIFFICULT.
#    Expected sign: senses(many) < 0 (facilitation), ambiguity(ambiguous) > 0 (inhibition).
#    Both factors are fitted in ONE model with effect coding (+-0.5) and their interaction.
#    Secondary proxies: logit-lens KL area, word surprisal, next-token entropy, (ease rating).
#    NONE of these is a reaction time.
#
# 2. **Time course** (novel, exploratory): the same factorial model on d_l at every layer tests the
#    semantic-settling-dynamics prediction (Armstrong & Plaut 2016) that the polysemy advantage
#    emerges earlier than the homonymy disadvantage: effect-weighted depth of the ambiguity effect minus
#    that of the senses effect (> 0 predicted), with a cell-stratified bootstrap CI.
#
# 3. **Representational geometry of the 2x2 design**
#    * factorial selectivity: out-of-fold probe scores for ambiguity and for senses regressed on both
#      factors (does the ambiguity read-out also respond to senses, and vice versa?);
#    * cross-condition generalisation performance, CCGP (Bernardi et al., Cell 2020): train an
#      ambiguity decoder on few-sense words and test on many-sense words (and the reverse), etc. High
#      CCGP means the factor is coded along a direction that is abstract w.r.t. the other factor.
#





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
        mu, sd = standardise_fit(h)
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
        est, _ = ols_hc3(Yz[:, j], X, names)
        for t_i, t in enumerate(names):
            if t == "intercept" or t in controls:
                continue
            bb = boots[:, t_i, j]
            rows.append({"appendix": appendix, "outcome": oc, "term": t, "n": int(ok.sum()), "design": design,
                         "controls": "+".join(controls) or "none", "beta_z": est[t]["beta"], "se_hc3": est[t]["se"],
                         "p_hc3": est[t]["p"], "ci_lo_hc3": est[t]["ci_lo"], "ci_hi_hc3": est[t]["ci_hi"],
                         "ci_lo_boot": np.nanpercentile(bb, 2.5), "ci_hi_boot": np.nanpercentile(bb, 97.5),
                         "p_boot": float(boot_p_two_sided(bb[:, None])[0])})
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
            mu, sd = standardise_fit(X[tr])
            Xs = (X - mu) / sd
            K = Xs @ Xs.T
            tr_idx = np.flatnonzero(tr)
            Yp = yf[perms[tr_lv]]                                   # permuted labels within condition
            Y = np.column_stack([yf[tr_idx], Yp])
            one = np.ones((len(tr_idx), 1))
            pred, *_ = fwl_kernel_ridge(K[np.ix_(tr, tr)], K[np.ix_(te, tr)], one, np.ones((te.sum(), 1)),
                                           Y, lambdas)
            y01 = (yf[te] > 0).astype(int)
            au = auroc(y01, pred.T)                               # (1+P,)
            obs.append(au[0])
            null.append(au[1:])
        o_m, n_m = np.mean(obs), np.mean(null, axis=0)
        rows.append({"factor": factor, "across": other, "layer": layer, "ccgp_auroc": o_m,
                     "ccgp_dir1": obs[0], "ccgp_dir2": obs[1], "null_mean": float(np.mean(n_m)),
                     "null_q95": float(np.quantile(n_m, 0.95)), "p_perm": float((1 + np.sum(n_m >= o_m)) / (1 + P))})
    df = pd.DataFrame(rows)
    df["q_bh_layers"] = bh_fdr(df.p_perm.to_numpy())
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
            est, _ = ols_hc3(_z(pred[layer]), X, ["intercept", "ambiguity", "senses", "ambiguity:senses"])
            for t in ("ambiguity", "senses", "ambiguity:senses"):
                rows.append({"probe": probe, "layer": layer, "term": t, "n": len(it), **{k: v for k, v in est[t].items()}})
    df = pd.DataFrame(rows)
    if len(df):
        df["q_bh_layers"] = df.groupby(["probe", "term"]).p.transform(lambda p: bh_fdr(p.to_numpy()))
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
            co["q_bh_proxies"] = co.groupby("term").p_boot.transform(lambda x: bh_fdr(x.to_numpy()))
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
        co["q_bh_layers"] = co.groupby("term").p_boot.transform(lambda x: bh_fdr(x.to_numpy()))
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
            s = factorial_selectivity(pred_dir, f"{slug(mid)}__{pooling}", es)
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
        co["q_bh_proxies"] = bh_fdr(co.p_boot.to_numpy())
        coef_rows.append(co)
    tag = lambda df: df.assign(model_id=mid, family=model_meta["family"], variant=model_meta["variant"],
                               size=model_meta["size"])
    res = {"coefs": tag(pd.concat(coef_rows, ignore_index=True)) if coef_rows else pd.DataFrame(),
           "verdicts": tag(pd.DataFrame(verdicts)),
           "timecourse": tag(pd.concat(tc_rows, ignore_index=True)) if tc_rows else pd.DataFrame(),
           "ccgp": tag(pd.concat(ccgp_rows, ignore_index=True)) if ccgp_rows else pd.DataFrame(),
           "selectivity": tag(pd.concat(sel_rows, ignore_index=True)) if sel_rows else pd.DataFrame()}
    for k, v in boots_store.items():
        np.savez_compressed(out_dir / f"rodd_boot__{slug(mid)}__{k}.npz", names=np.array(v["names"]),
                            outcomes=np.array(v["outcomes"]), boots=v["boots"], words=v["words"])
    return res


# ==================================================================================================
# Stage 2: WordNet-derived generalisation pool   [section: analyse_stage2]
# ==================================================================================================
# Stage 2: WordNet-derived generalisation pool (spec §7 Stage 2). NOT a replication of Rodd et al.
#
# Inferential model per representation-derived score (each proxy is a one-number-per-word summary, so
# no OLS on thousands of hidden dimensions is ever fitted):
#
#     z(proxy) ~ z(log1p wn_n_all) + z(GCIDE etymologies) + z(log_subtlex) + z(length) + z(concreteness)
#
# OLS with HC3 SEs, VIFs reported. For the higher-is-harder proxies the Rodd-compatible pattern is
# beta(sense count) < 0 and beta(etymologies) > 0. GCIDE etymology counts are a computed proxy for
# unrelated meanings, not a human norm; zero = "no etymology recorded" and is handled by policy.
#






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
            est, res = ols_hc3(su_z(d[oc]), Z, ["intercept"] + preds)
            vifs = vif(Z[:, 1:])
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
        df["q_bh_proxies"] = df.groupby(key).p_hc3.transform(lambda p: bh_fdr(p.to_numpy()))
    return df


def su_z(x) -> np.ndarray:
    x = np.asarray(x, float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * 0


# ==================================================================================================
# Base vs instruct paired comparisons (RQ5)   [section: compare_base_instruct]
# ==================================================================================================
# Base vs instruct comparisons within matched pairs only (spec §7, RQ5).
#
# Primary comparison = predictive-performance curves, not raw activations. Both members of a pair are
# evaluated on the same items with the same folds, so every bootstrap resample draws the SAME words
# for both models (paired word-level bootstrap). The paired difference is reported per layer (pairs
# share architecture, hence identical layer counts) and as a shift in the decodability centre of mass.
# Instruct inputs use the native chat template, so differences are "under each variant's native input
# format" and not a prompt-identical causal effect of instruction tuning.
#
# Exploratory: linear CKA (Kornblith et al., ICML 2019) between base and instruct target-word states
# at each layer.
#






def _metric(kind, y, pred):
    if kind == "reg":
        return r2(y, pred)
    return auroc((y > 0).astype(int), pred)


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
            bcog += [centre_of_mass_depth(depth, mi[:, k] - shift) - centre_of_mass_depth(depth, mb[:, k] - shift)
                     for k in range(mb.shape[1])]
        bd, bdd, bcog = np.concatenate(bd, 1).T, np.concatenate(bdd, 1).T, np.array(bcog)
        p = boot_p_two_sided(bd)
        q = bh_fdr(p)
        pdd = boot_p_two_sided(bdd)
        cog_b = centre_of_mass_depth(depth, m_b - shift)
        cog_i = centre_of_mass_depth(depth, m_i - shift)
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
            lambda x: bh_fdr(x.to_numpy()))
    return df


def cka_by_layer(acts_b: np.ndarray, acts_i: np.ndarray, valid: np.ndarray, base_id: str, inst_id: str,
                 pooling: str) -> pd.DataFrame:
    L = acts_b.shape[1]
    rows = [{"base_id": base_id, "instruct_id": inst_id, "pooling": pooling, "layer": l, "rel_depth": l / (L - 1),
             "n": int(valid.sum()),
             "linear_cka": linear_cka(acts_b[valid, l].astype(np.float64), acts_i[valid, l].astype(np.float64))}
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
                         "p_boot": float(boot_p_two_sided(d[:, None])[0])})
    return pd.DataFrame(rows)


# ==================================================================================================
# Publication figures (spec §8)   [section: make_figures]
# ==================================================================================================
# Publication figures (spec §8): vector PDF + SVG and 300-dpi PNG; plotted values exported to CSV.
#
# Visual system: one sans typeface, recessive axes/grid, 1.6-pt lines, fixed categorical order
# (blue, orange, aqua, yellow, ...) assigned by model size within family, line style = variant
# (solid base, dashed instruct). Every panel states the pool (published / derived) and its N.
#





SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_br", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a7a6", "#e66767", "#a32b2a"])
FAMILIES = ["smollm", "qwen", "llama"]
FAMILY_NAME = {"smollm": "SmolLM", "qwen": "Qwen", "llama": "Llama"}
POOL_NAME = {"published": "Published (Rodd et al., 2002)", "derived": "WordNet-derived"}
SIZE_ORDER = ["360M", "1B", "1.5B", "3B", "4B", "8B"]


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
        "ytick.color": MUTED, "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "svg.fonttype": "none", "pdf.fonttype": 42, "legend.frameon": False})


def save(fig, dirs: dict, name: str, data: pd.DataFrame | None = None):
    for ext in ("pdf", "svg"):
        fig.savefig(dirs["figures"] / f"{name}.{ext}", bbox_inches="tight")
    fig.savefig(dirs["figures_png"] / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    if data is not None and len(data):
        data.to_csv(dirs["figures"] / f"{name}_data.csv", index=False)


def _letter(ax, s):
    ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, va="bottom")


def _colour_map(models: pd.DataFrame) -> dict:
    cmap = {}
    for fam, g in models.groupby("family"):
        sizes = sorted(g["size"].unique(), key=lambda s: SIZE_ORDER.index(s) if s in SIZE_ORDER else 99)
        for _, r in g.iterrows():
            cmap[r.model_id] = SERIES[sizes.index(r["size"]) % len(SERIES)]
    return cmap


def _short(mid: str) -> str:
    return mid.split("/")[-1]


# ============================================================================ Figure 1
def fig_workflow(dirs):
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    w, gap = 17.0, 3.0
    top = ["CSV audit\n2,035 rows\n1,942 words", "Split pools\npublished A/B/C\nWordNet-derived",
           "Neutral prompt\nbase: 'Word: {w}'\ninstruct: chat tpl.", "Target-word states\nlayers 0..L\nlast & mean pooling",
           "Grouped-CV probes\nconditional ridge\nperm. + bootstrap"]
    bottom = ["Stage 1: Rodd B/C\n2x2 on settling\nproxy, CCGP", "Stage 2: derived\nsenses vs GCIDE\n(generalisation)",
              "Base vs instruct\npaired bootstrap\nCoG shift, CKA", "Robustness\npooling, controls,\nzeros, range",
              "Outputs\nfigures, tables,\nreport, manifest"]
    arrow = dict(arrowstyle="-|>", color=INK2, lw=0.8, shrinkA=0, shrinkB=0)
    for row, (y, labels) in enumerate(((29, top), (5, bottom))):
        for k, t in enumerate(labels):
            x = 1 + k * (w + gap)
            ax.add_patch(FancyBboxPatch((x, y), w, 11, boxstyle="round,pad=0.3,rounding_size=1.2",
                                        fc="#f0efec", ec=AXIS, lw=0.8))
            ax.text(x + w / 2, y + 5.5, t, ha="center", va="center", fontsize=6.4, color=INK, linespacing=1.35)
            if k < len(labels) - 1:
                ax.annotate("", xy=(x + w + gap - 0.4, y + 5.5), xytext=(x + w + 0.4, y + 5.5), arrowprops=arrow)
    ax.annotate("", xy=(1 + w / 2, 16.6), xytext=(1 + 4 * (w + gap) + w / 2, 28.4),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, connectionstyle="arc3,rad=-0.12"))
    ax.text(50, 44.0, "Appendix C is the auditory stimulus SET; every model receives it as text.",
            ha="center", fontsize=6.8, color=INK2, style="italic")
    ax.text(50, 0.6, "No labels, counts or construct words ever enter a prompt; labels are joined only after extraction.",
            ha="center", fontsize=6.8, color=INK2, style="italic")
    save(fig, dirs, "fig01_workflow")


# ============================================================================ layer-wise curves
def _curve_grid(metrics, target, control_set, metric, ylabel, dirs, name, title, eval_sets=("pub_all", "der_matched"),
                pooling="last"):
    m = metrics[(metrics.target == target) & (metrics.control_set == control_set) & (metrics.pooling == pooling)
                & metrics.eval_set.isin(eval_sets) & (metrics.status == "ok")]
    if m.empty:
        LOG.warning("figure %s: no data", name)
        return
    cmap = _colour_map(m[["model_id", "family", "size"]].drop_duplicates())
    fams = [f for f in FAMILIES if f in set(m.family)]
    fig, axes = plt.subplots(len(eval_sets), len(fams), figsize=(2.5 * len(fams) + 0.6, 2.2 * len(eval_sets)),
                             sharex=True, squeeze=False)
    lo, hi = (f"{metric}_ci_lo", f"{metric}_ci_hi")
    for r, es in enumerate(eval_sets):
        for c, fam in enumerate(fams):
            ax = axes[r, c]
            g = m[(m.eval_set == es) & (m.family == fam)]
            for mid, gm in g.groupby("model_id"):
                gm = gm.sort_values("rel_depth")
                ls = "--" if gm.variant.iloc[0] == "instruct" else "-"
                ax.plot(gm.rel_depth * 100, gm[metric], color=cmap[mid], ls=ls, label=_short(mid))
                if lo in gm:
                    ax.fill_between(gm.rel_depth * 100, gm[lo], gm[hi], color=cmap[mid], alpha=0.12, lw=0)
            n = int(g.n.max()) if len(g) else 0
            pool = POOL_NAME["published" if es.startswith("pub") else "derived"]
            ax.set_title(f"{FAMILY_NAME[fam]} | {pool}\n{es}, N = {n}", loc="left")
            ax.axhline(0.5 if metric == "auroc" else 0, color=AXIS, lw=0.8, zorder=0)
            if r == len(eval_sets) - 1:
                ax.set_xlabel("Relative depth (% of blocks; 0 = embeddings)")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="best", handlelength=2.2)
            if r == 0 and c == 0:
                _letter(ax, "a")
    fig.suptitle(title, x=0.01, ha="left", fontsize=9, color=INK)
    fig.supylabel(ylabel, fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, dirs, name, m)


def fig_heatmaps(metrics, dirs, targets, name, title, control_set="none"):
    m = metrics[metrics.target.isin(targets) & (metrics.control_set == control_set) & (metrics.pooling == "last")
                & metrics.eval_set.isin(["pub_all", "der_matched"]) & (metrics.status == "ok")]
    if m.empty:
        return
    grid = np.linspace(0, 1, 21)
    sets = [s for s in ["pub_all", "der_matched"] if s in set(m.eval_set)]
    fig, axes = plt.subplots(len(sets), len(targets), squeeze=False, layout="constrained",
                             figsize=(2.3 * len(targets) + 1.2, (0.24 * m.model_id.nunique() + 1.3) * len(sets) + 0.5))
    rows_out = []
    vmax = max(0.05, float(np.nanquantile(m.r2_lin, 0.99)))
    for r, es in enumerate(sets):
        for c, t in enumerate(targets):
            ax = axes[r, c]
            g = m[(m.eval_set == es) & (m.target == t)]
            models = sorted(g.model_id.unique(), key=lambda s: (g[g.model_id == s].family.iloc[0], s))
            mat = np.array([np.interp(grid, *g[g.model_id == mid].sort_values("rel_depth")[["rel_depth", "r2_lin"]].to_numpy().T)
                            for mid in models]) if models else np.zeros((0, len(grid)))
            for mid, row in zip(models, mat):
                rows_out += [{"eval_set": es, "target": t, "model_id": mid, "rel_depth": d, "r2_lin_interp": v}
                             for d, v in zip(grid, row)]
            if not models:
                ax.set_axis_off()
                continue
            im = ax.imshow(mat, aspect="auto", cmap=SEQ, vmin=0, vmax=vmax, extent=(0, 100, len(models), 0))
            ax.set_yticks(np.arange(len(models)) + 0.5)
            ax.set_yticklabels([_short(x) for x in models] if c == 0 else [])
            ax.grid(False)
            n = int(g.n.max()) if len(g) else 0
            ax.set_title(f"{t}\n{'published' if es.startswith('pub') else 'derived'}, N = {n}", loc="left", fontsize=7.5)
            if r == len(sets) - 1:
                ax.set_xlabel("Relative depth (%)")
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01)
    cb.set_label("Out-of-fold R$^2$\n(interpolated to a common depth grid)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=9)
    save(fig, dirs, name, pd.DataFrame(rows_out))


# ============================================================================ Figure 4
def fig_rodd(rodd_coefs, rodd_tc, dirs, primary):
    if rodd_coefs is None or rodd_coefs.empty:
        return
    c = rodd_coefs[(rodd_coefs.outcome == primary) & (rodd_coefs.control_set == "primary")
                   & rodd_coefs.term.isin(["ambiguity", "senses"]) & rodd_coefs.appendix.isin(["B", "C"])]
    if c.empty:
        return
    models = sorted(c.model_id.unique(), key=lambda s: (c[c.model_id == s].family.iloc[0], s))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.28 * len(models) + 1.6), sharey=True, squeeze=False)
    col = {"ambiguity": SERIES[1], "senses": SERIES[0]}
    off = {"ambiguity": 0.17, "senses": -0.17}
    for k, app in enumerate(["B", "C"]):
        ax = axes[0, k]
        for t in ("ambiguity", "senses"):
            g = c[(c.appendix == app) & (c.term == t)].set_index("model_id").reindex(models)
            y = np.arange(len(models)) + off[t]
            ax.errorbar(g.beta_z, y, xerr=[g.beta_z - g.ci_lo_boot, g.ci_hi_boot - g.beta_z], fmt="o", ms=4,
                        color=col[t], ecolor=col[t], elinewidth=1.2, capsize=0,
                        label={"ambiguity": "Ambiguity (ambiguous - unambiguous)", "senses": "Senses (many - few)"}[t])
        ax.axvline(0, color=AXIS, lw=0.8)
        n = int(c[c.appendix == app].n.max())
        lab = "Appendix B: visual stimulus set" if app == "B" else "Appendix C: auditory stimulus set (text input)"
        ax.set_title(f"{lab}\nN = {n}, published pool", loc="left")
        ax.set_xlabel("Effect on RSC (SD units)")
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels([_short(m) for m in models])
        ax.invert_yaxis()
        _letter(ax, "ab"[k])
    fig.tight_layout()
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    fig.text(0.01, -0.22, "Points: z-scored effect; bars: 95% cell-stratified bootstrap CI. Orientation: higher RSC = slower "
             "representational settling (harder).\nPredicted: senses < 0 (facilitation), ambiguity > 0 (inhibition). "
             "Model-derived proxy, not a reaction time.",
             fontsize=6.8, color=INK2)
    save(fig, dirs, "fig04_rodd_factorial", c)
    if rodd_tc is None or rodd_tc.empty:
        return
    tc = rodd_tc[rodd_tc.term.isin(["ambiguity", "senses"])]
    models = sorted(tc.model_id.unique())
    ncol = min(4, len(models))
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol + 0.6, 2.0 * nrow + 0.8), squeeze=False, sharex=True,
                             layout="constrained")
    for i, mid in enumerate(models):
        ax = axes[i // ncol, i % ncol]
        for t in ("ambiguity", "senses"):
            for app, ls in (("B", "-"), ("C", "--")):
                g = tc[(tc.model_id == mid) & (tc.term == t) & (tc.appendix == app)].sort_values("rel_depth")
                ax.plot(g.rel_depth * 100, g.beta_z, color=col[t], ls=ls, lw=1.3)
                ax.fill_between(g.rel_depth * 100, g.ci_lo_boot, g.ci_hi_boot, color=col[t], alpha=0.08, lw=0)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(_short(mid), loc="left")
    for j in range(len(models), nrow * ncol):
        axes[j // ncol, j % ncol].set_axis_off()
    handles = [Line2D([], [], color=col["ambiguity"], label="ambiguity"), Line2D([], [], color=col["senses"], label="senses"),
               Line2D([], [], color=INK2, ls="-", label="Appendix B"), Line2D([], [], color=INK2, ls="--", label="Appendix C (text)")]
    for a in axes[-1]:
        a.set_xlabel("Relative depth (%)")
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    fig.supylabel("Effect on d$_l$ = 1 - cos(z$_l$, z$_L$) (SD)", fontsize=8, color=INK2)
    save(fig, dirs, "supp_fig_rodd_timecourse", tc)


# ============================================================================ Figure 6
def fig_pairs(pairs_df, dirs, target, eval_set, control_set, name, title):
    if pairs_df is None or pairs_df.empty:
        return
    p = pairs_df[(pairs_df.target == target) & (pairs_df.eval_set == eval_set) & (pairs_df.control_set == control_set)
                 & (pairs_df.pooling == "last")]
    if p.empty:
        return
    prs = p[["base_id", "instruct_id"]].drop_duplicates().to_numpy()
    ncol = min(3, len(prs))
    nrow = int(np.ceil(len(prs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.1 * nrow + 0.5), squeeze=False, sharex=True)
    for i, (b, ins) in enumerate(prs):
        ax = axes[i // ncol, i % ncol]
        g = p[(p.base_id == b) & (p.instruct_id == ins)].sort_values("rel_depth")
        ax.plot(g.rel_depth * 100, g.base, color=SERIES[0], label="base")
        ax.plot(g.rel_depth * 100, g.instruct, color=SERIES[1], ls="--", label="instruct")
        sig = g[g.q_bh_layers < 0.05]
        y0 = np.nanmin(np.r_[g.base, g.instruct])
        ax.scatter(sig.rel_depth * 100, np.full(len(sig), y0), marker="|", s=30, color=INK2,
                   label="paired diff. q < .05" if i == 0 else None)
        ax.set_title(f"{_short(b)} vs -Instruct\nN = {int(g.n.max())}, {g.metric.iloc[0]}", loc="left")
        if i == 0:
            ax.legend(loc="best")
    for j in range(len(prs), nrow * ncol):
        axes[j // ncol, j % ncol].set_axis_off()
    fig.supxlabel("Relative depth (%)", fontsize=8, color=INK2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    save(fig, dirs, name, p)


# ============================================================================ Figure 7
def fig_robustness(metrics, stage2, dirs, primary):
    panels = []
    if stage2 is not None and len(stage2):
        s = stage2[(stage2.outcome == primary) & stage2.role.isin(["sense_count", "meanings_proxy"])]
        panels.append(("Stage 2 regression of RSC (derived pool)", s, "analysis_variant", "role", "beta_z", "ci_lo", "ci_hi",
                       "Standardised beta (SD), 95% HC3 CI"))
    if metrics is not None and len(metrics):
        m = metrics[(metrics.status == "ok") & metrics.target.isin(["log1p_wn_n_all", "log_subtlex"])
                    & metrics.eval_set.isin(["der_matched", "der_full", "pub_all"])]
        if len(m):
            best = m.loc[m.groupby(["model_id", "pooling", "eval_set", "target", "control_set"]).r2_lin.idxmax()]
            best = best.assign(variant_label=best.eval_set + " | " + best.control_set + " | " + best.pooling)
            panels.append(("Peak out-of-fold R$^2$ across layers", best, "variant_label", "target", "r2_lin",
                           "r2_lin_ci_lo", "r2_lin_ci_hi", "Peak R$^2$ (95% word-level bootstrap CI)"))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6), squeeze=False)
    out = []
    for k, (title, d, ycol, ccol, x, lo, hi, xlabel) in enumerate(panels):
        ax = axes[0, k]
        labels = sorted(d[ycol].unique())
        cats = sorted(d[ccol].unique())
        for ci_, cat in enumerate(cats):
            g = d[d[ccol] == cat]
            jit = (ci_ - (len(cats) - 1) / 2) * 0.22
            y = np.array([labels.index(v) for v in g[ycol]]) + jit
            ax.errorbar(g[x], y, xerr=[g[x] - g[lo], g[hi] - g[x]], fmt="o", ms=3, color=SERIES[ci_],
                        ecolor=SERIES[ci_], elinewidth=0.8, alpha=0.8, capsize=0, label=cat)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=len(cats))
        ax.invert_yaxis()
        _letter(ax, "ab"[k])
        out.append(d.assign(panel=title))
    fig.tight_layout()
    save(fig, dirs, "fig07_robustness", pd.concat(out, ignore_index=True))


# ============================================================================ Figure 8
def fig_audit(df, dirs, cfg):
    cols = ["log_subtlex", "length", "concreteness", "ortho_n", "n_syll", "n_phon", "wn_n_all", "gcide_etymologies",
            "sim_top2", "min_sim", "mean_sim"]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.4), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0, 0]
    d = df[df.provenance == "wordnet-derived"].drop_duplicates("word_lc")[cols].corr(method="spearman")
    p = df[df.provenance == "published"].drop_duplicates("word_lc")[cols].corr(method="spearman")
    mat = np.triu(d.to_numpy(), 1) + np.tril(p.to_numpy(), -1)
    im = ax.imshow(mat, cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.grid(False)
    ax.set_title("Spearman correlations\nupper: derived (N=1,633)\nlower: published (N=309)", loc="left")
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.03)
    cb.set_label("Spearman rho", color=INK2)
    _letter(ax, "a")
    ax = axes[0, 1]
    design = {"word", "provenance", "source_set", "pool_condition", "appendix", "experiment", "modality",
              "ambiguity", "senses"}            # missing by design in one pool: not data loss
    miss = df.groupby("provenance")[[c for c in REQUIRED_COLS if c not in design]].apply(lambda g: g.isna().mean() * 100).T
    miss = miss[(miss > 0).any(axis=1)]
    yy = np.arange(len(miss))
    for k, prov in enumerate(miss.columns):
        ax.barh(yy + (k - 0.5) * 0.38, miss[prov], height=0.36, color=SERIES[k], label=prov)
    ax.set_yticks(yy)
    ax.set_yticklabels(miss.index)
    ax.invert_yaxis()
    ax.set_xlabel("% missing")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title("Missingness of measured variables\n(design labels excluded: missing by construction)", loc="left")
    _letter(ax, "b")
    ax = axes[1, 0]
    pub = df[(df.provenance == "published") & df.senses.notna()]
    cells = pub.groupby(["appendix", "ambiguity", "senses"]).size().reset_index(name="n")
    labels = cells.appendix + " | " + cells.ambiguity + " | " + cells.senses
    ax.barh(range(len(cells)), cells.n, color=[SERIES[0] if a == "B" else SERIES[2] for a in cells.appendix], height=0.6)
    for i, v in enumerate(cells.n):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Rows")
    ax.set_title("Published 2x2 cells (B visual; C auditory set, text input)", loc="left")
    _letter(ax, "c")
    ax = axes[1, 1]
    attr = stage2_attrition(df, cfg)
    ax.barh(range(len(attr)), attr.n_unique_words, color=SERIES[0], height=0.6)
    for i, v in enumerate(attr.n_unique_words):
        ax.text(v + 10, i, str(v), va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(attr)))
    short = ["WordNet-derived pool", "length in range", "log_subtlex in range", "GCIDE observed",
             "GCIDE > 0 (primary set)", "+ concreteness observed"]
    ax.set_yticklabels(short[:len(attr)])
    ax.invert_yaxis()
    ax.set_xlabel("Unique words")
    ax.set_title("Stage 2 filter attrition (derived pool)", loc="left")
    _letter(ax, "d")
    fig.tight_layout()
    save(fig, dirs, "fig08_data_audit", attr)


def fig_ccgp(ccgp, dirs):
    if ccgp is None or ccgp.empty:
        return
    c = ccgp[ccgp.pooling == "last"]
    models = sorted(c.model_id.unique())
    ncol = min(4, len(models))
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol + 0.6, 2.0 * nrow + 0.8), squeeze=False, sharex=True,
                             sharey=True, layout="constrained")
    col = {"ambiguity": SERIES[1], "senses": SERIES[0]}
    for i, mid in enumerate(models):
        ax = axes[i // ncol, i % ncol]
        for f in ("ambiguity", "senses"):
            for app, ls in (("B", "-"), ("C", "--")):
                g = c[(c.model_id == mid) & (c.factor == f) & (c.appendix == app)].sort_values("rel_depth")
                ax.plot(g.rel_depth * 100, g.ccgp_auroc, color=col[f], ls=ls, lw=1.3)
                ax.plot(g.rel_depth * 100, g.null_q95, color=col[f], ls=":", lw=0.8)
        ax.axhline(0.5, color=AXIS, lw=0.8)
        ax.set_title(_short(mid), loc="left")
    for j in range(len(models), nrow * ncol):
        axes[j // ncol, j % ncol].set_axis_off()
    handles = [Line2D([], [], color=col["ambiguity"], label="ambiguity across senses"),
               Line2D([], [], color=col["senses"], label="senses across ambiguity"),
               Line2D([], [], color=INK2, ls="--", label="Appendix C (text)"),
               Line2D([], [], color=INK2, ls=":", label="95th pct. permutation null")]
    for a in axes[-1]:
        a.set_xlabel("Relative depth (%)")
    fig.legend(handles=handles, loc="outside lower center", ncol=2)
    fig.supylabel("Cross-condition generalisation AUROC", fontsize=8, color=INK2)
    save(fig, dirs, "supp_fig_ccgp", c)


def make_all(dirs, df, eval_sets, metrics, rodd, stage2, pairs_df, cfg):
    style()
    fig_workflow(dirs)
    fig_audit(df, dirs, cfg)
    if metrics is None or metrics.empty:
        return
    _curve_grid(metrics, "log_subtlex", "none", "r2_lin", "Out-of-fold R$^2$ (log SUBTLEX frequency)", dirs,
                "fig02_frequency_layerwise", "Frequency information by layer (ridge probe, grouped 5-fold CV, 95% bootstrap CI)")
    _curve_grid(metrics, "log1p_wn_n_all", "none", "r2_lin", "Out-of-fold R$^2$ (log1p WordNet senses)", dirs,
                "fig03_sense_count_layerwise", "Sense-count information by layer")
    _curve_grid(metrics, "log_subtlex", "primary", "delta_r2_lin", "$\\Delta$R$^2$ beyond sense count, length, concreteness",
                dirs, "supp_fig_frequency_conditional", "Frequency information beyond covariates (conditional probe)")
    _curve_grid(metrics, "log1p_wn_n_all", "primary", "delta_r2_lin", "$\\Delta$R$^2$ beyond frequency, length, concreteness",
                dirs, "supp_fig_sense_count_conditional", "Sense-count information beyond covariates (conditional probe)")
    _curve_grid(metrics, "wn_n_all", "none", "r2_lin", "Out-of-fold R$^2$ (raw WordNet sense count)", dirs,
                "supp_fig_sense_count_raw", "Raw sense count (sensitivity)")
    _curve_grid(metrics, "ambiguity_bin", "none", "auroc", "AUROC (ambiguous vs unambiguous)", dirs,
                "supp_fig_rodd_ambiguity_probe", "Published ambiguity factor by layer", eval_sets=("pub_all", "pub_B", "pub_C"))
    _curve_grid(metrics, "senses_bin", "none", "auroc", "AUROC (many vs few senses)", dirs,
                "supp_fig_rodd_senses_probe", "Published senses factor by layer", eval_sets=("pub_B", "pub_C"))
    fig_heatmaps(metrics, dirs, ["gcide_nonzero", "sim_top2", "min_sim", "mean_sim"], "fig05_meaning_proxy_distance",
                 "GCIDE meanings proxy (computed) and WordNet semantic distance: layer-wise decodability")
    fig_rodd(rodd.get("coefs"), rodd.get("timecourse"), dirs, cfg["proxies"]["primary"])
    fig_ccgp(rodd.get("ccgp"), dirs)
    fig_pairs(pairs_df, dirs, "log1p_wn_n_all", "der_matched", "none", "fig06_base_vs_instruct",
              "Base vs instruct: sense-count decodability (derived matched pool; native input formats)")
    fig_pairs(pairs_df, dirs, "log_subtlex", "der_matched", "none", "supp_fig_pairs_frequency",
              "Base vs instruct: frequency decodability")
    fig_pairs(pairs_df, dirs, "ambiguity_bin", "pub_all", "none", "supp_fig_pairs_ambiguity",
              "Base vs instruct: published ambiguity factor")
    fig_robustness(metrics, stage2, dirs, cfg["proxies"]["primary"])


# ==================================================================================================
# Publication tables (spec §9)   [section: make_tables]
# ==================================================================================================
# Publication tables (spec §9): CSV + Markdown + LaTeX; every table carries N and the pool.





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


# ==================================================================================================
# Markdown results report (spec §12)   [section: make_report]
# ==================================================================================================
# Markdown results report (spec §12). Every number is read from the analysis outputs; if an output
# is missing, the section says so instead of inventing a result.





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



# ==================================================================================================
# Pipeline stages and command-line interface
# ==================================================================================================
STAGES = ["audit", "extract", "analyse", "compare", "report", "all"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML config (default: the embedded DEFAULT_CONFIG_YAML)")
    p.add_argument("--data", default=None, help="path to ambiguity_items_combined.csv (overrides config)")
    p.add_argument("--models", nargs="*", default=None, help="subset of configured model ids")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--dry-run", action="store_true", help="10 words, one small model, reduced resampling")
    p.add_argument("--dry-run-n", type=int, default=None, help="override the dry-run word count")
    p.add_argument("--resume", action="store_true", help="skip validated shards / finished analyses")
    p.add_argument("--device", default=None, help="cuda, cuda:0, mps or cpu (default: auto)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--self-test", action="store_true", help="run the built-in offline test-suite and exit")
    p.add_argument("--write-config", default=None, metavar="PATH", help="write the embedded config and exit")
    return p.parse_args(argv)


def resolve_data_path(cfg: dict, override: str | None = None) -> Path:
    p = Path(override or cfg["data"]["csv"])
    if p.is_absolute() or p.exists():
        return p.resolve()
    if (ROOT / p).exists():
        return ROOT / p
    raise FileNotFoundError(f"dataset not found: {p} (looked in the working directory and {ROOT}); use --data")


def auto_device(requested):
    if requested:
        return requested
    try:
        import torch as _t
        if _t.cuda.is_available():
            return "cuda"
        if getattr(_t.backends, "mps", None) and _t.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def apply_dry_run(cfg, n_override):
    dr = cfg["dry_run"]
    cfg["probes"].update({"n_folds": 2, "n_repeats": 1, "n_permutations": 20, "n_bootstrap": 50, "min_n": 6})
    cfg["rodd"].update({"n_bootstrap": 100, "n_permutations": 20})
    cfg["extraction"]["shard_size"] = 4          # exercises multi-shard resume logic
    cfg["extraction"]["alignment_audit_n"] = 5
    dr["n_words"] = n_override or dr["n_words"]
    return cfg


def dry_run_words(words: pd.DataFrame, df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Deterministic, design-covering sample: round-robin over published cells and the derived pool."""
    first = df.drop_duplicates("word_lc")
    strata = [first[(first.appendix == "A") & (first.ambiguity == a)] for a in ("ambiguous", "unambiguous")]
    for app in ("B", "C"):
        for a in ("ambiguous", "unambiguous"):
            for s in ("few", "many"):
                strata.append(first[(first.appendix == app) & (first.ambiguity == a) & (first.senses == s)])
    strata.append(first[first.provenance == "wordnet-derived"])
    rng = np.random.default_rng(seed)
    pools = [list(rng.permutation(s.word_lc.to_numpy())) for s in strata]
    picked: list[str] = []
    while len(picked) < n and any(pools):
        for pool in pools:
            while pool and pool[0] in picked:
                pool.pop(0)
            if pool and len(picked) < n:
                picked.append(pool.pop(0))
    return words[words.word_lc.isin(picked)].sort_values("word_idx").reset_index(drop=True)


def restrict_eval_sets(eval_sets: dict[str, EvalSet], word_idx: set[int]) -> dict[str, EvalSet]:
    out = {}
    for k, es in eval_sets.items():
        keep = es.items.word_idx.isin(word_idx).to_numpy()
        out[k] = EvalSet(es.name, es.pool, es.description, es.items[keep].reset_index(drop=True), es.strat[keep], es.notes)
    return out


def stage_audit(cfg, dirs, data_path):
    df = load_and_validate(data_path, cfg["data"])
    eval_sets = build_eval_sets(df, cfg)
    key = write_audit(df, cfg, dirs, eval_sets)
    write_json(dirs["audit"] / "audit_key_counts.json", key)
    return df, eval_sets, key


def stage_extract(cfg, dirs, specs, words, device, resume):
    status = {}
    for spec in specs:
        try:
            d = extract_model(spec, words, cfg, dirs["activations"], device, resume)
            status[spec.id] = {"status": "ok", "dir": str(d)}
            if cfg["ease_rating"]["enabled"] and spec.variant == "instruct":
                tok, model, _ = load_model(spec, cfg["extraction"]["revision"], cfg["extraction"]["dtype"], device)
                rate_words(tok, model, words, cfg, device).to_parquet(d / "ease_rating.parquet", index=False)
                del model
        except Exception as err:  # never silently skip: record and continue with the next model
            LOG.error("extraction failed for %s: %s", spec.id, err)
            status[spec.id] = {"status": "failed", "error": str(err)}
    write_json(dirs["activations"] / "extraction_status.json", status)
    return status


def stage_analyse(cfg, dirs, specs, eval_sets, resume):
    for spec in specs:
        tag = slug(spec.id)
        done = dirs["metrics"] / f"probe_metrics__{tag}.parquet"
        if resume and done.exists():
            LOG.info("%s: analysis exists, skipping (--resume)", spec.id)
            continue
        try:
            d = model_dir(dirs["activations"], spec.id)
        except FileNotFoundError as err:
            LOG.warning("%s", err)
            continue
        acts, wmeta, mmeta = {}, None, None
        for pooling in cfg["extraction"]["poolings"]:
            acts[pooling], wmeta, mmeta = load_activations(d, pooling)
        mmeta = {**mmeta, "size": spec.size, "family": spec.family, "variant": spec.variant, "paired_base": spec.pair}
        es = restrict_eval_sets(eval_sets, set(wmeta.word_idx.astype(int)))
        jobs = build_jobs(cfg, es)
        frames = [run_probes(a, wmeta, mmeta, es, jobs, cfg, pooling, dirs["predictions"]) for pooling, a in acts.items()]
        metrics = pd.concat(frames, ignore_index=True)
        prox, d_last = compute_proxies(acts["last"], acts.get("mean"), wmeta)
        prox.to_parquet(dirs["metrics"] / f"proxies__{tag}.parquet", index=False)
        ease = pd.read_parquet(d / "ease_rating.parquet") if (d / "ease_rating.parquet").exists() else None
        rodd = run_stage1(mmeta, prox, d_last, acts, wmeta, es, cfg, dirs["predictions"], dirs["metrics"], ease)
        for k, v in rodd.items():
            v.to_parquet(dirs["metrics"] / f"rodd_{k}__{tag}.parquet", index=False)
        run_stage2(mmeta, prox, es, cfg, ease).to_parquet(dirs["metrics"] / f"stage2__{tag}.parquet", index=False)
        write_json(dirs["metrics"] / f"model_meta__{tag}.json", mmeta)
        metrics.to_parquet(done, index=False)          # written last = completion marker
        LOG.info("%s: analysis complete (%d metric rows)", spec.id, len(metrics))


def _collect(dirs, prefix):
    frames = [pd.read_parquet(f) for f in sorted(dirs["metrics"].glob(f"{prefix}__*.parquet"))]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def stage_compare(cfg, dirs, specs):
    metrics = _collect(dirs, "probe_metrics")
    rodd_coefs = _collect(dirs, "rodd_coefs")
    ids = {s.id for s in specs}
    out, cka, rp = [], [], []
    for base, inst in pairs(registry(cfg)):
        if not ({base.id, inst.id} <= ids) or metrics.empty or not ({base.id, inst.id} <= set(metrics.model_id)):
            continue
        LOG.info("comparing %s vs %s", base.id, inst.id)
        out.append(paired_curves(base.id, inst.id, metrics, dirs["predictions"], cfg))
        if len(rodd_coefs):
            rp.append(paired_rodd(base.id, inst.id, dirs["metrics"], rodd_coefs, cfg["proxies"]["primary"]))
        for pooling in cfg["extraction"]["poolings"]:
            ab, mb, _ = load_activations(model_dir(dirs["activations"], base.id), pooling)
            ai, mi, _ = load_activations(model_dir(dirs["activations"], inst.id), pooling)
            if not np.array_equal(mb.word_idx.to_numpy(), mi.word_idx.to_numpy()):
                LOG.warning("CKA skipped: word sets differ for %s / %s", base.id, inst.id)
                continue
            valid = ((mb.status == "ok") & (mi.status == "ok")).to_numpy()
            cka.append(cka_by_layer(ab, ai, valid, base.id, inst.id, pooling))

    def cat(xs):
        xs = [x for x in xs if len(x)]
        return pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()

    pairs_df, cka_df, rp_df = cat(out), cat(cka), cat(rp)
    pairs_df.to_parquet(dirs["metrics"] / "pairs_curves.parquet", index=False)
    cka_df.to_parquet(dirs["metrics"] / "pairs_cka.parquet", index=False)
    rp_df.to_parquet(dirs["metrics"] / "pairs_rodd.parquet", index=False)
    return pairs_df, cka_df, rp_df


def stage_report(cfg, dirs, df, eval_sets, audit_key, dry_run):
    metrics = _collect(dirs, "probe_metrics")
    rodd = {k: _collect(dirs, f"rodd_{k}") for k in ("coefs", "verdicts", "timecourse", "ccgp", "selectivity")}
    stage2 = _collect(dirs, "stage2")
    rd = lambda n: pd.read_parquet(dirs["metrics"] / n) if (dirs["metrics"] / n).exists() else pd.DataFrame()
    pairs_df, cka_df, rp_df = rd("pairs_curves.parquet"), rd("pairs_cka.parquet"), rd("pairs_rodd.parquet")
    metas = [json.loads(p.read_text()) for p in sorted(dirs["metrics"].glob("model_meta__*.json"))]
    solo = {s.id for s in base_only(registry(cfg))}
    T = dirs["tables"]
    table1(audit_key, stage2_attrition(df, cfg), eval_sets, T)
    table2(metas, solo, T)
    table3(df, T)
    miss = pd.read_csv(dirs["audit"] / "audit_missingness.csv")
    if len(metrics):
        table4(metrics, T)
        table8(metrics, T)
        tableS(metrics, T, miss)
    table5(rodd["coefs"], rodd["verdicts"], T)
    table6(stage2, T)
    table7(pairs_df, T)
    if len(rp_df):
        write_table(rp_df, T / "table7b_base_instruct_rodd", "Table 7b. Instruct - base difference in Rodd proxy effects")
    if len(cka_df):
        write_table(cka_df, T / "tableS3_cka", "Table S3. Exploratory linear CKA, base vs instruct, per layer")
    for k in ("ccgp", "selectivity", "timecourse"):
        if len(rodd[k]):
            rodd[k].to_csv(T / f"tableS_rodd_{k}.csv", index=False)
    make_all(dirs, df, eval_sets, metrics, rodd, stage2, pairs_df, cfg)
    return write_report(dirs, audit_key, metrics, rodd, stage2, pairs_df, cka_df, metas, cfg, dry_run)


def write_manifest(cfg, dirs, args, t0, data_path):
    import importlib.metadata as im
    pk = {}
    for name in ("torch", "transformers", "tokenizers", "numpy", "pandas", "scipy", "scikit-learn", "statsmodels",
                 "matplotlib", "pyarrow", "pyyaml"):
        try:
            pk[name] = im.version(name)
        except im.PackageNotFoundError:
            pk[name] = None
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = None
    outputs = {str(p.relative_to(dirs["root"])): sha256_file(p) for p in sorted(dirs["root"].rglob("*"))
               if p.is_file() and "activations" not in p.parts and p.name != "run_manifest.json"}
    acts = {str(p.relative_to(dirs["root"])): json.loads(p.read_text()).get("meta", {}).get("model_commit")
            for p in dirs["activations"].glob("*/manifest.json")}
    write_json(dirs["root"] / "run_manifest.json", {
        "started": t0, "finished": dt.datetime.now(dt.timezone.utc).isoformat(), "argv": sys.argv,
        "args": vars(args), "script": Path(__file__).name, "script_version": __version__,
        "script_sha256": sha256_file(Path(__file__)), "git_commit": commit, "python": platform.python_version(),
        "platform": platform.platform(), "packages": pk, "config": cfg, "data_path": str(data_path),
        "data_sha256": sha256_file(data_path), "activation_manifests": acts, "output_sha256": outputs})


# ==================================================================================================
# Built-in self-test (offline): counts, duplicates, alignment, leakage, folds, estimator exactness,
# planted-signal recovery, conditional probing, Stage 1/2 + outputs, extraction mechanics, full CLI.
# Weights in the extraction checks are RANDOM: they validate code paths, never scientific results.
# ==================================================================================================
def _tiny_tokenizer(df):
    """Offline byte-level BPE tokenizer trained on the stimulus words (multi-subtoken words guaranteed)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["<s>", "<pad>", "<|im_start|>", "<|im_end|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator([f"Word: {w}" for w in df.word_lc.unique()] + ["user assistant system"], trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", pad_token="<pad>", eos_token="<|im_end|>")
    fast.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                          "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    return fast


def _synthetic_acts(eval_sets, n_layers=6, d=48, seed=0, signal_layer=3, target="log_subtlex"):
    """States whose layer `signal_layer` linearly encodes `target` (+ noise); other layers are noise."""
    items = pd.concat([es.items for es in eval_sets.values()]).drop_duplicates("word_idx").sort_values("word_idx")
    rng = np.random.default_rng(seed)
    n = len(items)
    acts = rng.normal(size=(n, n_layers, d)).astype(np.float32)
    y = items[target].fillna(items[target].mean()).to_numpy()
    y = (y - y.mean()) / y.std()
    w = rng.normal(size=d)
    acts[:, signal_layer] += (np.outer(y, w) / np.linalg.norm(w) * 3).astype(np.float32)
    meta = pd.DataFrame({"word_idx": items.word_idx.to_numpy(), "word": items.word_lc.to_numpy(), "status": "ok",
                         "n_subtokens": rng.integers(1, 4, n), "surprisal_mean": rng.normal(size=n),
                         "next_entropy": rng.normal(size=n)})
    return acts, meta


def self_test(data_path: Path) -> int:
    import shutil
    tmp = Path(tempfile.mkdtemp(prefix="lexamb_selftest_"))
    setup_logging(tmp / "logs", "selftest")
    logging.getLogger("lexamb").setLevel(logging.WARNING)
    cfg = load_config()
    cfg["probes"].update({"n_folds": 3, "n_repeats": 2, "n_permutations": 30, "n_bootstrap": 60, "min_n": 10})
    cfg["rodd"].update({"n_bootstrap": 100, "n_permutations": 20})
    df = load_and_validate(data_path, cfg["data"])
    eval_sets = build_eval_sets(df, cfg)
    meta0 = {"model_id": "test/model", "family": "smollm", "variant": "base", "size": "360M", "paired_base": None}
    results = []

    def check(name, fn):
        t0 = time.time()
        try:
            fn()
            results.append((name, "PASS", ""))
        except ImportError as err:
            results.append((name, "SKIP", str(err)))
        except Exception as err:  # report every failure, keep going
            results.append((name, "FAIL", f"{type(err).__name__}: {err}"))
        print(f"  [{results[-1][1]}] {name} ({time.time() - t0:.1f}s) {results[-1][2]}", flush=True)

    def t_counts():
        assert df.shape[0] == 2035
        assert df.provenance.value_counts().to_dict() == {"wordnet-derived": 1633, "published": 402}
        pub = df[df.provenance == "published"]
        assert pub.word_lc.nunique() == 309
        assert pub[pub.senses.notna()].appendix.value_counts().to_dict() == {"B": 128, "C": 92}
        att = stage2_attrition(df, cfg).set_index("step")
        assert att.loc[2, "n_unique_words"] == 646 and att.loc[3, "n_unique_words"] == 596

    def t_dedup():
        for es in eval_sets.values():
            assert es.items.word_lc.is_unique and len(es.strat) == len(es.items), es.name
        assert len(eval_sets["pub_all"].items) == 309
        assert set(eval_sets["pub_all"].items.provenance) == {"published"}
        assert set(eval_sets["der_full"].items.provenance) == {"wordnet-derived"}
        assert (eval_sets["pub_B"].items.groupby(["ambiguity", "senses"]).size() == 32).all()

    def t_fail_loudly():
        try:
            load_and_validate(data_path, dict(cfg["data"], expected_rows=9999))
        except DataValidationError:
            return
        raise AssertionError("validation did not fail on a wrong expected row count")

    tok_box = {}

    def t_alignment():
        tok_box["tok"] = tok = _tiny_tokenizer(df)
        for variant in ("base", "instruct"):
            fails, multi = [], 0
            for w in df.word_lc.unique():
                r = render(tok, w, variant, cfg["extraction"])
                if not r.ok:
                    fails.append((w, r.reason))
                    continue
                assert tok.decode([r.input_ids[k] for k in r.target_idx]).strip() == w
                assert r.target_idx == list(range(r.target_idx[0], r.target_idx[-1] + 1))
                multi += len(r.target_idx) > 1
            assert not fails, fails[:5]
            assert multi > 0

    def t_leakage():
        tok = tok_box.get("tok") or _tiny_tokenizer(df)
        t = add_derived_targets(df)
        forbidden = ["ambiguous", "unambiguous", "homonym", "polyseme", "senses", "meaning", "few", "many", "etymolog"]
        for _, row in t.sample(200, random_state=0).iterrows():
            w = row.word_lc
            rb, ri = render(tok, w, "base", cfg["extraction"]), render(tok, w, "instruct", cfg["extraction"])
            assert rb.text == f"Word: {w}"
            assert ri.text == f"<|im_start|>user\nWord: {w}<|im_end|>\n"
            for f in forbidden:
                if f not in w:
                    assert f not in rb.text.lower() and f not in ri.text.lower()

    def t_folds():
        es = eval_sets["pub_B"]
        f1, f2 = make_folds(es, 5, 3, 7), make_folds(es, 5, 3, 7)
        assert np.array_equal(f1, f2)
        cells = es.items.ambiguity + es.items.senses
        for r in range(3):
            assert pd.crosstab(f1[r], cells).min().min() >= 1
        assert not np.array_equal(f1[0], f1[1])

    def t_kernel_ridge():
        rng = np.random.default_rng(1)
        n_t, n_e, d = 50, 15, 120
        X = rng.normal(size=(n_t + n_e, d))
        C = np.column_stack([np.ones(n_t + n_e), rng.normal(size=(n_t + n_e, 2))])
        y = X[:, 0] + 2 * C[:, 1] + rng.normal(size=n_t + n_e)
        T, E = slice(0, n_t), slice(n_t, None)
        K = X @ X.T
        pred, loo, _, _ = fwl_kernel_ridge(K[T, T], K[E, T], C[T], C[E], y[T, None], np.array([0.3]))
        M = np.eye(n_t) - C[T] @ np.linalg.pinv(C[T])
        a = 0.3 * np.clip(np.linalg.eigvalsh(M @ K[T, T] @ M), 0, None).sum() / (n_t - 3)
        A = np.block([[C[T].T @ C[T], C[T].T @ X[T]], [X[T].T @ C[T], X[T].T @ X[T] + a * np.eye(d)]])
        coef = np.linalg.solve(A, np.concatenate([C[T].T @ y[T], X[T].T @ y[T]]))
        assert np.allclose(C[E] @ coef[:3] + X[E] @ coef[3:], pred[:, 0], atol=1e-8)
        loo_b = []
        for i in range(n_t):
            m = np.ones(n_t, bool)
            m[i] = False
            Xi, Ci, yi = X[T][m], C[T][m], y[T][m]
            Ai = np.block([[Ci.T @ Ci, Ci.T @ Xi], [Xi.T @ Ci, Xi.T @ Xi + a * np.eye(d)]])
            cf = np.linalg.solve(Ai, np.concatenate([Ci.T @ yi, Xi.T @ yi]))
            loo_b.append(C[T][i] @ cf[:3] + X[T][i] @ cf[3:])
        assert np.allclose(loo_b, loo[:, 0], atol=1e-8)
        assert np.allclose(bh_fdr(np.array([0.01, 0.04, 0.03, 0.5])), [0.04, 0.16 / 3, 0.16 / 3, 0.5])
        assert auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0

    def t_planted():
        acts, meta = _synthetic_acts(eval_sets, signal_layer=3)
        sub = {"der_matched": eval_sets["der_matched"]}
        c = copy.deepcopy(cfg)
        c["probes"]["plan"] = {"der_matched": {"log_subtlex": ["none"]}}
        m1 = run_probes(acts, meta, meta0, sub, build_jobs(c, sub), c, "last", tmp)
        m2 = run_probes(acts, meta, meta0, sub, build_jobs(c, sub), c, "last", tmp)
        pd.testing.assert_frame_equal(m1, m2)                    # bit-for-bit reproducible
        f = m1.set_index("layer")
        assert f.loc[3, "r2_lin"] > 0.5 and f.drop(index=3).r2_lin.max() < 0.1
        assert f.loc[3, "p_maxT_layers"] <= 0.05

    def t_conditional():
        acts, meta = _synthetic_acts(eval_sets, signal_layer=2, target="log_subtlex")
        sub = {"der_matched": eval_sets["der_matched"]}
        c = copy.deepcopy(cfg)
        c["probes"]["plan"] = {"der_matched": {"log1p_wn_n_all": ["none", "primary"]}}
        m = run_probes(acts, meta, meta0, sub, build_jobs(c, sub), c, "last", tmp)
        assert abs(m[m.control_set == "primary"].set_index("layer").loc[2, "delta_r2_lin"]) < 0.05

    def t_stage_outputs():
        acts, meta = _synthetic_acts(eval_sets, n_layers=5, d=32)
        dirs = out_dirs(tmp / "out_synth")
        c = copy.deepcopy(cfg)
        c["probes"]["plan"] = {"pub_B": {"ambiguity_bin": ["none"], "senses_bin": ["none"]},
                               "der_matched": {"log_subtlex": ["none", "primary"], "log1p_wn_n_all": ["none"]}}
        sub = {k: eval_sets[k] for k in ("pub_A", "pub_B", "pub_C", "der_matched", "der_full")}
        metrics = run_probes(acts, meta, meta0, sub, build_jobs(c, sub), c, "last", dirs["predictions"])
        prox, d_last = compute_proxies(acts, acts, meta)
        assert np.allclose(d_last[:, -1], 0, atol=1e-6) and prox.rsc_last.notna().all()
        rodd = run_stage1(meta0, prox, d_last, {"last": acts}, meta, sub, c, dirs["predictions"], dirs["metrics"], None)
        assert {"ambiguity", "senses", "ambiguity:senses"} <= set(rodd["coefs"].term)
        assert len(rodd["ccgp"]) and rodd["ccgp"].ccgp_auroc.between(0, 1).all()
        s2 = run_stage2(meta0, prox, sub, c, None)
        assert {"sense_count", "meanings_proxy"} <= set(s2.role)
        table4(metrics, dirs["tables"])
        make_all(dirs, df, sub, metrics, rodd, s2, None, c)
        for f in ("fig01_workflow", "fig02_frequency_layerwise", "fig04_rodd_factorial", "fig08_data_audit"):
            assert (dirs["figures"] / f"{f}.pdf").exists() and (dirs["figures_png"] / f"{f}.png").exists()

    model_box = {}

    def _tiny_model_dir():
        if "d" not in model_box:
            import transformers
            tok = tok_box.get("tok") or _tiny_tokenizer(df)
            mc = transformers.LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                                          pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
                                          eos_token_id=tok.eos_token_id)
            torch.manual_seed(0)
            d = tmp / "tiny_llama"
            transformers.LlamaForCausalLM(mc).save_pretrained(d)
            tok.save_pretrained(d)
            model_box["d"] = d
        return model_box["d"]

    def t_extraction():
        d = _tiny_model_dir()
        words = unique_words(df).iloc[np.r_[0:10, 400:410, 1500:1510]].sort_values("word_idx").reset_index(drop=True)
        for variant in ("base", "instruct"):
            c = copy.deepcopy(cfg)
            c["extraction"].update({"shard_size": 7, "batch_size": 8, "dtype": "float32", "storage_dtype": "float32"})
            spec = ModelSpec(str(d), "llama", variant, "tiny", None)
            d1 = extract_model(spec, words, c, tmp / f"a_{variant}", "cpu", resume=True)
            acts, meta, mm = load_activations(d1, "last")
            assert acts.shape == (len(words), 4, 32) and (meta.status == "ok").all()
            assert mm["final_hidden_state_is_normed"] is True
            assert np.allclose(np.array([json.loads(s) for s in meta.logit_lens_kl])[:, -1], 0, atol=1e-4)
            c1 = copy.deepcopy(c)
            c1["extraction"]["batch_size"] = 1
            acts1, _, _ = load_activations(extract_model(spec, words, c1, tmp / f"b_{variant}", "cpu", resume=False), "last")
            assert np.allclose(acts, acts1, atol=1e-4)               # batching never changes an item's states
            before = json.loads((d1 / "manifest.json").read_text())["shards"]
            assert extract_model(spec, words, c, tmp / f"a_{variant}", "cpu", resume=True) == d1
            assert before == json.loads((d1 / "manifest.json").read_text())["shards"]

    def t_full_cli():
        g = globals()
        d = _tiny_model_dir()
        base, inst = str(d), str(d) + "#chat"
        c = copy.deepcopy(load_config())
        c["models"] = [{"id": base, "family": "llama", "variant": "base", "size": "1B"},
                       {"id": inst, "family": "llama", "variant": "instruct", "size": "1B", "pair": base}]
        c["dry_run"] = {"n_words": 120, "models": [base, inst]}
        c["extraction"]["dtype"] = "float32"
        p = tmp / "cfg.yaml"
        p.write_text(yaml.safe_dump(c))
        orig = g["load_model"]

        def _load(spec, revision, dtype, device):     # same weights; '#chat' entry uses the chat template
            real = ModelSpec(spec.id.split("#")[0], spec.family, spec.variant, spec.size, spec.pair)
            tok, model, meta = orig(real, revision, dtype, device)
            meta["model_id"] = spec.id
            return tok, model, meta

        g["load_model"] = _load
        try:
            main(["--config", str(p), "--data", str(data_path), "--dry-run", "--output-dir", str(tmp / "cli"),
                  "--device", "cpu"])
        finally:
            g["load_model"] = orig
            logging.getLogger("lexamb").setLevel(logging.WARNING)
        out = tmp / "cli"
        for f in ("reports/results_report.md", "run_manifest.json", "metrics/pairs_curves.parquet",
                  "figures/fig06_base_vs_instruct.pdf", "tables/table7_base_instruct.tex"):
            assert (out / f).exists(), f

    print(f"lexamb self-test (scratch dir: {tmp})")
    for name, fn in [("dataset counts & Stage 2 attrition", t_counts), ("within-set deduplication", t_dedup),
                     ("validation fails loudly", t_fail_loudly), ("target alignment, all 1,942 words", t_alignment),
                     ("no label leakage in prompts", t_leakage), ("grouped/stratified folds", t_folds),
                     ("kernel ridge == primal, exact LOO, BH, AUROC", t_kernel_ridge),
                     ("planted signal recovered + reproducible", t_planted),
                     ("conditional probe removes covariate-only signal", t_conditional),
                     ("Stage 1 / Stage 2 / figures / tables", t_stage_outputs),
                     ("extraction: layers, batching, resume (tiny random Llama)", t_extraction),
                     ("full CLI end-to-end with a base/instruct pair", t_full_cli)]:
        check(name, fn)
    n_fail = sum(r[1] == "FAIL" for r in results)
    print(f"{len(results) - n_fail}/{len(results)} checks passed or skipped; {n_fail} failed")
    if n_fail == 0:
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if n_fail else 0


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.write_config:
        Path(args.write_config).write_text(DEFAULT_CONFIG_YAML)
        print(f"wrote {args.write_config}")
        return 0
    data_path = resolve_data_path(cfg, args.data)
    if args.self_test:
        return self_test(data_path)
    if args.dry_run:
        cfg = apply_dry_run(cfg, args.dry_run_n)
    out = Path(args.output_dir or (cfg["output_dir"] + ("_dryrun" if args.dry_run else "")))
    dirs = out_dirs(out if out.is_absolute() else Path.cwd() / out)
    setup_logging(dirs["root"] / "logs", "dryrun" if args.dry_run else "run")
    set_seeds(cfg["seed"])
    t0 = dt.datetime.now(dt.timezone.utc).isoformat()
    device = auto_device(args.device)
    specs = select(registry(cfg), args.models or (cfg["dry_run"]["models"] if args.dry_run else None))
    LOG.info("stage=%s device=%s models=%s output=%s", args.stage, device, [s.id for s in specs], dirs["root"])

    df, eval_sets, key = stage_audit(cfg, dirs, data_path)
    words = unique_words(df).sort_values("word_idx").reset_index(drop=True)
    if args.dry_run:
        words = dry_run_words(words, df, cfg["dry_run"]["n_words"], cfg["seed"])
        LOG.info("dry-run words: %s", list(words.word_lc))
    st = args.stage
    if st in ("extract", "all"):
        stage_extract(cfg, dirs, specs, words, device, args.resume)
    if st in ("analyse", "all"):
        stage_analyse(cfg, dirs, specs, eval_sets, args.resume)
    if st in ("compare", "all"):
        stage_compare(cfg, dirs, specs)
    if st in ("report", "all"):
        LOG.info("report written to %s", stage_report(cfg, dirs, df, eval_sets, key, args.dry_run))
    write_manifest(cfg, dirs, args, t0, data_path)
    LOG.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
