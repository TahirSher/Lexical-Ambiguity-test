# Layer-wise lexical frequency and semantic ambiguity in base vs instruct LLMs

End-to-end, reproducible pipeline implementing `semantic_ambiguity_layerwise_claude_spec.md`. It
asks how word frequency, WordNet sense count, the published Rodd et al. (2002) meanings/senses
factors, a computed GCIDE meanings proxy and WordNet semantic distance are linearly recoverable from
the target-word representation at every layer of 9 base and 6 instruction-tuned LLMs (6 matched
pairs). It also asks whether a pre-registered, model-derived settling proxy shows the Rodd
directional dissociation: facilitation from many senses, inhibition from unrelated meanings.

Scientific rationale, research gaps, hypotheses and novelty are covered in
[`docs/RESEARCH_RATIONALE.md`](docs/RESEARCH_RATIONALE.md).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # pick the torch wheel that matches your CUDA
huggingface-cli login                       # Llama 3.x repos are gated: accept the licence first
```

## Run

```bash
python run_pipeline.py --stage audit                 # validates the CSV, writes outputs/audit/*
python run_pipeline.py --dry-run                     # 10 words, SmolLM2-360M, every stage -> outputs_dryrun/
python -m pytest tests -q                            # offline unit tests (tiny tokenizer + synthetic states)
python run_pipeline.py --stage all --resume --device cuda          # full study
python run_pipeline.py --stage extract --models Qwen/Qwen2.5-3B Qwen/Qwen2.5-3B-Instruct --resume
```

Stages: `audit -> extract -> analyse -> compare -> report` (`all` runs them in order). `--resume`
skips activation shards whose SHA-256 checksums validate, and models whose analysis already
finished. Every analytic decision lives in `configs/experiment.yaml`. That includes model list,
seeds, dtype, prompts, pooling, folds, repeats, permutations, bootstrap counts, filter ranges, the
GCIDE zero policy, control sets and the probe plan. For the final run, pin `extraction.revision` to
a commit hash. The resolved commit is always logged.

**Compute.** Extraction is one short forward pass per unique word (1,942 words). The 8B models need
roughly 18 GB of GPU memory in bf16. Probing is CPU-bound linear algebra, at about 0.5–1 h per model
with the default 3×5-fold CV and 200 permutations. Lower `probes.n_repeats` or `n_permutations` to
trade precision for time.

## What is computed

| Spec item | Implementation |
|---|---|
| Neutral prompts, no leakage | `Word: {word}` (base) / native chat template with the same content (instruct). Labels are joined only after extraction; a unit test asserts the exact prompt text. |
| Target position | Character-offset alignment to the target word's subtokens. Last subtoken = primary, mean = sensitivity. Failures are logged, never silently skipped; a 50-word manual audit sheet is written per tokenizer. |
| Layers | `hidden_states` length is checked to be L+1 (0 = embeddings). Whether the final state is already normed is verified numerically. |
| Probes (Outcome A) | Dual-form ridge with exact LOO-selected α. Repeated StratifiedGroupKFold by word, with identical folds for every model. Out-of-fold R², MAE and Spearman, or AUROC, balanced accuracy, macro-F1 and Brier (Platt-calibrated on LOO scores). |
| Unique information (Outcome B) | Conditional probes: covariates are unpenalised and partialled out by Frisch–Waugh–Lovell inside each fold, and ΔR² over the covariate-only baseline is reported. The partner variable is controlled (e.g. frequency beyond sense count). |
| Inference | Freedman–Lane permutation null shared across layers gives a per-layer p, BH-FDR across layers and Westfall–Young max-T. 95% CIs come from a word-level bootstrap. |
| Outcome C | Length-normalised word log-likelihood under the common prefix, next-token entropy and logit-lens KL. All are secondary and analysed separately. |
| Outcome D | Optional ease rating from the restricted "1".."7" distribution, with 3 paraphrases. Exploratory, off by default. |
| Stage 1 | B and C each get one effect-coded factorial model (amb × senses + controls), with HC3 and cell-stratified bootstrap CIs and the operational pass rule. Appendix A is analysed separately and flagged as confounded. Also: layer-wise time course, CCGP, factorial selectivity. |
| Stage 2 | Proxy ~ sense count + GCIDE + controls on the matched range. Sensitivity runs cover zero recoding, a zero indicator, the full pool, expanded controls, noun senses and semantic distance. |
| Base vs instruct | Paired word-level bootstrap per layer, CoG shift, paired Rodd-coefficient differences, exploratory CKA. Base-only models are never paired. |
| Outputs | `outputs/{audit,activations,predictions,metrics,tables,figures,figures_png,reports}` plus `run_manifest.json` (config, data hash, versions, model commits, output hashes). |

## Interpretation guard-rails (enforced in the report text)

- A probe peak means "most linearly decodable under this protocol". It does not mean the model resolves ambiguity at that layer.
- None of the proxies is a human reaction time. Appendix C is an auditory stimulus set given to text models.
- GCIDE etymologies are a computed proxy, and WordNet candidate labels are rule-derived. Stage 2 is generalisation, not replication.
- Base and instruct differences are measured under each variant's native input format.

## Layout

```
configs/experiment.yaml   run_pipeline.py   requirements.txt   tests/
src/lexamb/  audit_data · model_registry · prompt_formatting · extract_hidden_states ·
             score_token_likelihood · ease_rating · train_layerwise_probes · stats_utils ·
             analyse_rodd_factorial · analyse_stage2 · compare_base_instruct ·
             make_figures · make_tables · make_report · common
```
`stats_utils.py` corresponds to the spec's `statistics.py`. It was renamed so it does not shadow Python's standard-library `statistics` module.
