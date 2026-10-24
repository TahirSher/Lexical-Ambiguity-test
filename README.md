# Layer-wise lexical frequency and semantic ambiguity in base vs instruct LLMs

The whole study is one self-contained Python script, **`lexical_ambiguity_pipeline.py`**. It runs
the data audit, hidden-state extraction, layer-wise probes, the Rodd Stage 1 tests, the Stage 2
generalisation, base vs instruct comparisons, figures, tables, the report and the run manifest. The
experiment configuration is embedded in the script as `DEFAULT_CONFIG_YAML`, and a built-in
`--self-test` replaces a separate test suite.

It measures how word frequency, WordNet sense count, the published Rodd et al. (2002)
meanings/senses factors, a computed GCIDE meanings proxy and WordNet semantic distance are linearly
recoverable from the target-word representation at every layer. The inventory is 9 base and 6
instruction-tuned LLMs, forming 6 matched pairs. It also tests whether a pre-registered settling
proxy shows the Rodd directional dissociation: facilitation from many senses, inhibition from
unrelated meanings.

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
python lexical_ambiguity_pipeline.py --self-test          # 12 offline checks, no model download (~1 min)
python lexical_ambiguity_pipeline.py --stage audit        # validate the CSV -> outputs/audit/
python lexical_ambiguity_pipeline.py --dry-run            # 10 words, SmolLM2-360M, every stage -> outputs_dryrun/
python lexical_ambiguity_pipeline.py --stage all --resume --device cuda               # full study
python lexical_ambiguity_pipeline.py --stage extract --models Qwen/Qwen2.5-3B Qwen/Qwen2.5-3B-Instruct --resume
python lexical_ambiguity_pipeline.py --write-config my.yaml   # dump the embedded config, edit it, then
python lexical_ambiguity_pipeline.py --config my.yaml --stage all
```

Stages run in the order `audit -> extract -> analyse -> compare -> report`, and `all` runs all of
them. `--resume` skips activation shards whose SHA-256 checksums validate, and models whose analysis
already finished. The CSV is found next to the script or in the working directory; use `--data PATH`
to point elsewhere.

Every analytic decision lives in the embedded config: models, seeds, dtype, prompts, pooling, folds,
repeats, permutations, bootstraps, filter ranges, the GCIDE zero policy, control sets and the probe
plan. For the final run, pin `extraction.revision` to a commit hash. The resolved commit is always
logged.

**Compute.** Extraction is one short forward pass per unique word (1,942 words). The 8B models need
roughly 18 GB of GPU memory in bf16. Probing is CPU-bound linear algebra, at about 0.5–1 h per model
with the default 3×5-fold CV and 200 permutations.

## Script layout (sections, in order)

| Section | What it does |
|---|---|
| configuration | `DEFAULT_CONFIG_YAML`, `load_config` |
| common | logging, hashing, atomic IO, CSV/Markdown/LaTeX table writer |
| audit_data | `load_and_validate` (hard count checks), `build_eval_sets`, audit tables |
| model_registry | inventory, base/instruct pairing, `load_model` (gated-repo message) |
| prompt_formatting | `render`: neutral prompts plus character-offset target-subtoken alignment |
| score_token_likelihood | word log-likelihood, next-token entropy, logit lens |
| extract_hidden_states | `extract_model`: resumable, checksummed shards, L+1 layer check, final-norm check |
| ease_rating | optional exploratory 1–7 rating, off by default |
| stats_utils | `fwl_kernel_ridge` (exact LOO), metrics, BH, max-T, bootstrap, HC3, CKA |
| train_layerwise_probes | `run_probes`: repeated grouped CV, conditional increments, Freedman–Lane permutations |
| analyse_rodd_factorial | settling proxy, B/C 2×2 factorial with pass rule, time course, CCGP, selectivity |
| analyse_stage2 | derived-pool regressions plus GCIDE-zero, full-pool and distance sensitivities |
| compare_base_instruct | paired word-level bootstrap curves, CoG shift, Rodd differences, CKA |
| make_figures / make_tables / make_report | Figures 1–8 plus supplementary, Tables 1–8 and S1–S3, Markdown report |
| CLI | stages, `run_manifest.json`, `--self-test`, `main` |

## Interpretation guard-rails (enforced in the report text)

- A probe peak means "most linearly decodable under this protocol". It does not mean the model resolves ambiguity at that layer.
- None of the proxies is a human reaction time. Appendix C is an auditory stimulus set given to text models.
- GCIDE etymologies are a computed proxy, and WordNet candidate labels are rule-derived. Stage 2 is generalisation, not replication.
- Base and instruct differences are measured under each variant's native input format.
