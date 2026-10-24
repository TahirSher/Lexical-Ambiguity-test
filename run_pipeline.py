#!/usr/bin/env python
"""End-to-end pipeline: audit -> extract -> analyse -> compare -> report.

Examples
--------
  python run_pipeline.py --stage audit
  python run_pipeline.py --dry-run                         # 10 words, SmolLM2-360M, all stages
  python run_pipeline.py --stage all --resume --device cuda
  python run_pipeline.py --stage extract --models meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-1B-Instruct
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lexamb import audit_data, model_registry as mr  # noqa: E402
from lexamb.audit_data import EvalSet  # noqa: E402
from lexamb.common import (load_config, out_dirs, set_seeds, setup_logging, sha256_file, slug,  # noqa: E402
                           write_json, write_table)

LOG = logging.getLogger("lexamb")
STAGES = ["audit", "extract", "analyse", "compare", "report", "all"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    p.add_argument("--models", nargs="*", default=None, help="subset of configured model ids")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--dry-run", action="store_true", help="10 words, one small model, reduced resampling")
    p.add_argument("--dry-run-n", type=int, default=None, help="override the dry-run word count")
    p.add_argument("--resume", action="store_true", help="skip validated shards / finished analyses")
    p.add_argument("--device", default=None, help="cuda, cuda:0, mps or cpu (default: auto)")
    p.add_argument("--output-dir", default=None)
    return p.parse_args(argv)


def auto_device(requested):
    if requested:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
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


# ============================================================================ stages
def stage_audit(cfg, dirs):
    df = audit_data.load_and_validate(ROOT / cfg["data"]["csv"], cfg["data"])
    eval_sets = audit_data.build_eval_sets(df, cfg)
    key = audit_data.write_audit(df, cfg, dirs, eval_sets)
    write_json(dirs["audit"] / "audit_key_counts.json", key)
    return df, eval_sets, key


def stage_extract(cfg, dirs, specs, words, device, resume):
    from lexamb.extract_hidden_states import extract_model
    status = {}
    for spec in specs:
        try:
            d = extract_model(spec, words, cfg, dirs["activations"], device, resume)
            status[spec.id] = {"status": "ok", "dir": str(d)}
            if cfg["ease_rating"]["enabled"] and spec.variant == "instruct":
                from lexamb.ease_rating import rate_words
                tok, model, _ = mr.load_model(spec, cfg["extraction"]["revision"], cfg["extraction"]["dtype"], device)
                rate_words(tok, model, words, cfg, device).to_parquet(d / "ease_rating.parquet", index=False)
                del model
        except Exception as err:  # never silently skip: record and continue with the next model
            LOG.error("extraction failed for %s: %s", spec.id, err)
            status[spec.id] = {"status": "failed", "error": str(err)}
    write_json(dirs["activations"] / "extraction_status.json", status)
    return status


def stage_analyse(cfg, dirs, specs, eval_sets, resume):
    from lexamb.analyse_rodd_factorial import compute_proxies, run_stage1
    from lexamb.analyse_stage2 import run_stage2
    from lexamb.extract_hidden_states import load_activations, model_dir
    from lexamb.train_layerwise_probes import build_jobs, run_probes

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
        frames = []
        for pooling, a in acts.items():
            frames.append(run_probes(a, wmeta, mmeta, es, jobs, cfg, pooling, dirs["predictions"]))
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
    files = sorted(dirs["metrics"].glob(f"{prefix}__*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def stage_compare(cfg, dirs, specs):
    from lexamb.compare_base_instruct import cka_by_layer, paired_curves, paired_rodd
    from lexamb.extract_hidden_states import load_activations, model_dir
    metrics = _collect(dirs, "probe_metrics")
    rodd_coefs = _collect(dirs, "rodd_coefs")
    ids = {s.id for s in specs}
    out, cka, rp = [], [], []
    for base, inst in mr.pairs(mr.registry(cfg)):
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
    cat = lambda xs: pd.concat([x for x in xs if len(x)], ignore_index=True) if any(len(x) for x in xs) else pd.DataFrame()
    pairs_df, cka_df, rp_df = cat(out), cat(cka), cat(rp)
    pairs_df.to_parquet(dirs["metrics"] / "pairs_curves.parquet", index=False)
    cka_df.to_parquet(dirs["metrics"] / "pairs_cka.parquet", index=False)
    rp_df.to_parquet(dirs["metrics"] / "pairs_rodd.parquet", index=False)
    return pairs_df, cka_df, rp_df


def stage_report(cfg, dirs, df, eval_sets, audit_key, dry_run):
    from lexamb import make_figures, make_report, make_tables
    metrics = _collect(dirs, "probe_metrics")
    rodd = {k: _collect(dirs, f"rodd_{k}") for k in ("coefs", "verdicts", "timecourse", "ccgp", "selectivity")}
    stage2 = _collect(dirs, "stage2")
    rd = lambda n: pd.read_parquet(dirs["metrics"] / n) if (dirs["metrics"] / n).exists() else pd.DataFrame()
    pairs_df, cka_df, rp_df = rd("pairs_curves.parquet"), rd("pairs_cka.parquet"), rd("pairs_rodd.parquet")
    metas = [json.loads(p.read_text()) for p in sorted(dirs["metrics"].glob("model_meta__*.json"))]
    base_only = {s.id for s in mr.base_only(mr.registry(cfg))}
    T = dirs["tables"]
    make_tables.table1(audit_key, audit_data.stage2_attrition(df, cfg), eval_sets, T)
    make_tables.table2(metas, base_only, T)
    make_tables.table3(df, T)
    miss = pd.read_csv(dirs["audit"] / "audit_missingness.csv")
    if len(metrics):
        make_tables.table4(metrics, T)
        make_tables.table8(metrics, T)
        make_tables.tableS(metrics, T, miss)
    make_tables.table5(rodd["coefs"], rodd["verdicts"], T)
    make_tables.table6(stage2, T)
    make_tables.table7(pairs_df, T)
    if len(rp_df):
        write_table(rp_df, T / "table7b_base_instruct_rodd", "Table 7b. Instruct - base difference in Rodd proxy effects")
    if len(cka_df):
        write_table(cka_df, T / "tableS3_cka", "Table S3. Exploratory linear CKA, base vs instruct, per layer")
    for k in ("ccgp", "selectivity", "timecourse"):
        if len(rodd[k]):
            rodd[k].to_csv(T / f"tableS_rodd_{k}.csv", index=False)
    make_figures.make_all(dirs, df, eval_sets, metrics, rodd, stage2, pairs_df, cfg)
    return make_report.write_report(dirs, audit_key, metrics, rodd, stage2, pairs_df, cka_df, metas, cfg, dry_run)


def write_manifest(cfg, dirs, args, t0):
    import importlib.metadata as im
    pk = {}
    for name in ("torch", "transformers", "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib",
                 "pyarrow", "pyyaml"):
        try:
            pk[name] = im.version(name)
        except im.PackageNotFoundError:
            pk[name] = None
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    outputs = {str(p.relative_to(dirs["root"])): sha256_file(p) for p in sorted(dirs["root"].rglob("*"))
               if p.is_file() and "activations" not in p.parts and p.name != "run_manifest.json"}
    acts = {str(p.relative_to(dirs["root"])): json.loads(p.read_text()).get("meta", {}).get("model_commit")
            for p in dirs["activations"].glob("*/manifest.json")}
    write_json(dirs["root"] / "run_manifest.json", {
        "started": t0, "finished": dt.datetime.now(dt.timezone.utc).isoformat(), "argv": sys.argv,
        "args": vars(args), "git_commit": commit, "python": platform.python_version(), "platform": platform.platform(),
        "packages": pk, "config": cfg, "data_sha256": sha256_file(ROOT / cfg["data"]["csv"]),
        "activation_manifests": acts, "output_sha256": outputs})


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.dry_run:
        cfg = apply_dry_run(cfg, args.dry_run_n)
    out = args.output_dir or (cfg["output_dir"] + ("_dryrun" if args.dry_run else ""))
    dirs = out_dirs(ROOT / out if not Path(out).is_absolute() else out)
    setup_logging(ROOT / "logs", "dryrun" if args.dry_run else "run")
    set_seeds(cfg["seed"])
    t0 = dt.datetime.now(dt.timezone.utc).isoformat()
    device = auto_device(args.device)
    specs = mr.select(mr.registry(cfg), args.models or (cfg["dry_run"]["models"] if args.dry_run else None))
    LOG.info("stage=%s device=%s models=%s output=%s", args.stage, device, [s.id for s in specs], dirs["root"])

    df, eval_sets, key = stage_audit(cfg, dirs)
    words = audit_data.unique_words(df).sort_values("word_idx").reset_index(drop=True)
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
        path = stage_report(cfg, dirs, df, eval_sets, key, args.dry_run)
        LOG.info("report written to %s", path)
    write_manifest(cfg, dirs, args, t0)
    LOG.info("done")


if __name__ == "__main__":
    main()
