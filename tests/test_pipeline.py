"""Unit tests required by spec §11: counts, duplicates, prompt rendering, alignment, layer count, leakage,
grouped-split integrity, reproducibility, estimator correctness, figure/table generation."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_acts
from lexamb import audit_data, stats_utils as su
from lexamb.prompt_formatting import render
from lexamb.train_layerwise_probes import build_jobs, make_folds, run_probes

MODEL_META = {"model_id": "test/model", "family": "smollm", "variant": "base", "size": "360M", "paired_base": None}


# ----------------------------------------------------------------------------- data
def test_data_counts(df):
    assert df.shape[0] == 2035
    assert df.provenance.value_counts().to_dict() == {"wordnet-derived": 1633, "published": 402}
    pub = df[df.provenance == "published"]
    assert pub.word_lc.nunique() == 309
    assert pub[pub.senses.notna()].appendix.value_counts().to_dict() == {"B": 128, "C": 92}


def test_stage2_counts_recomputed(df, cfg):
    att = audit_data.stage2_attrition(df, cfg).set_index("step")
    assert att.loc[2, "n_unique_words"] == 646      # spec: ~646 in range
    assert att.loc[3, "n_unique_words"] == 596      # spec: ~596 with GCIDE observed


def test_duplicate_handling(eval_sets):
    for es in eval_sets.values():
        assert es.items.word_lc.is_unique, es.name
        assert len(es.strat) == len(es.items)
    assert len(eval_sets["pub_all"].items) == 309
    assert set(eval_sets["pub_all"].items.provenance) == {"published"}
    assert set(eval_sets["der_full"].items.provenance) == {"wordnet-derived"}
    cells = eval_sets["pub_B"].items.groupby(["ambiguity", "senses"]).size()
    assert (cells == 32).all()


def test_validation_fails_loudly(cfg, tmp_path):
    bad = dict(cfg["data"], expected_rows=9999)
    with pytest.raises(audit_data.DataValidationError):
        audit_data.load_and_validate(Path(__file__).parents[1] / cfg["data"]["csv"], bad)


# ----------------------------------------------------------------------------- prompts / alignment
@pytest.mark.parametrize("variant", ["base", "instruct"])
def test_alignment_all_words(tiny_tokenizer, df, cfg, variant):
    fails = []
    for w in df.word_lc.unique():
        r = render(tiny_tokenizer, w, variant, cfg["extraction"])
        if not r.ok:
            fails.append((w, r.reason))
            continue
        assert tiny_tokenizer.decode([r.input_ids[k] for k in r.target_idx]).strip() == w
        assert r.target_idx == list(range(r.target_idx[0], r.target_idx[-1] + 1))
    assert not fails, fails[:5]


def test_multi_subtoken_words_exist(tiny_tokenizer, df, cfg):
    n = [len(render(tiny_tokenizer, w, "base", cfg["extraction"]).target_idx) for w in df.word_lc.unique()[:300]]
    assert max(n) > 1


def test_no_label_leakage(tiny_tokenizer, df, cfg):
    """The prompt is exactly the neutral template; no label/count/construct text can appear."""
    t = audit_data.add_derived_targets(df)
    forbidden = ["ambiguous", "unambiguous", "homonym", "polyseme", "senses", "meaning", "few", "many", "etymolog"]
    for _, row in t.sample(200, random_state=0).iterrows():
        w = row.word_lc
        rb = render(tiny_tokenizer, w, "base", cfg["extraction"])
        assert rb.text == f"Word: {w}"
        ri = render(tiny_tokenizer, w, "instruct", cfg["extraction"])
        assert ri.text == f"<|im_start|>user\nWord: {w}<|im_end|>\n"
        for f in forbidden:
            if f not in w:
                assert f not in rb.text.lower() and f not in ri.text.lower()
        for c in audit_data.LABEL_COLS:
            v = row[c]
            if isinstance(v, str):
                assert v not in rb.text.replace(w, "")


# ----------------------------------------------------------------------------- folds
def test_grouped_split_integrity(eval_sets):
    es = eval_sets["pub_B"]
    f1 = make_folds(es, 5, 3, 7)
    f2 = make_folds(es, 5, 3, 7)
    assert np.array_equal(f1, f2)                      # deterministic -> identical across models
    for r in range(3):
        for w in np.unique(es.items.word_idx):
            assert len(np.unique(f1[r, es.items.word_idx.to_numpy() == w])) == 1
        # cell stratification keeps every fold populated by all 4 cells
        cells = es.items.ambiguity + es.items.senses
        assert pd.crosstab(f1[r], cells).min().min() >= 1
    assert not np.array_equal(f1[0], f1[1])            # repeats differ


# ----------------------------------------------------------------------------- estimator
def test_kernel_ridge_matches_primal():
    rng = np.random.default_rng(1)
    n_t, n_e, d = 50, 15, 120
    X = rng.normal(size=(n_t + n_e, d))
    C = np.column_stack([np.ones(n_t + n_e), rng.normal(size=(n_t + n_e, 2))])
    y = X[:, 0] + 2 * C[:, 1] + rng.normal(size=n_t + n_e)
    T, E = slice(0, n_t), slice(n_t, None)
    K = X @ X.T
    pred, loo, _, _ = su.fwl_kernel_ridge(K[T, T], K[E, T], C[T], C[E], y[T, None], np.array([0.3]))
    M = np.eye(n_t) - C[T] @ np.linalg.pinv(C[T])
    s = np.clip(np.linalg.eigvalsh(M @ K[T, T] @ M), 0, None)
    a = 0.3 * s.sum() / (n_t - 3)
    A = np.block([[C[T].T @ C[T], C[T].T @ X[T]], [X[T].T @ C[T], X[T].T @ X[T] + a * np.eye(d)]])
    coef = np.linalg.solve(A, np.concatenate([C[T].T @ y[T], X[T].T @ y[T]]))
    assert np.allclose(C[E] @ coef[:3] + X[E] @ coef[3:], pred[:, 0], atol=1e-8)


def test_bh_and_perm_helpers():
    q = su.bh_fdr(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.5])
    assert su.auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_probe_recovers_planted_signal_and_is_reproducible(eval_sets, cfg, tmp_path):
    acts, meta = synthetic_acts(eval_sets, signal_layer=3, target="log_subtlex")
    sub = {"der_matched": eval_sets["der_matched"]}
    c = dict(cfg)
    c["probes"] = dict(cfg["probes"], plan={"der_matched": {"log_subtlex": ["none"], "log1p_wn_n_all": ["none"]}})
    jobs = build_jobs(c, sub)
    m1 = run_probes(acts, meta, MODEL_META, sub, jobs, c, "last", tmp_path)
    m2 = run_probes(acts, meta, MODEL_META, sub, jobs, c, "last", tmp_path)
    pd.testing.assert_frame_equal(m1, m2)
    f = m1[m1.target == "log_subtlex"].set_index("layer")
    assert f.loc[3, "r2_lin"] > 0.5
    assert f.drop(index=3).r2_lin.max() < 0.1
    assert f.loc[3, "p_maxT_layers"] <= 0.05
    assert f.loc[0, "p_perm"] > 0.05 or f.loc[0, "r2_lin"] < 0.05


def test_conditional_probe_removes_covariate_only_signal(eval_sets, cfg, tmp_path):
    """Layer encodes ONLY log_subtlex; the sense-count probe beyond frequency must find ~no increment."""
    acts, meta = synthetic_acts(eval_sets, signal_layer=2, target="log_subtlex")
    sub = {"der_matched": eval_sets["der_matched"]}
    c = dict(cfg)
    c["probes"] = dict(cfg["probes"], plan={"der_matched": {"log1p_wn_n_all": ["none", "primary"]}})
    m = run_probes(acts, meta, MODEL_META, sub, build_jobs(c, sub), c, "last", tmp_path)
    prim = m[(m.control_set == "primary")].set_index("layer")
    assert abs(prim.loc[2, "delta_r2_lin"]) < 0.05


# ----------------------------------------------------------------------------- Rodd / stage 2 / outputs
def test_stage1_stage2_and_outputs(eval_sets, df, cfg, tmp_path):
    from lexamb.analyse_rodd_factorial import compute_proxies, run_stage1
    from lexamb.analyse_stage2 import run_stage2
    from lexamb import make_figures, make_tables
    from lexamb.common import out_dirs

    acts, meta = synthetic_acts(eval_sets, n_layers=5, d=32)
    dirs = out_dirs(tmp_path / "out")
    c = dict(cfg)
    c["probes"] = dict(cfg["probes"], plan={"pub_B": {"ambiguity_bin": ["none"], "senses_bin": ["none"]},
                                            "der_matched": {"log_subtlex": ["none", "primary"],
                                                            "log1p_wn_n_all": ["none"]}})
    sub = {k: eval_sets[k] for k in ("pub_A", "pub_B", "pub_C", "der_matched", "der_full")}
    metrics = run_probes(acts, meta, MODEL_META, sub, build_jobs(c, sub), c, "last", dirs["predictions"])
    prox, d_last = compute_proxies(acts, acts, meta)
    assert np.allclose(d_last[:, -1], 0, atol=1e-6)            # final layer distance to itself is zero
    assert prox.rsc_last.notna().all()
    rodd = run_stage1(MODEL_META, prox, d_last, {"last": acts}, meta, sub, c, dirs["predictions"], dirs["metrics"], None)
    assert set(rodd["coefs"].appendix) >= {"B", "C", "A"}
    assert {"ambiguity", "senses", "ambiguity:senses"} <= set(rodd["coefs"].term)
    assert len(rodd["ccgp"]) and rodd["ccgp"].ccgp_auroc.between(0, 1).all()
    s2 = run_stage2(MODEL_META, prox, sub, c, None)
    assert {"sense_count", "meanings_proxy"} <= set(s2.role)
    assert (s2[s2.analysis_variant == "primary"].n <= 542).all()
    make_tables.table4(metrics, dirs["tables"])
    make_tables.table5(rodd["coefs"], rodd["verdicts"], dirs["tables"])
    make_tables.table6(s2, dirs["tables"])
    assert (dirs["tables"] / "table4_layerwise_performance.tex").exists()
    make_figures.make_all(dirs, df, sub, metrics, rodd, s2, None, c)
    for f in ("fig01_workflow", "fig02_frequency_layerwise", "fig04_rodd_factorial", "fig08_data_audit"):
        assert (dirs["figures"] / f"{f}.pdf").exists() and (dirs["figures_png"] / f"{f}.png").exists()
