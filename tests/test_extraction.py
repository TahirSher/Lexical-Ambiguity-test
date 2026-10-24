"""Extraction mechanics on a tiny randomly-initialised local Llama (no download needed).

This validates layer count, target alignment inside real forward passes, batching = single-item
equivalence, resumable shards, logit-lens/final-norm handling and the full CLI (all stages, with a
base/instruct pair). Weights are random: outputs are schema checks, never scientific results.
"""
import json

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from lexamb.extract_hidden_states import extract_model, load_activations  # noqa: E402
from lexamb.model_registry import ModelSpec  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model_dir(tiny_tokenizer, tmp_path_factory):
    cfg = transformers.LlamaConfig(vocab_size=len(tiny_tokenizer), hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                                   max_position_embeddings=128, pad_token_id=tiny_tokenizer.pad_token_id,
                                   bos_token_id=tiny_tokenizer.bos_token_id, eos_token_id=tiny_tokenizer.eos_token_id)
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(cfg)
    d = tmp_path_factory.mktemp("tiny_llama")
    model.save_pretrained(d)
    tiny_tokenizer.save_pretrained(d)
    return d


def _words(df, n=30):
    from lexamb.audit_data import unique_words
    return unique_words(df).iloc[np.r_[0:10, 400:410, 1500:1510][:n]].sort_values("word_idx").reset_index(drop=True)


@pytest.mark.parametrize("variant", ["base", "instruct"])
def test_extract_layers_alignment_batching_resume(tiny_model_dir, df, cfg, tmp_path, variant):
    words = _words(df)
    c = json.loads(json.dumps(cfg))
    c["extraction"].update({"shard_size": 7, "batch_size": 8, "dtype": "float32", "storage_dtype": "float32"})
    spec = ModelSpec(str(tiny_model_dir), "llama", variant, "tiny", None)
    d1 = extract_model(spec, words, c, tmp_path / "a", "cpu", resume=True)
    acts, meta, mm = load_activations(d1, "last")
    assert acts.shape == (len(words), 4, 32)                       # L+1 = 4 (embeddings + 3 blocks)
    assert (meta.status == "ok").all()
    assert mm["final_hidden_state_is_normed"] is True
    kl = np.array([json.loads(s) for s in meta.logit_lens_kl])
    assert np.allclose(kl[:, -1], 0, atol=1e-4)                    # final layer vs itself
    # batching must not change any item's states (right padding + causal mask)
    c1 = json.loads(json.dumps(c))
    c1["extraction"]["batch_size"] = 1
    d2 = extract_model(spec, words, c1, tmp_path / "b", "cpu", resume=False)
    acts1, _, _ = load_activations(d2, "last")
    assert np.allclose(acts, acts1, atol=1e-4)
    # resume: validated shards are skipped and the result is identical
    m_before = (d1 / "manifest.json").read_text()
    d3 = extract_model(spec, words, c, tmp_path / "a", "cpu", resume=True)
    assert d3 == d1
    assert json.loads(m_before)["shards"] == json.loads((d3 / "manifest.json").read_text())["shards"]


def test_full_cli_with_local_pair(tiny_model_dir, cfg, tmp_path):
    import run_pipeline
    c = json.loads(json.dumps(cfg))
    base, inst = str(tiny_model_dir), str(tiny_model_dir) + "#chat"
    # same weights, second entry uses the chat template -> exercises pairing code paths
    c["models"] = [{"id": base, "family": "llama", "variant": "base", "size": "1B"},
                   {"id": inst, "family": "llama", "variant": "instruct", "size": "1B", "pair": base}]
    c["dry_run"] = {"n_words": 120, "models": [base, inst]}
    c["extraction"]["dtype"] = "float32"
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(c))
    import lexamb.model_registry as mr
    orig = mr.load_model

    def _load(spec, revision, dtype, device):
        real = ModelSpec(spec.id.split("#")[0], spec.family, spec.variant, spec.size, spec.pair)
        tok, model, meta = orig(real, revision, dtype, device)
        meta["model_id"] = spec.id
        return tok, model, meta

    mr.load_model = _load
    import lexamb.extract_hidden_states as ehs
    ehs.load_model = _load
    try:
        run_pipeline.main(["--config", str(p), "--dry-run", "--output-dir", str(tmp_path / "out"), "--device", "cpu"])
    finally:
        mr.load_model = orig
        ehs.load_model = orig
    out = tmp_path / "out"
    assert (out / "reports" / "results_report.md").exists()
    assert (out / "run_manifest.json").exists()
    assert (out / "metrics" / "pairs_curves.parquet").exists()
    for f in ("fig01_workflow", "fig02_frequency_layerwise", "fig08_data_audit"):
        assert (out / "figures" / f"{f}.pdf").exists()
