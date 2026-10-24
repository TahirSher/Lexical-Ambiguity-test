import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lexamb import audit_data  # noqa: E402
from lexamb.common import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    c = load_config(ROOT / "configs" / "experiment.yaml")
    c["probes"].update({"n_folds": 3, "n_repeats": 2, "n_permutations": 30, "n_bootstrap": 60, "min_n": 10})
    c["rodd"].update({"n_bootstrap": 100, "n_permutations": 20})
    return c


@pytest.fixture(scope="session")
def df(cfg):
    return audit_data.load_and_validate(ROOT / cfg["data"]["csv"], cfg["data"])


@pytest.fixture(scope="session")
def eval_sets(df, cfg):
    return audit_data.build_eval_sets(df, cfg)


@pytest.fixture(scope="session")
def tiny_tokenizer(df):
    """Offline byte-level BPE tokenizer trained on the stimulus words (multi-subtoken words guaranteed)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["<s>", "<pad>", "<|im_start|>", "<|im_end|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    corpus = [f"Word: {w}" for w in df.word_lc.unique()] + ["user assistant system"]
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", pad_token="<pad>", eos_token="<|im_end|>")
    fast.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                          "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    return fast


def synthetic_acts(eval_sets, n_layers=6, d=48, seed=0, signal_layer=3, target="log_subtlex"):
    """Activations whose layer `signal_layer` linearly encodes `target` (+ noise); other layers are noise."""
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
