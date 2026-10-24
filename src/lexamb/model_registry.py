"""Model inventory, base/instruct pairing and loading (spec §4)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

LOG = logging.getLogger("lexamb")


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
