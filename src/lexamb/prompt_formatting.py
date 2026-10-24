"""Neutral prompt rendering and target-subtoken alignment (spec §5).

Base models see ``Word: {word}`` as plain text; instruct models see the same content inside their
native chat template (add_generation_prompt=False). No label, count or construct word ever enters a
prompt. The hidden state is read at the target word's own subtokens, located by character offsets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .common import sha256_text


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
