"""Shared utilities: configuration, logging, hashing, atomic IO, table writers."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

LOG = logging.getLogger("lexamb")

OUTPUT_SUBDIRS = ("audit", "activations", "predictions", "metrics", "tables",
                  "figures", "figures_png", "reports")


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
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
