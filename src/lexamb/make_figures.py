"""Publication figures (spec §8): vector PDF + SVG and 300-dpi PNG; plotted values exported to CSV.

Visual system: one sans typeface, recessive axes/grid, 1.6-pt lines, fixed categorical order
(blue, orange, aqua, yellow, ...) assigned by model size within family, line style = variant
(solid base, dashed instruct). Every panel states the pool (published / derived) and its N.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

LOG = logging.getLogger("lexamb")

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
    from .audit_data import REQUIRED_COLS, stage2_attrition
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
