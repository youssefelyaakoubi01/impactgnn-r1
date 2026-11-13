"""Paper figures, drawn only from results/*.json. ``.venv/bin/python paper/figures.py`` -> paper/figures/*.pdf|png

Palette: validated categorical slots 1-5 (blue, orange, aqua, yellow, magenta); markers differ per series
(secondary encoding), and every figure has a table with the same numbers in the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUT = HERE / "figures"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def table_numbers() -> dict:
    """Numbers of the generated tables (``make_tables.py``)."""
    return json.loads((RES / "rev1" / "stats" / "tables.json").read_text())


def fig_text_structure() -> None:
    """Figure 3: text representation x model, primary protocol (mean and standard deviation over the 15 runs)."""
    t = table_numbers()
    texts = [("none", "no text"), ("tfidf", "TF-IDF"), ("scibert", "SciBERT"), ("specter", "SPECTER")]
    models = [("mlp", "MLP (no graph)"), ("sage", "GraphSAGE"), ("gat", "GAT")]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
    for ax, (bib, title) in zip(axes, (("nobiblio", "without bibliometric features"),
                                       ("biblio", "with bibliometric features"))):
        for j, (m, lab) in enumerate(models):
            xs = [i + (j - 1) * 0.22 for i in range(len(texts))]
            mean = [t[f"rq5.{tx}.{bib}.{m}"]["mean"] for tx, _ in texts]
            std = [t[f"rq5.{tx}.{bib}.{m}"]["sd"] for tx, _ in texts]
            ax.errorbar(xs, mean, yerr=std, fmt=MARKERS[j], color=SLOTS[j], ecolor=INK2, elinewidth=0.8,
                        capsize=2, markersize=6, markeredgecolor="white", markeredgewidth=0.8, label=lab)
        ax.set_xticks(range(len(texts)), [tx for _, tx in texts])
        ax.set_title(title, fontsize=9, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Spearman ρ (test, k = 3)")
    axes[1].legend(loc="lower right", fontsize=8)
    save(fig, "fig_text_structure")


def fig_hetero() -> None:
    """Figure 1: paired difference in ROC-AUC with the MLP and its 95 % interval, for each protocol."""
    t = table_numbers()
    models = [("mlp+refagg_in", "MLP + in-graph ref. aggregates"), ("mlp+refagg", "MLP + all-reference aggregates"),
              ("sage", "GraphSAGE"), ("gat", "GAT"), ("han_temporal", "HAN (temporal rule)"),
              ("han_t0", "HAN ($T_0$ rule)"), ("hgt_t0", "HGT ($T_0$ rule)"), ("hgt_roll", "HGT (rolling rule)"),
              ("hgt_naive", "HGT (naive)")]
    protos = [("primary", "primary protocol, k = 3"), ("kfold3", "retrospective K-fold, k = 3"),
              ("kfold5", "retrospective K-fold, k = 5")]
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    ax.axvspan(-0.01, 0.01, color=INK2, alpha=0.12, lw=0)
    ax.axvline(0, color=INK2, lw=0.8)
    for j, (tag, lab) in enumerate(protos):
        ys, mean, lo, hi = [], [], [], []
        for i, (m, _) in enumerate(models):
            c = t.get(f"rq3.{tag}.{m}-mlp")
            if c is None:
                continue
            ys.append(i + (j - 1) * 0.25)
            mean.append(c["diff"])
            lo.append(c["diff"] - c["ci"][0])
            hi.append(c["ci"][1] - c["diff"])
        ax.errorbar(mean, ys, xerr=[lo, hi], fmt=MARKERS[j], color=SLOTS[j], ecolor=SLOTS[j], elinewidth=1.0,
                    capsize=2, markersize=5, markeredgecolor="white", markeredgewidth=0.6, label=lab)
    ax.set_yticks(range(len(models)), [lab for _, lab in models])
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC of the model − ROC-AUC of the MLP (paired, 95 % interval)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5, frameon=False)
    save(fig, "fig_hetero")


def fig_anomaly() -> None:
    """ROC-AUC against the injection intensity for four injected patterns (revision: 30 draws, seven patterns in
    the tables; the two submitted patterns and the two one-directional ones are drawn here)."""
    r = json.loads((RES / "rev1" / "anomaly.json").read_text())
    s, lv = r["summary"], r["intensities"]
    levels = list(lv)
    dets = [("heur_reciprocity", "reciprocity heuristic"), ("heur_self_ratio", "self-citation ratio"),
            ("null_excess", "null-model excess"), ("dominant_full", "DOMINANT"), ("gae_full", "graph autoencoder")]
    panels = [("cartel", "reciprocal cartel"), ("selfcit", "self-citation"), ("ring", "one-directional ring"),
              ("stacking", "citation stacking")]
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.6), sharey=True, sharex=True)
    for ax, (pat, title) in zip(axes.ravel(), panels):
        ax.axhline(0.5, color=INK2, lw=0.8, ls=(0, (4, 3)))
        for j, (d, lab) in enumerate(dets):
            mean = [s[pat][l][d]["roc_auc"]["mean"] for l in levels]
            std = [s[pat][l][d]["roc_auc"]["std"] for l in levels]
            ax.errorbar([lv[l] for l in levels], mean, yerr=std, fmt="-" + MARKERS[j], color=SLOTS[j], lw=1.6,
                        ecolor=INK2, elinewidth=0.8, capsize=2, markersize=5, markeredgecolor="white",
                        markeredgewidth=0.8, label=lab)
        ax.set_xticks([lv[l] for l in levels], [f"{lv[l]:.2f}" for l in levels])
        ax.set_title(title, fontsize=9, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("ROC-AUC")
    fig.supxlabel("injection intensity (share of the venue's outgoing citations)", fontsize=9, y=0.03)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_anomaly")


def fig_workflow() -> None:
    """Workflow of the study: (a) graph construction and evaluation, (b) citation windows of the labelled cohorts
    against the prediction date of each protocol."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    fig = plt.figure(figsize=(6.3, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.2)

    # ---- (a) pipeline
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")
    ax.grid(False)
    boxes = [
        (1, 25, "1  OpenAlex snapshot", "works, authors, venues,\nyearly citation counts"),
        (26, 25, "2  Sample", "48 topics × 250 papers,\n2017-2022, and seed papers"),
        (51, 25, "3  Reference selection", "at least 2 citing papers;\ncap, or citers before T only"),
        (76, 25, "4  Features before y", "authors, venue, references:\ncitations received before y"),
        (1, 3, "5  Graphs", "citation (reference → paper),\nheterogeneous, venue graph"),
        (26, 3, "6  Labels", "citations in years y .. y+k;\ntop 10 % of the cohort"),
        (51, 3, "7  Protocols", "mature labels at T (primary),\nretrospective, K-fold, holdout"),
        (76, 3, "8  Comparison", "tuned MLP and GNNs;\npaired differences, 95 % CI"),
    ]
    w, h = 23, 15
    for x, y, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc="#f3f6fb",
                                    ec=SLOTS[0], lw=0.9))
        ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="center", fontsize=7.6, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + 5.4, body, ha="center", va="center", fontsize=6.2, color=INK2, linespacing=1.25)
    arrow = dict(arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK2)
    for x in (24.4, 49.4, 74.4):
        ax.add_patch(FancyArrowPatch((x, 32.5), (x + 1.4, 32.5), **arrow))
        ax.add_patch(FancyArrowPatch((x, 10.5), (x + 1.4, 10.5), **arrow))
    ax.add_patch(FancyArrowPatch((87.5, 24.4), (87.5, 21.4), connectionstyle="arc3", **arrow))
    ax.plot([87.5, 87.5, 12.5, 12.5], [21.4, 21.4, 21.4, 21.4], color=INK2, lw=0.9)
    ax.add_patch(FancyArrowPatch((12.5, 21.4), (12.5, 18.6), **arrow))
    ax.text(0, 43, "(a)", fontsize=9, fontweight="bold", color=INK)

    # ---- (b) citation windows and prediction dates (k = 3)
    ax = fig.add_subplot(gs[1])
    cohorts = list(range(2017, 2023))
    role = {2017: "train", 2018: "train", 2019: "retro", 2020: "retro", 2021: "retro", 2022: "test"}
    colour = {"train": SLOTS[0], "retro": "#b9b8b3", "test": SLOTS[1]}
    for i, y in enumerate(cohorts):
        ax.add_patch(Rectangle((y, i - 0.32), 4, 0.64, fc=colour[role[y]], ec="white", lw=0.6, zorder=2))
        ax.plot([y], [i], marker="|", color=INK, markersize=9, markeredgewidth=1.2)
        ax.text(y + 2, i, f"window {y}-{y + 3}", ha="center", va="center", fontsize=7, zorder=6,
                color="white" if role[y] != "retro" else INK,
                bbox=dict(fc=colour[role[y]], ec="none", pad=1.0))     # the label stays readable under the T lines
    ax.set_axisbelow(True)
    ax.axvline(2022, color=INK, lw=1.2, zorder=4)
    ax.text(2022.08, 6.55, "prediction date T = 2022 (primary protocol)", fontsize=7, va="center", color=INK)
    ax.axvline(2021, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=4)
    ax.text(2020.92, 6.55, "T = 2021 of the retrospective split", fontsize=7, va="center", ha="right", color=INK2)
    ax.axvspan(2026, 2027, color="#efeeea", zorder=0)
    ax.text(2026.5, 2.5, "2026: incomplete year", rotation=90, ha="center", va="center", fontsize=6.6, color=INK2)
    ax.set_xlim(2016.6, 2027)
    ax.set_ylim(-0.7, 7.0)
    ax.set_yticks(range(len(cohorts)), [f"cohort {y}" for y in cohorts])
    ax.set_xticks(range(2017, 2027))
    ax.set_xlabel("calendar year (a bar covers the citation window y .. y+3 of the cohort published in year y)")
    ax.grid(axis="y", visible=False)
    handles = [Rectangle((0, 0), 1, 1, fc=colour[r]) for r in ("train", "retro", "test")]
    ax.legend(handles, ["training and validation labels (closed before T)",
                        "labels still open at T (retrospective protocols only)", "test cohort"],
              loc="upper center", fontsize=6.8, ncol=3, bbox_to_anchor=(0.46, -0.2), columnspacing=1.2, handlelength=1.4)
    ax.set_title("(b)", loc="left", fontsize=9, fontweight="bold", color=INK, x=-0.17)
    save(fig, "fig_workflow")


if __name__ == "__main__":
    fig_text_structure()
    fig_hetero()
    fig_anomaly()
    fig_workflow()
    print("figures written to", OUT)
