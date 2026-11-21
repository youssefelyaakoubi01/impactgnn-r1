"""Paper figures. ``.venv/bin/python paper/figures.py [name ...]`` -> paper/figures/*.pdf|png

The result figures are drawn only from results/*.json. The workflow figure has no data: its panel (a) is the UML
activity diagram ``figures/fig_workflow_a.puml``, rendered by PlantUML (``PLANTUML_JAR=/path/to/plantuml.jar`` with
Java, or a ``plantuml`` command) and Ghostscript, and set next to panel (b) with pdflatex.

Palette: validated categorical slots 1-5 (blue, orange, aqua, yellow, magenta); markers differ per series
(secondary encoding), and every figure has a table with the same numbers in the paper.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
    # one key for both panels, outside the data area and centred under the two axes; markers only, because the
    # error bars of the handles read as data points
    handles, labels = axes[0].get_legend_handles_labels()
    centre = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
    fig.legend([h[0] for h in handles], labels, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(centre, 0.0),
               columnspacing=2.4, handletextpad=0.3)
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


def plantuml_command() -> list[str] | None:
    """Command that runs PlantUML: the jar named by ``PLANTUML_JAR``, or ``plantuml`` on the PATH."""
    jar = os.environ.get("PLANTUML_JAR")
    if jar and Path(jar).expanduser().is_file() and shutil.which("java"):
        return ["java", "-Djava.awt.headless=true", "-jar", str(Path(jar).expanduser())]
    exe = shutil.which("plantuml")
    return [exe] if exe else None


def workflow_panel_a() -> Path:
    """Panel (a): ``fig_workflow_a.puml`` rendered by PlantUML (EPS, text as outlines) and cropped to a PDF by
    Ghostscript. Without PlantUML or Ghostscript the versioned ``fig_workflow_a.pdf`` is reused."""
    src, pdf = OUT / "fig_workflow_a.puml", OUT / "fig_workflow_a.pdf"
    cmd = plantuml_command()
    if cmd is None or not shutil.which("gs"):
        if not pdf.exists():
            raise SystemExit(f"{pdf.name} is missing and cannot be drawn: set PLANTUML_JAR (with java) or install "
                             "plantuml, and install Ghostscript")
        print(f"WARNING: PlantUML or Ghostscript not found; {pdf.name} is reused and NOT redrawn from {src.name}")
        return pdf
    eps = subprocess.run(cmd + ["-pipe", "-teps", "-charset", "UTF-8"], input=src.read_bytes(),
                         capture_output=True, check=True).stdout
    if not eps.startswith(b"%!PS") or b"Syntax Error" in eps:
        raise SystemExit(f"PlantUML could not render {src.name}")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.eps").write_bytes(eps)
        subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dEPSCrop", "-sDEVICE=pdfwrite", f"-sOutputFile={pdf}",
                        str(Path(tmp) / "a.eps")], check=True)
    return pdf


def workflow_panel_b(path: Path, width: float, height: float) -> None:
    """Panel (b): citation windows of the labelled cohorts against the prediction dates (k = 3); size in inches."""
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(width, height))
    cohorts = list(range(2017, 2023))
    role = {2017: "train", 2018: "train", 2019: "retro", 2020: "retro", 2021: "retro", 2022: "test"}
    colour = {"train": SLOTS[0], "retro": "#b9b8b3", "test": SLOTS[1]}
    for i, y in enumerate(cohorts):
        ax.add_patch(Rectangle((y, i - 0.32), 4, 0.64, fc=colour[role[y]], ec="white", lw=0.6, zorder=2))
        ax.plot([y], [i], marker="|", color=INK, markersize=9, markeredgewidth=1.2)
        ax.text(y + 2, i, f"window {y}-{y + 3}", ha="center", va="center", fontsize=6.6, zorder=6,
                color="white" if role[y] != "retro" else INK,
                bbox=dict(fc=colour[role[y]], ec="none", pad=1.0))     # the label stays readable under the T lines
    ax.set_axisbelow(True)
    ax.axvline(2022, color=INK, lw=1.2, zorder=4)
    ax.text(2022.12, 6.12, "prediction date T = 2022\n(primary protocol)", fontsize=6.6, va="center", color=INK,
            linespacing=1.15)
    ax.axvline(2021, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=4)
    ax.text(2020.88, 6.12, "T = 2021 of the\nretrospective split", fontsize=6.6, va="center", ha="right", color=INK2,
            linespacing=1.15)
    ax.axvspan(2026, 2027, color="#efeeea", zorder=0)
    ax.text(2026.5, 2.5, "2026: incomplete year", rotation=90, ha="center", va="center", fontsize=6.6, color=INK2)
    ax.set_xlim(2016.6, 2027)
    ax.set_ylim(-0.7, 6.7)
    ax.set_yticks(range(len(cohorts)), [f"cohort {y}" for y in cohorts], fontsize=7.5)
    ax.set_xticks(range(2017, 2027), [str(y) for y in range(2017, 2027)], fontsize=6.8)
    ax.set_xlabel("calendar year (a bar covers the citation window\ny .. y+3 of the cohort published in year y)",
                  fontsize=7.2, linespacing=1.2)
    ax.grid(axis="y", visible=False)
    handles = [Rectangle((0, 0), 1, 1, fc=colour[r]) for r in ("train", "retro", "test")]
    ax.legend(handles, ["training and validation labels (closed before T)",
                        "labels still open at T (retrospective protocols only)", "test cohort"],
              loc="upper center", fontsize=6.8, ncol=1, bbox_to_anchor=(0.42, -0.17), handlelength=1.4)
    fig.savefig(path)
    plt.close(fig)


WORKFLOW_TEX = r"""\documentclass[border=1pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{mathptmx,graphicx,xcolor}
\begin{document}\color{black}%% explicit: the article sets the float in red
\begin{tabular}{@{}l@{\hspace{%(gap).1fpt}}l@{}}
\fontsize{9}{11}\selectfont\textbf{(a)} & \fontsize{9}{11}\selectfont\textbf{(b)}\\[2pt]
\raisebox{-\height}{\includegraphics[scale=%(scale).3f]{a.pdf}} &
\raisebox{-\height}{\includegraphics[width=%(wb).1fpt]{b.pdf}}
\end{tabular}%%
\end{document}
"""


def fig_workflow() -> None:
    """Workflow of the study: (a) graph construction and evaluation, as a UML activity diagram drawn by PlantUML,
    (b) citation windows of the labelled cohorts against the prediction date of each protocol. The two panels are
    set side by side with LaTeX, at the text width of the article."""
    total, gap, scale = 470.0, 10.0, 0.62          # points: text width of the article, gutter; scale of panel (a)
    a = workflow_panel_a()
    box = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", a.read_bytes())
    wa, ha = (float(box.group(3)) - float(box.group(1))) * scale, (float(box.group(4)) - float(box.group(2))) * scale
    wb = total - gap - wa
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(a, tmp / "a.pdf")
        workflow_panel_b(tmp / "b.pdf", wb / 72, 0.9 * ha / 72)
        (tmp / "fig_workflow.tex").write_text(WORKFLOW_TEX % {"gap": gap, "scale": scale, "wb": wb})
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig_workflow.tex"], cwd=tmp,
                           capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("LaTeX failed on the workflow figure:\n" + r.stdout[-1500:])
        OUT.mkdir(exist_ok=True)
        shutil.copy(tmp / "fig_workflow.pdf", OUT / "fig_workflow.pdf")
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", "-r300", "-dTextAlphaBits=4",
                    "-dGraphicsAlphaBits=4", f"-sOutputFile={OUT / 'fig_workflow.png'}", str(OUT / "fig_workflow.pdf")],
                   check=True)


if __name__ == "__main__":
    FIGURES = {"text_structure": fig_text_structure, "hetero": fig_hetero, "anomaly": fig_anomaly,
               "workflow": fig_workflow}
    names = sys.argv[1:] or list(FIGURES)             # ``figures.py workflow`` redraws that figure only
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; choose among {list(FIGURES)}")
    for n in names:
        FIGURES[n]()
    print("figures written to", OUT)
