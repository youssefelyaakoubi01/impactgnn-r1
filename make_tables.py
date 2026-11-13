"""Tables of the revised article, generated from the result files of the revision.

``python make_tables.py [name ...]``   (no name: every table whose inputs exist)

Each table is written to ``results/rev1/tables/<name>.md`` (caption line and pipe table, as used in
``paper/axis1.md``) and its numbers to ``results/rev1/stats/tables.json``. ``paper/include_tables.py`` copies the
tables into the article between ``<!--table:name-->`` markers, so no number of a table is typed by hand.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd

import stats as S
from config import ROOT

RES = ROOT / "results" / "rev1"
OUT = RES / "tables"
NUMBERS: dict = {}


# ====================================================================== formatting
def f3(x: float) -> str:
    return f"{x:.3f}".replace("-", "−")


def pm(mean: float, sd: float) -> str:
    return f"{mean:.3f} ± {sd:.3f}".replace("-", "−")


def signed(x: float, nd: int = 3) -> str:
    r = round(x, nd)
    return ("+" if r > 0 else "−" if r < 0 else "") + f"{abs(r):.{nd}f}"


def diff_ci(c: dict) -> str:
    """Paired difference with its 95 % interval; ᵃ = equivalent within the margin, ᵇ = interval excludes zero."""
    mark = "ᵃ" if c["equivalent"] else ("ᵇ" if c["different"] else "")

    def bound(x: float) -> str:          # a bound that rounds to zero keeps one more digit, so that its sign is visible
        return signed(x, 4) if round(x, 3) == 0 and round(x, 4) != 0 else signed(x)

    return f"{signed(c['diff'])} [{bound(c['ci'][0])}, {bound(c['ci'][1])}]{mark}"


# Role of each table, printed at the end of its caption: the primary analyses were specified before the experiments of
# the revision were run; the holdout is confirmatory; the other tables are sensitivity or exploratory analyses.
PRIMARY = "Primary analysis."
MIXED = "Primary analysis for the rows of the primary protocol; the K-fold rows are retrospective."
EXPLORATORY = "Exploratory analysis."
ROLE = {"rq1": MIXED, "rq3": MIXED, "coldstart": "Primary analysis for the primary protocol; the other columns are "
        "retrospective.", "rq5": PRIMARY, "anomaly_auc": PRIMARY, "anomaly_ap": PRIMARY,
        "holdout": "Confirmatory analysis.", "lab": "",
        "hpo": "Model selection on validation data; no test result.", "topics_full": EXPLORATORY,
        "topics_full2": EXPLORATORY}

# Which contrasts of a table with a primary role are registered (adjusted with Holm's method); its other rows are not.
REGISTERED = {
    "rq1": "Registered contrasts (Holm-adjusted p-values in Section 5.10): GraphSAGE, GAT and MLP + all-reference "
           "aggregates against the MLP in the primary protocol; the other differences are exploratory.",
    "rq3": "Registered contrasts (Holm-adjusted p-values in Section 5.10): HGT (rolling rule) and GraphSAGE against "
           "the MLP in the primary protocol; the other differences are exploratory.",
    "coldstart": "None of these differences is a registered contrast.",
    "rq5": "None of these differences is a registered contrast.",
}

# Abbreviations defined under every table that uses them (pattern searched in the header and the cells).
GLOSSARY = [
    (r"\bMLP\b", "MLP: multilayer perceptron"),
    (r"\bGAT\b", "GAT: graph attention network"),
    (r"\bHAN\b", "HAN: heterogeneous graph attention network"),
    (r"\bHGT\b", "HGT: heterogeneous graph transformer"),
    (r"R²", "R²: coefficient of determination"),
    (r"\bMAE\b", "MAE: mean absolute error on the log scale"),
    (r"ROC-AUC", "ROC-AUC: area under the ROC curve"),
    (r"\bAP\b", "AP: average precision"),
    (r"P@10 %", "P@10 %: precision among the 10 % highest-scored papers"),
    (r"TF-IDF", "TF-IDF: term frequency–inverse document frequency"),
    (r"K-fold", "K-fold: 6-fold cross-validation repeated twice (retrospective)"),
    (r"Holm p", "Holm p: p-value adjusted with Holm's method"),
    (r"Wilcoxon p", "Wilcoxon p: p-value of the Wilcoxon signed-rank test over the topics"),
    (r"\*f\*", "*f*: injected share of the outgoing citations of a venue"),
    (r"\*k\*", "*k*: horizon in years"),
    (r"T0 rule", "T0 rule: author, venue and keyword nodes aggregate only the papers published before 2017"),
    (r"rolling", "rolling rule: year-specific author, venue and keyword nodes that aggregate only earlier papers"),
    (r"temporal rule", "temporal rule: a metapath neighbour is kept only if it is earlier than the target paper"),
    (r"naive", "naive: no temporal rule on the author, venue and keyword nodes"),
    (r"Combined", "Combined: the larger of the two heuristics, each expressed as a percentile rank"),
    (r"Δ", "Δ: paired difference (first model minus second model, or model minus reference)"),
]


def glossary(header: list[str], rows: list[list[str]]) -> str:
    body = " ".join(header) + " " + " ".join(c for r in rows for c in r)
    found = [d for pat, d in GLOSSARY if re.search(pat, body)]
    return "Abbreviations: " + "; ".join(found) + "." if found else ""


def write(name: str, caption: str, header: list[str], rows: list[list[str]], note: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    caption = (caption + " " + ROLE.get(name, EXPLORATORY)).strip()
    lines = [f"**Table @{name}.** {caption}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    if note.startswith(NOTE):                      # only the markers that the table uses
        body = " ".join(c for r in rows for c in r)
        parts = [t for mark, t in (("ᵃ", "ᵃ Equivalent: the 95 % interval lies within ±0.01."),
                                   ("ᵇ", "ᵇ Different: the 95 % interval excludes zero.")) if mark in body]
        if parts:
            other = "Other differences are inconclusive." if "ᵃ" in body and "ᵇ" in body else \
                "The other differences are inconclusive."
            parts.append(other)
        note = (" ".join(parts) + note[len(NOTE):]).strip()
    note = " ".join(x for x in (note, REGISTERED.get(name, ""), glossary(header, rows)) if x)
    if note:
        lines += ["", note]
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n")
    print(f"table {name}: {len(rows)} rows")


def record(name: str, value) -> None:
    NUMBERS[name] = value


# ====================================================================== RQ4: anomalies
PATTERN_NAMES = {"cartel": "Reciprocal cartel", "selfcit": "Self-citation", "cartel_rewire": "Budget-preserving cartel",
                 "cartel_large": "Large cartel", "cartel_camo": "Camouflaged cartel", "ring": "One-directional ring",
                 "stacking": "Citation stacking"}
DETECTORS = [("heur_reciprocity", "Reciprocity"), ("heur_self_ratio", "Self-citation ratio"), ("heur_max", "Combined"),
             ("null_excess", "Null-model excess"), ("dominant_full", "DOMINANT"), ("gae_full", "Graph autoencoder"),
             ("iforest_full", "Isolation Forest")]
LEVELS = [("mild", "0.10"), ("medium", "0.25"), ("strong", "0.50")]


def tables_anomaly() -> None:
    a = json.loads((RES / "anomaly.json").read_text())
    s = a["summary"]
    record("anomaly", {"draws": a["draws"], "prevalence": a["prevalence"], "summary": s,
                       "real_graph": a["real_graph"]})
    head = ["Injected pattern", "*f*"] + [d for _, d in DETECTORS]
    for metric, name, cap in (
            ("roc_auc", "anomaly_auc", "ROC-AUC of each detector for the injected venues against all other venues"),
            ("ap", "anomaly_ap", "Average precision of each detector at each injection intensity; the prevalence is "
                                 "the average precision of a random ranking")):
        rows, sd_max = [], 0.0
        for p, pname in PATTERN_NAMES.items():
            for lvl, f in LEVELS:
                vals = [s[p][lvl][d][metric] for d, _ in DETECTORS]
                sd_max = max(sd_max, max(v["std"] for v in vals))
                best = max(v["mean"] for v in vals)
                cells = [("**" + f3(v["mean"]) + "**") if v["mean"] == best else f3(v["mean"]) for v in vals]
                first = [pname if lvl == "mild" else ""]
                if metric == "ap":
                    first.append(f"{a['prevalence'][p]:.4f}" if lvl == "mild" else "")
                rows.append(first + [f] + cells)
        write(name, f"{cap} (mean over {a['draws']} draws; the largest standard deviation of a cell is "
                    f"{sd_max:.3f}; best detector of each row in bold). Learned detectors use content and "
                    f"structural features.", head if metric == "roc_auc" else head[:1] + ["Prevalence"] + head[1:], rows)
    rows = []
    for p, pname in PATTERN_NAMES.items():
        for lvl, f in (("medium", "0.25"), ("strong", "0.50")):
            vals = [s[p][lvl][d]["roc_auc_matched"]["mean"] for d, _ in DETECTORS]
            rows.append([pname if lvl == "medium" else "", f] + [("**" + f3(v) + "**") if v == max(vals) else f3(v)
                                                                 for v in vals])
    write("anomaly_matched", f"ROC-AUC of each detector for the injected venues against controls matched on size "
                             f"(mean over {a['draws']} draws; best detector of each row in bold).", head, rows)
    rng = lambda det, pats, metric="roc_auc": [  # noqa: E731
        round(min(s[p][l][det][metric]["mean"] for p in pats for l, _ in LEVELS), 3),
        round(max(s[p][l][det][metric]["mean"] for p in pats for l, _ in LEVELS), 3)]
    allp, recip = list(PATTERN_NAMES), ["cartel", "cartel_rewire", "cartel_large", "cartel_camo"]
    record("anomaly_ranges", {
        "reciprocity_reciprocal_patterns": rng("heur_reciprocity", recip),
        "reciprocity_ring_stacking": rng("heur_reciprocity", ["ring", "stacking"]),
        "self_ratio_selfcit": rng("heur_self_ratio", ["selfcit"]),
        "self_ratio_other_patterns": rng("heur_self_ratio", [p for p in allp if p != "selfcit"]),
        "null_excess_all": rng("null_excess", allp), "null_excess_strong": [
            round(min(s[p]["strong"]["null_excess"]["roc_auc"]["mean"] for p in allp), 3),
            round(max(s[p]["strong"]["null_excess"]["roc_auc"]["mean"] for p in allp), 3)],
        "dominant_all": rng("dominant_full", allp), "gae_all": rng("gae_full", allp),
        "iforest_all": rng("iforest_full", allp),
        "content_only_all": [min(rng(d, allp)[0] for d in ("dominant_content", "gae_content", "iforest_content")),
                             max(rng(d, allp)[1] for d in ("dominant_content", "gae_content", "iforest_content"))],
        "ap_max": max(s[p][l][d]["ap"]["mean"] for p in allp for l, _ in LEVELS for d, _ in DETECTORS),
        "dominant_matched_strong": [
            round(min(s[p]["strong"]["dominant_full"]["roc_auc_matched"]["mean"] for p in allp), 3),
            round(max(s[p]["strong"]["dominant_full"]["roc_auc_matched"]["mean"] for p in allp), 3)],
        "best_learned_count": sum(max(DETECTORS[4:], key=lambda d: s[p][l][d[0]]["roc_auc"]["mean"])[0] ==
                                  max(DETECTORS, key=lambda d: s[p][l][d[0]]["roc_auc"]["mean"])[0]
                                  for p in allp for l, _ in LEVELS)})
    ext = RES / "anomaly_external.json"
    if ext.exists():
        e = json.loads(ext.read_text())
        record("anomaly_external", e)
        rows = [[name, f3(e["detectors"][d]["roc_auc"]), f"{e['detectors'][d]['p_perm']:.3f}",
                 f3(e["detectors"][d]["roc_auc_matched"]), f"{e['detectors'][d]['median_percentile']:.2f}",
                 str(e["detectors"][d]["in_top_decile"])] for d, name in DETECTORS]
        write("anomaly_external",
              f"Descriptive check on the real venue graph: rank of the {e['n_suppressed_in_graph']} venues that were "
              f"suppressed from the Journal Citation Reports for anomalous citation patterns (2018–2025 releases) "
              f"among the {e['n_venues']:,} venues. The p-value comes from 5,000 permutations of the labels.",
              ["Detector", "ROC-AUC", "p", "ROC-AUC against size-matched controls",
               "Median percentile of the suppressed venues", "Suppressed venues in the top decile"], rows)


# ====================================================================== node tasks: helpers
KF = ("kfold", "topicfold")
NAMES = {"mlp": "MLP", "mlp+refagg_in": "MLP + in-graph aggregates",
         "mlp+refagg": "MLP + all-reference aggregates", "sage": "GraphSAGE",
         "sage_e": "GraphSAGE + edge features", "sage+refagg": "GraphSAGE + all-reference aggregates",
         "gat": "GAT", "gat_e": "GAT + edge features", "gat+refagg": "GAT + all-reference aggregates",
         "han_temporal": "HAN (temporal rule)", "han_t0": "HAN (T0 rule)", "hgt_t0": "HGT (T0 rule)",
         "hgt_roll": "HGT (rolling rule)", "hgt_naive": "HGT (naive)"}
_RUNS: dict = {}


PENDING: set = set()


def runs(*grids: str) -> pd.DataFrame:
    key = "+".join(grids)
    if key not in _RUNS:
        parts = []
        for g in grids:
            try:
                d = S.runs(g)
            except FileNotFoundError:          # grid not started yet: its cells stay empty
                d = pd.DataFrame()
            if d.empty:
                PENDING.add(g)
            else:
                parts.append(d)
        cols = ["arm", "task", "k", "protocol", "id", "rep", "fold", "train_subsample", "test_group"]
        _RUNS[key] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    return _RUNS[key]


def sel(df: pd.DataFrame, **kw) -> pd.DataFrame:
    for k, v in kw.items():
        df = df[df[k] == v]
    return df


def ms(d: pd.DataFrame, metric: str) -> tuple[float, float, int]:
    """Mean and standard deviation of a test metric: over test folds (seed-averaged) for K-fold, over runs else."""
    col = f"m_{metric}"
    v = d.groupby(["rep", "fold"])[col].mean() if d.protocol.iloc[0] in KF else d[col]
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, int(len(v))


def cell(df: pd.DataFrame, arm: str, metric: str, key: str | None = None) -> str:
    d = df[df.arm == arm]
    if d.empty:
        return "–"
    m, sd, n = ms(d, metric)
    expected = {"mature": 15, "kfold": 12, "topicfold": 12}.get(d.protocol.iloc[0])
    if expected and n != expected:
        PENDING.add(f"{arm} ({d.protocol.iloc[0]}): {n} of {expected} runs or folds")
    if key:
        record(key, {"mean": m, "sd": sd, "n": n})
    return pm(m, sd)


_CACHE_FILE = RES / "stats" / "contrasts_cache.json"
_CACHE: dict = json.loads(_CACHE_FILE.read_text()) if _CACHE_FILE.exists() else {}


def con(df: pd.DataFrame, a: str, b: str, metric: str, key: str | None = None) -> str:
    """Paired contrast between two arms. The bootstrap is slow, so a contrast is stored with the identifiers of its
    runs and computed again only when the runs change."""
    da, db = df[df.arm == a], df[df.arm == b]
    if da.empty or db.empty:
        return "–"
    ck = hashlib.sha1("|".join([metric] + sorted(da.id) + ["vs"] + sorted(db.id)).encode()).hexdigest()
    if ck not in _CACHE:
        try:
            _CACHE[ck] = S.contrast(df, a, b, metric)
        except AssertionError as exc:          # a grid that is still running: the two arms are not paired yet
            PENDING.add(f"{a} vs {b}: {exc}")
            return "pending"
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(_CACHE, default=float))
    if key:
        record(key, _CACHE[ck])
    return diff_ci(_CACHE[ck])


AGG = (' "Aggregates": seven summary statistics of the references (Section 4.3), computed over the in-graph '
       "references or over all references.")
NOTE = ("ᵃ Equivalent: the 95 % interval lies within ±0.01. ᵇ Different: the 95 % interval excludes zero. "
        "Other differences are inconclusive.")


# ====================================================================== RQ1
PROTO_NAMES = (("primary", "Primary (15 runs)"),
               ("kfold3", "K-fold, *k* = 3 (12 folds)"),
               ("kfold5", "K-fold, *k* = 5 (12 folds)"))


def tables_rq1() -> None:
    kf = runs("kfold_main")
    data = {"primary": sel(runs("mature_main"), task="reg"), "kfold3": sel(kf, task="reg", k=3),
            "kfold5": sel(kf, task="reg", k=5)}
    arms = ["mlp", "mlp+refagg_in", "mlp+refagg", "sage", "sage_e", "gat", "gat_e"]
    rows = []
    for tag, pname in PROTO_NAMES:
        d = data[tag]
        for a in arms:
            rows.append([pname if a == "mlp" else "", NAMES[a], cell(d, a, "spearman", f"rq1.{tag}.{a}.spearman"),
                         cell(d, a, "r2", f"rq1.{tag}.{a}.r2"), cell(d, a, "mae", f"rq1.{tag}.{a}.mae"),
                         "reference" if a == "mlp" else con(d, a, "mlp", "spearman", f"rq1.{tag}.{a}-mlp")])
    # validation papers of the primary protocol belong to the training cohorts: their metric shows what the models
    # reach without the change of cohort
    val = {a: [float(x) for x in (data["primary"][data["primary"].arm == a].v_spearman.mean(),
                                  data["primary"][data["primary"].arm == a].v_spearman.std(ddof=1))] for a in arms}
    record("rq1.primary.validation", val)
    write("rq1", "Regression of log(1 + citations): mean ± std over runs (primary protocol) or over test folds "
                 "(retrospective K-fold). Δ is the paired difference in Spearman correlation with the MLP and its "
                 "95 % interval. Primary: labels closed at the prediction date; K-fold: retrospective. Tuned hyperparameters, "
                 "SciBERT text.",
          ["Protocol", "Model", "Spearman", "R²", "MAE", "Δ Spearman vs MLP"], rows, NOTE + AGG)


def tables_attention() -> None:
    from experiments_rev1 import read_runs

    acc: dict = {}
    for r in read_runs("kfold_attention"):
        j = r["job"]
        for kind in ("trained", "untrained", "trained_layer1"):
            a = r["attention"][kind]
            if a.get("n_targets", 0) == 0:
                continue
            key = (j["model"], j["k"], kind)
            acc.setdefault(key, []).append((j["rep"], j["fold"], a["per_paper_cit"]["mean"],
                                            a["per_paper_cit"]["share_positive"], a["per_paper_age"]["mean"],
                                            a["per_paper_age"]["share_positive"], a["n_targets"]))
    rows = []
    selfw = pd.DataFrame([(r["job"]["model"], r["job"]["k"], r["job"]["rep"], r["job"]["fold"],
                           r["attention"]["self_attention_test_mean"]) for r in read_runs("kfold_attention")],
                         columns=["model", "k", "rep", "fold", "w"])
    for (model, k), d in selfw.groupby(["model", "k"]):      # share of the attention on the self-loop, per test fold
        f = d.groupby(["rep", "fold"]).w.mean()
        record(f"attention.{model}.k{k}.self", [float(f.mean()), float(f.std(ddof=1))])
    for model, mname in (("gat", "GAT"), ("gat_e", "GAT + edge features")):
        for k in (3, 5):
            for kind, kname in (("trained", "trained"), ("untrained", "untrained")):
                d = pd.DataFrame(acc.get((model, k, kind), []), columns=["rep", "fold", "c", "cp", "a", "ap", "n"])
                if d.empty:
                    continue
                f = d.groupby(["rep", "fold"]).mean()
                record(f"attention.{model}.k{k}.{kind}", {c: [float(f[c].mean()), float(f[c].std(ddof=1))]
                                                         for c in ("c", "cp", "a", "ap", "n")})
                rows.append([mname, str(k), kname, pm(f.c.mean(), f.c.std(ddof=1)), f"{f.cp.mean():.2f}",
                             pm(f.a.mean(), f.a.std(ddof=1)), f"{f.ap.mean():.2f}", f"{f.n.mean():,.0f}"])
    write("attention", "Spearman correlation between the last-layer attention of GAT and the attributes of the "
                       "references, computed within each citing test paper with at least three in-graph references "
                       "(mean over papers, then mean ± std over 12 test folds of the retrospective K-fold).",
          ["Model", "*k*", "Network", "Correlation with prior citations", "Share of papers with a positive correlation",
           "Correlation with reference age", "Share positive (age)", "Test papers per fold"], rows)


# ====================================================================== RQ2
CS_NAMES = {"mlp_authors": "MLP: authors", "mlp_refagg": "MLP: reference aggregates",
            "mlp_authors_refagg": "MLP: authors + reference aggregates",
            "sage_refs": "GraphSAGE: references only (own text hidden)",
            "sage_refs_authors": "GraphSAGE: references + authors (own text hidden)",
            "sage_refs_authors_text": "GraphSAGE: references + authors + own text",
            "sage_full": "GraphSAGE: all features, edge features", "sage_plain_full": "GraphSAGE: all features",
            "mlp_full": "MLP: all features + reference aggregates"}


def tables_coldstart() -> None:
    pri = runs("mature_coldstart")
    retro = sel(runs("retro"), task="reg")
    r3, r5 = sel(retro, k=3), sel(retro, k=5)
    r3 = r3[r3.train_subsample.isna()] if "train_subsample" in r3 else r3
    # in the primary protocol the three complete variants are the models of the main grid: the same runs are used,
    # so that a configuration has one value in all tables
    same = {"cs|sage_full": "sage_e", "cs|sage_plain_full": "sage", "cs|mlp_full": "mlp+refagg"}
    main = sel(runs("mature_main"), task="reg")
    pri = pd.concat([pri[~pri.arm.isin(same)]] + [main[main.arm == a].assign(arm=c) for c, a in same.items()],
                    ignore_index=True)
    rows = [[CS_NAMES[v], cell(pri, f"cs|{v}", "spearman", f"cs.primary.{v}"),
             "reference" if v == "mlp_full" else con(pri, f"cs|{v}", "cs|mlp_full", "spearman", f"cs.primary.{v}-mlp_full"),
             cell(r3, f"cs|{v}", "spearman", f"cs.retro3.{v}"), cell(r5, f"cs|{v}", "spearman", f"cs.retro5.{v}")]
            for v in CS_NAMES]
    write("coldstart", "Cold start: test papers absent from the training graph (Spearman correlation). Primary "
                       "protocol: *k* = 3, labels closed before the prediction date, mean ± std over 15 runs. "
                       "Retrospective temporal split of the initial version: mean ± std over 5 seeds.",
          ["Variant", "Primary protocol", "Primary: Δ vs MLP, all features", "Retrospective, *k* = 3",
           "Retrospective, *k* = 5"], rows, NOTE)
    sub = sel(runs("retro"), task="reg", k=3)
    if "train_subsample" in sub:
        full, small = sub[sub.train_subsample.isna()], sub[sub.train_subsample.notna()]
        ctrl = {}
        for a in ("mlp", "sage", "mlp+refagg"):
            if not full[full.arm == a].empty and not small[small.arm == f"{a}|sub2446"].empty:
                ctrl[a] = {"full": ms(full[full.arm == a], "spearman")[:2],
                           "subsampled": ms(small[small.arm == f"{a}|sub2446"], "spearman")[:2]}
        record("cs.retro3.train_size_control", ctrl)


def tables_lab() -> None:
    d = runs("deploy")
    nt = S.node_table()
    out = {}
    for k in (3, 5):
        for arm in ("deploy|sage_full", "deploy|mlp_full"):
            ids = None
            ps = []
            for i in sel(d, k=k, arm=arm).id:
                z = S.preds(i)
                ids, _ = z["te_ids"], ps.append(z["te_p"])
            if ids is not None:
                out[(k, arm)] = pd.Series(np.mean(ps, 0), index=ids)
    lab = nt[nt.origin == "seed_lab"]
    rows, recent = [], []
    mlp3, sage3 = out.get((3, "deploy|mlp_full")), out.get((3, "deploy|sage_full"))
    for pid, r in lab.sort_values(["year", "title"]).iterrows():
        if mlp3 is None or pid not in mlp3.index:
            continue
        title = r.title[:70] + ("…" if len(r.title) > 70 else "")
        pm_, ps_ = float(np.expm1(mlp3[pid])), float(np.expm1(sage3[pid]))
        if 2017 <= r.year and r.year + 3 <= 2025:          # labelled cohort with a complete window
            rows.append([title, str(int(r.year)), str(int(r.c3)), f"{pm_:.1f}", f"{ps_:.1f}"])
        elif r.year > 2022:             # counted only: the article does not rank the recent laboratory papers
            recent.append(pid)
    record("lab", {"n_lab": int(len(lab)), "n_predicted": int(sum(p in mlp3.index for p in lab.index)) if mlp3 is not None else 0,
                   "n_complete_k3": len(rows), "rows": rows, "n_recent": len(recent)})
    write("lab", "Laboratory papers with a complete three-year window: actual citations and citations predicted by "
                 "the models trained on all labelled years (mean over 5 seeds). Retrospective and illustrative.",
          ["Paper", "Year", "Actual citations (*k* = 3)", "MLP, all features", "GraphSAGE, all features"], rows)


# ====================================================================== RQ3
def tables_rq3() -> None:
    kf = runs("kfold_main")
    data = {"primary": sel(runs("mature_main"), task="cls"), "kfold3": sel(kf, task="cls", k=3),
            "kfold5": sel(kf, task="cls", k=5)}
    arms = ["mlp", "mlp+refagg_in", "mlp+refagg", "sage", "gat", "han_temporal", "han_t0", "hgt_t0", "hgt_roll",
            "hgt_naive"]
    rows = []
    for tag, pname in PROTO_NAMES:
        d = data[tag]
        for a in arms:
            if d[d.arm == a].empty:
                continue
            rows.append([pname if a == "mlp" else "", NAMES[a], cell(d, a, "roc_auc", f"rq3.{tag}.{a}.roc_auc"),
                         cell(d, a, "ap", f"rq3.{tag}.{a}.ap"), cell(d, a, "p_at_10", f"rq3.{tag}.{a}.p_at_10"),
                         "reference" if a == "mlp" else con(d, a, "mlp", "roc_auc", f"rq3.{tag}.{a}-mlp")])
    write("rq3", "Detection of top-10 % papers: mean ± std over runs (primary protocol) or over test folds "
                 "(retrospective K-fold). AP: average precision; P@10 %: precision among the 10 % highest scores. Δ is "
                 "the paired difference in ROC-AUC with the MLP and its 95 % interval. Tuned hyperparameters, SciBERT "
                 "text.",
          ["Protocol", "Model", "ROC-AUC", "AP", "P@10 %", "Δ ROC-AUC vs MLP"], rows, NOTE + AGG + " The naive HGT lets "
          "every paper send messages to its authors, venue and keywords.")
    d = data["primary"]
    record("rq3.primary.validation", {a: [float(d[d.arm == a].v_roc_auc.mean()), float(d[d.arm == a].v_roc_auc.std(ddof=1))]
                                      for a in arms if not d[d.arm == a].empty})
    for tag in data:
        con(data[tag], "hgt_roll", "hgt_t0", "roc_auc", f"rq3.{tag}.hgt_roll-hgt_t0")
        con(data[tag], "hgt_naive", "hgt_roll", "roc_auc", f"rq3.{tag}.hgt_naive-hgt_roll")
        con(data[tag], "hgt_naive", "hgt_t0", "roc_auc", f"rq3.{tag}.hgt_naive-hgt_t0")
        con(data[tag], "hgt_roll", "sage", "roc_auc", f"rq3.{tag}.hgt_roll-sage")
        con(data[tag], "hgt_roll", "mlp+refagg_in", "roc_auc", f"rq3.{tag}.hgt_roll-mlp+refagg_in")
        con(data[tag], "hgt_roll", "mlp+refagg", "roc_auc", f"rq3.{tag}.hgt_roll-mlp+refagg")
        con(data[tag], "sage", "mlp+refagg_in", "roc_auc", f"rq3.{tag}.sage-mlp+refagg_in")
        con(data[tag], "han_temporal", "han_t0", "roc_auc", f"rq3.{tag}.han_temporal-han_t0")


# ====================================================================== RQ5
TEXT_NAMES = {"none": "No text", "tfidf": "TF-IDF", "scibert": "SciBERT", "specter": "SPECTER"}


def tables_rq5() -> None:
    for grid, name, cap in (("mature_text", "rq5", "primary protocol, mean ± std over 15 runs"),
                            ("kfold_text", "rq5_kfold", "retrospective K-fold, mean ± std over 12 test folds")):
        for task, metric, suffix in (("reg", "spearman", ""), ("cls", "roc_auc", "_cls")):
            df = sel(runs(grid), task=task)
            rows = []
            for bib in ("nobiblio", "biblio"):
                for t, tn in TEXT_NAMES.items():
                    a = lambda m: f"{m}|{t}|{bib}"  # noqa: E731
                    pre = f"{name}{suffix}.{t}.{bib}"
                    rows.append(["yes" if bib == "biblio" else "no", tn,
                                 cell(df, a("mlp"), metric, f"{pre}.mlp"), cell(df, a("sage"), metric, f"{pre}.sage"),
                                 cell(df, a("gat"), metric, f"{pre}.gat"),
                                 con(df, a("sage"), a("mlp"), metric, f"{pre}.sage-mlp"),
                                 con(df, a("gat"), a("mlp"), metric, f"{pre}.gat-mlp")])
            mname = "Spearman correlation" if task == "reg" else "ROC-AUC for the top-10 % class"
            write(f"{name}{suffix}", f"Text × structure ablation at *k* = 3 ({mname}; {cap}). Δ is the paired "
                                     f"difference between the graph model and the MLP with the same inputs.",
                  ["Author and venue features", "Text", "MLP", "GraphSAGE", "GAT", "Δ GraphSAGE − MLP", "Δ GAT − MLP"],
                  rows, NOTE)
            for m in ("mlp", "sage", "gat"):          # contribution of the text and of the bibliometric features
                for bib in ("nobiblio", "biblio"):
                    for t in ("tfidf", "scibert", "specter"):
                        con(df, f"{m}|{t}|{bib}", f"{m}|none|{bib}", metric, f"{name}{suffix}.{m}.{bib}.{t}-none")
                    con(df, f"{m}|specter|{bib}", f"{m}|scibert|{bib}", metric, f"{name}{suffix}.{m}.{bib}.specter-scibert")
                for t in TEXT_NAMES:
                    con(df, f"{m}|{t}|biblio", f"{m}|{t}|nobiblio", metric, f"{name}{suffix}.{m}.{t}.biblio-nobiblio")


# ====================================================================== text dimension
def tables_textdim() -> None:
    kf = runs("kfold_pca")
    rows = []
    for task, metric, models in (("reg", "spearman", ("mlp", "sage", "gat")),
                                 ("cls", "roc_auc", ("mlp", "han_temporal", "hgt_t0", "hgt_roll"))):
        d = sel(kf, task=task)
        for m in models:
            rows.append([NAMES[m], "Spearman" if task == "reg" else "ROC-AUC"] +
                        [cell(d, f"{m}|{t}|d{dim}", metric, f"textdim.kfold.{m}.{t}.d{dim}")
                         for t in ("specter", "scibert") for dim in (768, 64)])
    write("textdim", "Text dimension in the retrospective K-fold at *k* = 3, with the hyperparameters of the "
                     "initial version (mean ± std over 12 test folds). The 64 principal components are fitted on "
                     "the training graph of each fold.",
          ["Model", "Metric", "SPECTER, 768 dimensions", "SPECTER, 64 components", "SciBERT, 768 dimensions",
           "SciBERT, 64 components"], rows)
    pri = runs("mature_pca")
    rows = []
    for label, df in (("K-fold, initial hyperparameters", sel(kf, task="cls")),
                      ("Primary protocol, tuned hyperparameters", sel(pri, task="cls"))):
        tag = "kfold" if label.startswith("K-fold") else "primary"
        for t in ("specter", "scibert"):
            for h in ("hgt_t0", "hgt_roll", "han_temporal"):
                c768 = con(df, f"{h}|{t}|d768", f"mlp|{t}|d768", "roc_auc", f"textdim.{tag}.{h}-mlp.{t}.d768")
                c64 = con(df, f"{h}|{t}|d64", f"mlp|{t}|d64", "roc_auc", f"textdim.{tag}.{h}-mlp.{t}.d64")
                if c768 != "–" or c64 != "–":
                    rows.append([label, TEXT_NAMES[t], f"{NAMES[h]} − MLP", c768, c64])
    d = sel(kf, task="cls")
    if not d[d.arm == "hgt_t0|specter|d64|fit_all"].empty:
        rows.append(["K-fold, initial hyperparameters", "SPECTER, components fitted on all nodes (as in the initial version)",
                     "HGT (T0 rule) − MLP", "–",
                     con(d, "hgt_t0|specter|d64|fit_all", "mlp|specter|d64|fit_all", "roc_auc",
                         "textdim.kfold.hgt_t0-mlp.specter.d64.fit_all")])
    write("textdim_pairs", "Paired differences in ROC-AUC between the heterogeneous models and the MLP on the same "
                           "partitions, with 768-dimensional text and with 64 components (95 % intervals).",
          ["Protocol", "Text", "Contrast", "768 dimensions", "64 components"], rows, NOTE)
    d = sel(kf, task="reg")
    for m in ("sage", "gat"):
        for t in ("specter", "scibert"):
            for dim in (768, 64):
                con(d, f"{m}|{t}|d{dim}", f"mlp|{t}|d{dim}", "spearman", f"textdim.kfold.{m}-mlp.{t}.d{dim}")
    for task, metric in (("reg", "spearman"), ("cls", "roc_auc")):
        dk, dp = sel(kf, task=task), sel(pri, task=task)
        for df, tag in ((dk, "kfold"), (dp, "primary")):
            for t in ("specter", "scibert"):
                con(df, f"mlp|{t}|d64", f"mlp|{t}|d768", metric, f"textdim.{tag}.mlp.d64-d768.{t}.{task}")


# ====================================================================== bibliometric statistics and audit
BIB_NAMES = {"exact": "Exact (before publication)", "strict": "Strict (no citation counts)",
             "hind": "Initial version (hindsight)"}


def tables_biblio() -> None:
    pri, kf = runs("mature_biblio"), runs("kfold_biblio")
    rows = []
    for df, tag, pname in ((pri, "primary", "Primary (15 runs)"), (kf, "kfold", "K-fold (12 folds)")):
        first = True
        for text in ("none", "scibert"):
            for bib, bname in (("exact", "Exact"), ("strict", "Strict"), ("hind", "Initial")):
                row = [pname if first else "", TEXT_NAMES[text], bname]
                first = False
                for task, metric in (("reg", "spearman"), ("cls", "roc_auc")):
                    for m in ("mlp", "sage"):
                        row.append(cell(sel(df, task=task), f"{m}|{text}|{bib}", metric,
                                        f"biblio.{tag}.{task}.{m}.{text}.{bib}"))
                    if task == "reg":       # effect of the statistics on the MLP, paired with the exact ones
                        row.insert(4, "reference" if bib == "exact" else con(
                            sel(df, task="reg"), f"mlp|{text}|{bib}", f"mlp|{text}|exact", "spearman",
                            f"biblio.{tag}.reg.mlp.{text}.{bib}-exact"))
                rows.append(row)
    write("biblio", "Author and venue statistics at *k* = 3 (mean ± std over runs or over test folds). Exact: "
                    "citations received before publication; strict: no citation counts; initial: statistics of the "
                    "initial version, which contain hindsight. Δ is the paired difference of the MLP with the same "
                    "model on the exact statistics.",
          ["Protocol", "Text", "Variant", "MLP: Spearman", "MLP: Δ Spearman vs exact",
           "GraphSAGE: Spearman", "MLP: ROC-AUC", "GraphSAGE: ROC-AUC"], rows, NOTE)
    for df, tag in ((pri, "primary"), (kf, "kfold")):
        for task, metric in (("reg", "spearman"), ("cls", "roc_auc")):
            d = sel(df, task=task)
            for text in ("none", "scibert"):
                for m in ("mlp", "sage"):
                    con(d, f"{m}|{text}|hind", f"{m}|{text}|exact", metric, f"biblio.{tag}.{task}.{m}.{text}.hind-exact")
                    con(d, f"{m}|{text}|strict", f"{m}|{text}|exact", metric,
                        f"biblio.{tag}.{task}.{m}.{text}.strict-exact")
                for bib in BIB_NAMES:
                    con(d, f"sage|{text}|{bib}", f"mlp|{text}|{bib}", metric, f"biblio.{tag}.{task}.sage-mlp.{text}.{bib}")
    d = sel(kf, task="cls")
    for bib in BIB_NAMES:
        cell(d, f"hgt_t0|scibert|{bib}", "roc_auc", f"biblio.kfold.cls.hgt_t0.scibert.{bib}")
        con(d, f"hgt_t0|scibert|{bib}", f"mlp|scibert|{bib}", "roc_auc", f"biblio.kfold.cls.hgt_t0-mlp.scibert.{bib}")
    # removal of metadata whose state at the prediction date is uncertain
    df = runs("mature_main", "mature_audit")
    rows = []
    for arm, name, ref in (("mlp", "MLP, all features", None), ("mlp|no_author", "MLP without author features", "mlp"),
                           ("mlp|no_venue", "MLP without venue features", "mlp"),
                           ("mlp|base_only", "MLP, base features and text only", "mlp"),
                           ("sage", "GraphSAGE, all features", None),
                           ("sage|no_author", "GraphSAGE without author features", "sage"),
                           ("hgt_roll", "HGT (rolling rule), all node types", None),
                           ("hgt_roll|no_keyword", "HGT without keyword nodes", "hgt_roll"),
                           ("hgt_roll|no_author", "HGT without author nodes and features", "hgt_roll"),
                           ("hgt_roll|papers_venues", "HGT with papers and venues only", "hgt_roll")):
        row = [name]
        for task, metric in (("reg", "spearman"), ("cls", "roc_auc")):
            d = sel(df, task=task)
            c = cell(d, arm, metric, f"audit.{task}.{arm}")
            row += [c, "–" if c == "–" else "reference" if ref is None
                    else con(d, arm, ref, metric, f"audit.{task}.{arm}-{ref}")]
        rows.append(row)
    write("audit_runs", "Sensitivity to metadata whose state at the prediction date is uncertain (primary protocol, "
                        "*k* = 3, mean ± std over 15 runs). Δ is the paired difference with the same model on all "
                        "features.",
          ["Configuration", "Spearman", "Δ Spearman", "ROC-AUC", "Δ ROC-AUC"], rows, NOTE)


# ====================================================================== graphs, coverage and feature parity
def tables_graphs() -> None:
    df = runs("mature_main", "mature_graphs")
    gnames = {"submitted": "Initial (all years, cap by identifier)", "cut2022": "Restricted to 2022, strict",
              "cut2022lso": "Restricted to 2022, leave-self-out", "cut2022sym": "Restricted to 2022 (primary)"}
    reg, cls = sel(df, task="reg"), sel(df, task="cls")
    rows = [["None (the MLP does not use the graph)", "MLP", "Spearman", cell(reg, "mlp", "spearman", "graphs.mlp.spearman"),
             "reference"],
            ["", "MLP", "ROC-AUC", cell(cls, "mlp", "roc_auc", "graphs.mlp.roc_auc"), "reference"]]
    for g, gn in gnames.items():
        for i, (m, d, metric, mn) in enumerate((("mlp+refagg_in", reg, "spearman", "Spearman"),
                                                ("sage", reg, "spearman", "Spearman"), ("gat", reg, "spearman", "Spearman"),
                                                ("hgt_roll", cls, "roc_auc", "ROC-AUC"))):
            a_ = m if (g == "cut2022sym" and not d[d.arm == m].empty) else f"{m}@{g}"   # runs of the main grid
            rows.append([gn if i == 0 else "", NAMES[m], mn, cell(d, a_, metric, f"graphs.{g}.{m}"),
                         con(d, a_, "mlp", metric, f"graphs.{g}.{m}-mlp")])
    write("graphs", "Reference selection and the primary protocol (*k* = 3, mean ± std over 15 runs). Δ is the paired "
                    "difference with the MLP, which does not use the graph.",
          ["Graph", "Model", "Metric", "Test value", "Δ vs MLP"], rows, NOTE)

    kf = sel(runs("kfold_main", "kfold_caps"), task="reg", k=3)
    rows = []
    for g, gn in (("cap30k", "Cap 30,000"), ("cap45k", "Cap 45,000"), ("cap60k", "Cap 60,000 (ties by hash)"),
                  ("submitted", "Initial graph"), ("uncapped", "Uncapped")):
        a = (lambda m: m) if g == "submitted" else (lambda m: f"{m}@{g}")  # noqa: E731
        rows.append([gn, cell(kf, a("mlp+refagg_in"), "spearman", f"caps.{g}.mlp+refagg_in"),
                     cell(kf, a("sage"), "spearman", f"caps.{g}.sage"),
                     con(kf, a("sage"), "mlp", "spearman", f"caps.{g}.sage-mlp"),
                     cell(kf, a("gat"), "spearman", f"caps.{g}.gat"), con(kf, a("gat"), "mlp", "spearman", f"caps.{g}.gat-mlp")])
    write("caps", "Sensitivity to the node cap in the retrospective K-fold (*k* = 3, Spearman correlation, mean ± "
                  "std over 12 test folds). Δ is the paired difference with the MLP, which does not use the graph.",
          ["Graph", "MLP + in-graph aggregates", "GraphSAGE", "Δ GraphSAGE − MLP", "GAT", "Δ GAT − MLP"], rows, NOTE)
    rows = []
    for drop in (0, 0.25, 0.5, 0.75, 1.0):
        a = (lambda m: m) if drop == 0 else (lambda m: f"{m}|drop{drop}")  # noqa: E731
        rows.append([f"{100 * (1 - drop):.0f} %", cell(kf, a("sage"), "spearman", f"edgedrop.{drop}.sage"),
                     con(kf, a("sage"), "mlp", "spearman", f"edgedrop.{drop}.sage-mlp"),
                     cell(kf, a("gat"), "spearman", f"edgedrop.{drop}.gat"),
                     con(kf, a("gat"), "mlp", "spearman", f"edgedrop.{drop}.gat-mlp")])
    write("edgedrop", "Share of the reference links into sampled papers that is kept (retrospective K-fold, *k* = 3, "
                      "Spearman correlation, mean ± std over 12 test folds).",
          ["Reference links kept", "GraphSAGE", "Δ GraphSAGE − MLP", "GAT", "Δ GAT − MLP"], rows, NOTE)

    pri = pd.concat([sel(runs("mature_main"), task="reg"), sel(runs("mature_firsthop"), task="reg")],
                    ignore_index=True)
    kf_all = pd.concat([kf, sel(runs("kfold_firsthop"), task="reg", k=3)], ignore_index=True)
    rows = []
    for a, n in (("mlp", "MLP"), ("mlp+refagg_in", "MLP + aggregates of the in-graph references"),
                 ("mlp+refagg", "MLP + aggregates of all references"), ("sage", "GraphSAGE"),
                 ("sage+refagg", "GraphSAGE + aggregates of all references"), ("gat", "GAT"),
                 ("gat+refagg", "GAT + aggregates of all references"),
                 ("sage@firsthop", "GraphSAGE, every reference as a node"),
                 ("gat@firsthop", "GAT, every reference as a node")):
        rows.append([n, cell(pri, a, "spearman", f"parity.primary.{a}"), cell(kf_all, a, "spearman", f"parity.kfold.{a}")])
    write("parity", "Feature parity between the graph-free baselines and the GNNs at *k* = 3 (Spearman correlation; "
                    "primary protocol: mean ± std over 15 runs; retrospective K-fold: over 12 test folds).",
          ["Model and reference information", "Primary protocol", "Retrospective K-fold"], rows)
    pairs = [("sage", "mlp+refagg_in"), ("gat", "mlp+refagg_in"), ("sage+refagg", "mlp+refagg"),
             ("gat+refagg", "mlp+refagg"), ("sage", "mlp+refagg"), ("gat", "mlp+refagg"), ("mlp+refagg", "mlp+refagg_in"),
             ("sage@firsthop", "mlp+refagg"), ("gat@firsthop", "mlp+refagg"), ("sage@firsthop", "sage"),
             ("gat@firsthop", "gat"), ("sage@firsthop", "mlp"), ("gat@firsthop", "mlp")]
    pn = {"sage": "GraphSAGE", "gat": "GAT", "sage+refagg": "GraphSAGE + all-reference aggregates",
          "gat+refagg": "GAT + all-reference aggregates", "mlp+refagg": "MLP + all-reference aggregates",
          "mlp+refagg_in": "MLP + in-graph aggregates", "sage@firsthop": "GraphSAGE, every reference as a node",
          "gat@firsthop": "GAT, every reference as a node", "mlp": "MLP"}
    rows = []
    for a, b in pairs:
        rows.append([pn[a], pn[b], con(pri, a, b, "spearman", f"parity.primary.{a}-{b}"),
                     con(kf_all, a, b, "spearman", f"parity.kfold.{a}-{b}")])
    write("parity_pairs", "Paired differences in Spearman correlation for the feature-parity comparisons "
                          "(first model minus second model, 95 % intervals).",
          ["Model", "Compared with", "Primary protocol", "Retrospective K-fold"], rows, NOTE)


# ====================================================================== topics, fields, tuning
def tables_topics() -> None:
    from scipy.stats import spearmanr, wilcoxon

    kf = sel(runs("kfold_main"), task="reg", k=3)
    comp = json.loads((RES / "stats" / "composition.json").read_text())
    topics = pd.read_csv(RES / "tables" / "topics.csv").set_index("topic_id")
    per = {a: S.by_group(kf, a, "topic", "spearman") for a in ("mlp", "sage", "gat", "mlp+refagg")}
    base = per["mlp"].spearman
    base = base[base.index.isin(topics.index)]          # the labelled seeds have no sampled topic
    rows, out = [], {}
    for a in ("mlp", "mlp+refagg", "sage", "gat"):
        v = per[a].spearman.reindex(base.index)
        d = v - base
        rec = {"median": float(v.median()), "q25": float(v.quantile(0.25)), "q75": float(v.quantile(0.75)),
               "min": float(v.min()), "max": float(v.max()), "n_topics": int(len(v))}
        if a != "mlp":
            cov = topics.mean_coverage.reindex(base.index)
            rec.update(better=int((d > 0).sum()), median_gain=float(d.median()),
                       wilcoxon_p=float(wilcoxon(d).pvalue), gain_vs_coverage=float(spearmanr(d, cov).correlation),
                       gain_vs_coverage_p=float(spearmanr(d, cov).pvalue))
        out[a] = rec
        rows.append([NAMES[a], f"{rec['median']:.3f} [{rec['q25']:.3f}, {rec['q75']:.3f}]",
                     f"{rec['min']:.3f} to {rec['max']:.3f}", "–" if a == "mlp" else f"{rec['better']} of {rec['n_topics']}",
                     "–" if a == "mlp" else signed(rec["median_gain"]),
                     "–" if a == "mlp" else ("< 0.001" if rec["wilcoxon_p"] < 0.001 else f"{rec['wilcoxon_p']:.3f}"),
                     "–" if a == "mlp" else f"{rec['gain_vs_coverage']:+.2f}".replace("-", "−")])
    record("topics", {"per_topic": out, "composition": comp["per_topic"], "n_fields": comp["n_fields"]})
    write("topics", "Performance within each of the 48 topics (Spearman correlation at *k* = 3, out-of-fold "
                    "predictions of the retrospective K-fold, averaged over runs).",
          ["Model", "Median over topics [quartiles]", "Range", "Topics where the model exceeds the MLP",
           "Median gain over the MLP", "Wilcoxon p", "Correlation of the gain with the reference coverage of the topic"],
          rows)
    full = topics.assign(mlp=per["mlp"].spearman, sage=per["sage"].spearman).sort_values(["field", "topic"])
    trows = [[r.topic, r.field, f"{r.median_citations_3y:.0f}", f"{r.top10_rate:.3f}", f"{r.median_in_graph_refs:.0f}",
              f"{r.mean_coverage:.2f}", f3(r.mlp), f3(r.sage)] for r in full.itertuples()]
    thead = ["Topic", "Field", "Median citations", "Top-10 % rate", "In-graph references", "Coverage", "MLP", "GraphSAGE"]
    half = len(trows) // 2
    write("topics_full", "The 48 topics of the sample (250 labelled papers each), first part: median citations at "
                         "three years, share of papers in the top-10 % class, median number of in-graph references, "
                         "mean share of the references of a paper that are graph nodes (initial graph), and Spearman "
                         "correlation within the topic for the MLP and GraphSAGE (out-of-fold predictions of the "
                         "retrospective K-fold, *k* = 3).", thead, trows[:half])
    write("topics_full2", "The 48 topics of the sample, second part (columns as in Table @topics_full).", thead,
          trows[half:])
    tf = runs("topicfold")
    cross = tf[tf.protocol == "topicfold"]
    grp = tf[tf.protocol == "group"]
    rows, g = [], {}
    for task, metric, arms in (("reg", "spearman", ("mlp", "mlp+refagg", "sage", "gat")),
                               ("cls", "roc_auc", ("mlp", "sage", "hgt_roll"))):
        within = sel(runs("kfold_main"), task=task, k=3)
        c = sel(cross, task=task)
        mname = "Spearman" if task == "reg" else "ROC-AUC"
        for split, d, suffix, tag in (("Folds within topics", within, "", "within"),
                                      ("Folds over topics", c, "|topicfold", "cross")):
            for n, a in enumerate(arms):
                rows.append([split if n == 0 else "", NAMES[a], mname,
                             cell(d, a + suffix, metric, f"topicfold.{tag}.{task}.{a}"),
                             "reference" if a == "mlp" else con(d, a + suffix, "mlp" + suffix, metric,
                                                                 f"topicfold.{tag}.{task}.{a}-mlp")])
    for tg, split in (("lab", "Trained on the open-corpus topics, tested on the laboratory topics"),
                      ("open", "Trained on the laboratory topics, tested on the open-corpus topics")):
        d = sel(grp, test_group=tg, task="reg")
        for n, a in enumerate(("mlp", "sage", "gat")):
            arm_ = f"{a}|test_{tg}"
            if not d[d.arm == arm_].empty:
                g[arm_] = ms(d[d.arm == arm_], "spearman")[:2]
            rows.append([split if n == 0 else "", NAMES[a], "Spearman", cell(d, arm_, "spearman"),
                         "reference" if a == "mlp" else con(d, arm_, f"mlp|test_{tg}", "spearman",
                                                             f"group.test_{tg}.{a}-mlp")])
    record("group", g)
    write("topicfold", "Generalisation across topics at *k* = 3 (mean ± std over 12 test folds, or over runs for the "
                       "two splits between groups of topics). Folds within topics: papers of a topic are in training "
                       "and test folds; folds over topics: all papers of a topic are in the same fold. Δ is the "
                       "paired difference with the MLP on the same split.",
          ["Split", "Model", "Metric", "Test value", "Δ vs MLP"], rows, NOTE)


def tables_field() -> None:
    rows = []
    for a in ("mlp", "sage", "gat", "han_temporal", "hgt_t0", "hgt_roll"):
        row = [NAMES[a]]
        for grid, tag in (("mature_field", "primary"), ("kfold_field", "kfold")):
            d = runs(grid)
            arm = f"{a}|field_year"
            row += [cell(d, arm, "roc_auc", f"field.{tag}.{a}.roc_auc"), cell(d, arm, "ap", f"field.{tag}.{a}.ap"),
                    "reference" if a == "mlp" else con(d, arm, "mlp|field_year", "roc_auc", f"field.{tag}.{a}-mlp")]
        rows.append(row)
    write("field", "Top-10 % class defined within publication year and field (*k* = 3; primary protocol: mean ± std "
                   "over 15 runs; retrospective K-fold: over 12 test folds).",
          ["Model", "Primary: ROC-AUC", "Primary: AP", "Primary: Δ ROC-AUC vs MLP", "K-fold: ROC-AUC", "K-fold: AP",
           "K-fold: Δ ROC-AUC vs MLP"], rows, NOTE)


def tables_tuning() -> None:
    from grids_rev1 import HPO_FILE

    best = json.loads(HPO_FILE.read_text())
    record("hpo", best)
    fam = {"mlp": "MLP", "sage": "GraphSAGE", "gat": "GAT", "han": "HAN", "hgt": "HGT"}
    rows = []
    for key, b in best.items():
        f, task, pf = key.split("|")
        h = b["hp"]
        rows.append([fam[f], ("regression" if task == "reg" else "classification") + ", "
                     + ("primary" if pf == "mature" else "K-fold"), str(h["hidden"]), str(h["n_layers"]),
                     str(h["heads"]) if f in ("gat", "han", "hgt") else "–", f"{h['dropout']:.1f}", f"{h['lr']:g}",
                     f"{h['weight_decay']:g}", f"{b['val_default']:.3f}", f"{b['val']:.3f}"])
        assert (b["text_dim"] or 768) == 64, key          # stated in the caption
    write("hpo", "Hyperparameters selected for each model family, task and protocol among 24 configurations, with "
                 "the mean validation metric of the initial configuration and of the selected one (Spearman "
                 "correlation for regression, ROC-AUC for classification). Every selected configuration uses 64 text "
                 "components.",
          ["Family", "Task, protocol", "Hidden units", "Layers", "Heads", "Dropout", "Learning rate", "Weight decay",
           "Validation: initial", "Validation: selected"], rows)
    rows = []
    for g_def, g_tun, tag, extra, pname in (
            ("mature_default", "mature_main", "primary", {}, "Primary"),
            ("kfold_default", "kfold_main", "kfold", {"k": 3}, "K-fold")):
        first = True
        for task, metric, arms in (("reg", "spearman", ("mlp", "mlp+refagg", "sage", "gat")),
                                   ("cls", "roc_auc", ("mlp", "sage", "gat", "han_temporal", "hgt_t0", "hgt_roll"))):
            d0, d1 = sel(runs(g_def), task=task, **extra), sel(runs(g_tun), task=task, **extra)
            for a in arms:
                rows.append([f"{pname}, {'Spearman' if task == 'reg' else 'ROC-AUC'}" if a == "mlp" else "", NAMES[a],
                             cell(d0, a, metric, f"tuning.{tag}.{task}.{a}.default"),
                             "reference" if a == "mlp" else con(d0, a, "mlp", metric, f"tuning.{tag}.{task}.{a}-mlp.default"),
                             cell(d1, a, metric, f"tuning.{tag}.{task}.{a}.tuned"),
                             "reference" if a == "mlp" else con(d1, a, "mlp", metric, f"tuning.{tag}.{task}.{a}-mlp.tuned")])
                first = False
    write("tuning", "Initial and tuned hyperparameters at *k* = 3 (primary protocol: mean ± std over 15 runs; "
                    "retrospective K-fold: over 12 test folds). Δ is the paired difference with the MLP under the same "
                    "setting.",
          ["Protocol, metric", "Model", "Initial configuration", "Δ vs MLP (initial)", "Tuned configuration",
           "Δ vs MLP (tuned)"], rows, NOTE)


def tables_origin() -> None:
    df = runs("mature_origin2021")
    rows = []
    for task, metric, arms in (("reg", "spearman", ("mlp", "mlp+refagg", "sage", "gat")),
                               ("cls", "roc_auc", ("mlp", "sage", "gat", "hgt_roll"))):
        d = sel(df, task=task)
        for a in arms:
            rows.append([NAMES[a], "Spearman" if task == "reg" else "ROC-AUC",
                         cell(d, f"{a}@T2021", metric, f"origin2021.{task}.{a}"),
                         "reference" if a == "mlp" else con(d, f"{a}@T2021", "mlp@T2021", metric,
                                                             f"origin2021.{task}.{a}-mlp")])
    n = df.groupby("arm").n_train.first().iloc[0], df.groupby("arm").n_test.first().iloc[0]
    record("origin2021.sizes", {"n_train": int(n[0]), "n_test": int(n[1])})
    write("origin2021", "Second prediction date (2021): training labels from the 2017 cohort, test cohorts 2021 and "
                        "2022 (*k* = 3, mean ± std over 15 runs).", ["Model", "Metric", "Test value", "Δ vs MLP"], rows, NOTE)


def tables_holdout() -> None:
    h = json.loads((RES / "holdout.json").read_text())
    record("holdout", h)
    # the five registered contrasts on the original test cohort of the primary protocol, with Holm's adjustment
    keys = ["rq1.primary.sage-mlp", "rq1.primary.gat-mlp", "rq1.primary.mlp+refagg-mlp", "rq3.primary.hgt_roll-mlp",
            "rq3.primary.sage-mlp"]
    if all(k in NUMBERS for k in keys):
        adj = S.holm([NUMBERS[k]["p"] for k in keys])
        record("principal.primary.holm", {k: {"p": NUMBERS[k]["p"], "p_holm": float(a)} for k, a in zip(keys, adj)})
    rows = []

    def label(a: str, b: str) -> str:
        """Readable name of a contrast between two arms of the frozen grid."""
        short = {"mlp": "MLP", "sage": "GraphSAGE", "gat": "GAT", "hgt_roll": "HGT (rolling)",
                 "mlp+refagg": "MLP + all-reference aggregates"}
        ma, *ra = a.split("|")
        mb, *rb = b.split("|")
        if not ra:
            return f"{short[ma]} − {short[mb]}"
        da, db = ra[1].replace("d", ""), rb[1].replace("d", "")
        setting = "SPECTER, initial hyperparameters"
        if ma == mb:
            return f"{short[ma]}: {da} components − {db} dimensions ({setting})"
        unit = "dimensions" if da == "768" else "components"
        return f"{short[ma]} − {short[mb]}, {da} {unit} ({setting})"

    for c in h["contrasts"]:
        rows.append(["Spearman" if c["task"] == "reg" else "ROC-AUC",
                     label(c["a"], c["b"]), c["role"], f3(c["mean_a"]), f3(c["mean_b"]), diff_ci(c),
                     f"[{signed(c['topic_ci'][0])}, {signed(c['topic_ci'][1])}]",
                     f"{c['p_holm']:.3f}" if "p_holm" in c else "–"])
    write("holdout", f"Confirmatory holdout: frozen models evaluated once on {h['n_fresh']:,} fresh papers of 2022 "
                     f"({h['positives']} in the top-10 % class). Paired differences with 95 % intervals from a "
                     f"bootstrap over papers and over topics; Holm-adjusted p-values for the registered primary "
                     f"contrasts.",
          ["Metric", "Contrast", "Role", "First model", "Second model", "Δ [95 % interval, papers]",
           "95 % interval, topics", "Holm p"], rows, NOTE)


TABLES = {"anomaly": tables_anomaly, "rq1": tables_rq1, "attention": tables_attention, "coldstart": tables_coldstart,
          "lab": tables_lab, "rq3": tables_rq3, "rq5": tables_rq5, "textdim": tables_textdim, "biblio": tables_biblio,
          "graphs": tables_graphs, "topics": tables_topics, "field": tables_field, "tuning": tables_tuning,
          "origin": tables_origin, "holdout": tables_holdout}


def main() -> None:
    names = sys.argv[1:] or list(TABLES)
    path = RES / "stats" / "tables.json"
    if path.exists():
        NUMBERS.update(json.loads(path.read_text()))
    path.parent.mkdir(parents=True, exist_ok=True)
    for n in names:
        try:
            TABLES[n]()
        except FileNotFoundError as exc:
            print(f"table group {n}: stopped ({exc.filename} missing)")
        except AssertionError as exc:          # a grid that is still running: folds of two arms differ
            print(f"table group {n}: stopped ({exc})")
        path.write_text(json.dumps(NUMBERS, indent=1, default=float))
    if PENDING:
        print(f"incomplete grids, {len(PENDING)} cells or contrasts pending:")
        for x in sorted(PENDING)[:40]:
            print("  ", x)


if __name__ == "__main__":
    main()
