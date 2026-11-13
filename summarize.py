"""Generate ``results/summary.md``: dataset statistics and every result table, read from the files
written by collect.py, build_graph.py and experiments.py (no number is typed by hand).

``python summarize.py --profile small|full``
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict

import pandas as pd

from config import ROOT, Config, add_common_args, get_config

MODEL_NAMES = {
    "mlp": "MLP (no graph)", "mlp_refagg": "MLP + reference aggregates", "sage": "GraphSAGE",
    "sage_e": "GraphSAGE + edge features", "gat": "GAT", "gat_e": "GAT + edge features",
    "han_t0": "HAN (T0 rule)", "han_temporal": "HAN (temporal rule)", "hgt_t0": "HGT (T0 rule)",
    "hgt_naive": "HGT naive (leaky, reference only)",
}


def pm(d: dict | None, nd: int = 3) -> str:
    if not d:
        return "–"
    return f"{d['mean']:.{nd}f} ± {d['std']:.{nd}f}"


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load(cfg: Config, name: str):
    p = cfg.results_dir / name
    return json.loads(p.read_text()) if p.exists() else None


LOG_RE = re.compile(r"^(\S+ \S+),\d+ \S+ collect: (.*)$")


def credits_from_logs(limit: int = 10000) -> dict[str, dict[str, int]]:
    """Credits consumed per (profile, step), from successive remaining-credit values in the collection logs."""
    events = []
    for prof in ("small", "full"):
        p = ROOT / "logs" / f"collect_{prof}.log"
        if p.exists():
            for i, line in enumerate(p.read_text().splitlines()):
                m = LOG_RE.match(line)
                if m:
                    events.append((m.group(1), prof, i, m.group(2)))
    events.sort()
    step: dict[str, str] = {}
    out: dict[str, Counter] = defaultdict(Counter)
    prev = None
    for _, prof, _, msg in events:
        m = re.match(r"=== step (\w+) \(profile", msg)
        if m:
            step[prof] = m.group(1)
            continue
        m = re.search(r"credits left: (\d+)", msg) or re.search(r"(\d+) credits left", msg)
        if not m or prof not in step:
            continue
        r = int(m.group(1))
        if prev is not None:
            out[prof][step[prof]] += (limit - r) if r > prev else (prev - r)
        prev = r
    return {k: dict(v) for k, v in out.items()}


def section_data(cfg: Config) -> list[str]:
    L = ["## 1. Data"]
    seeds = pd.read_parquet(cfg.raw_dir / "seeds.parquet")
    rows = []
    for corpus, g in seeds.groupby("corpus"):
        rows.append([corpus, len(g), int((g.openalex_id != "").sum()), int(g.dup_work.sum()),
                     int(g.rejected_match.sum()), int(g.doi_in_pdf.sum()),
                     ", ".join(f"{k}: {v}" for k, v in g.match_method.replace("", "unmatched").value_counts().items())])
    L += ["### 1.1 PDF matching (PyMuPDF + OpenAlex)",
          table(["corpus", "distinct PDFs", "matched", "duplicate work", "inherited match rejected",
                 "OpenAlex DOI printed in PDF", "match method"], rows)]
    lab = json.loads((cfg.raw_dir / "lab_files.json").read_text())
    L.append(f"\nLab folder: {len(lab)} files, {sum(x['status'] == 'duplicate' for x in lab)} exact duplicates "
             f"(sha256) removed.")
    rep = json.loads((cfg.raw_dir / "collect_report.json").read_text())
    credits = credits_from_logs().get(cfg.profile, {})
    rows = [[s, v.get("calls", "–"), credits.get(s, "–"), v.get("seconds_last_run", "–")] for s, v in rep["steps"].items()]
    rows.append(["total", sum(v.get("calls", 0) for v in rep["steps"].values()), sum(credits.values()), ""])
    samp = pd.read_parquet(cfg.raw_dir / "sample.parquet", columns=["id", "refs"])
    sw = pd.read_parquet(cfg.raw_dir / "seed_works.parquet", columns=["id", "refs"])
    n_ref_req = len({r for refs in pd.concat([samp.refs, sw.refs]) for r in refs} - set(samp.id) - set(sw.id))
    n_ref_rec = len(pd.read_parquet(cfg.raw_dir / "refs.parquet", columns=["id"]))
    raw_counts = [["sampled papers", len(samp)], ["seed records", len(sw)],
                  ["unique references of sampled and seed papers (excluding themselves)", n_ref_req],
                  ["reference records returned by OpenAlex", n_ref_rec],
                  ["author records", len(pd.read_parquet(cfg.raw_dir / "authors.parquet", columns=["id"]))],
                  ["venue records", len(pd.read_parquet(cfg.raw_dir / "venues.parquet", columns=["id"]))]]
    L += ["### 1.2 OpenAlex collection",
          "Calls = distinct API calls of the step (each sent once, then served from the disk cache). Credits are "
          "reconstructed from the remaining-credit values printed in `logs/collect_<profile>.log` (an upward jump "
          "means a new API key with a daily limit of 10,000 credits).",
          table(["step", "calls", "credits consumed", "seconds (last run, from cache)"], rows),
          table(["collected", "count"], raw_counts)]
    topics = json.loads((cfg.raw_dir / "topics.json").read_text())
    sample = pd.read_parquet(cfg.raw_dir / "sample.parquet", columns=["sampled_topic", "topic_name"])
    names = dict(zip(sample.sampled_topic, sample.topic_name))  # sampling is by primary topic
    for t in topics:
        t["name"] = t.get("name") or names.get(t["topic_id"], "")
    L += ["### 1.3 Topics of the labeled sample",
          table(["topic", "name", "group", "stratum", "papers sampled"],
                [[t["topic_id"], t.get("name", ""), t["group"], t.get("stratum", ""), t.get("n_sampled", "")]
                 for t in topics])]
    st = json.loads((cfg.processed_dir / "stats.json").read_text())
    L += ["### 1.4 Citation graph",
          table(["quantity", "value"], [
              ["paper nodes", st["nodes"]],
              ["nodes by origin", ", ".join(f"{k}: {v}" for k, v in st["nodes_by_origin"].items())],
              ["known works (with metadata)", st["known_works"]],
              ["citation edges (reference -> citing)", st["edges"]],
              ["edges dropped (reference published after citing paper)", st["edges_dropped_ref_later"]],
              ["share of nodes with an abstract", f"{st['abstract_share']:.3f}"],
              ["author records / venue records", f"{st['author_records']} / {st['venue_records']}"],
              ["sample by topic group", ", ".join(f"{k}: {v}" for k, v in st["sample_by_group"].items())],
          ])]
    rows = []
    for k in cfg.horizons:
        s = st[f"labels_k{k}"]
        rows.append([k, s["n"], ", ".join(f"{y}: {c}" for y, c in s["by_year"].items()),
                     f"{s['citations_mean']:.2f}", f"{s['citations_median']:.1f}", f"{s['zero_share']:.3f}",
                     f"{s['top_positive_rate']:.3f}", f"{s['in_degree_mean']:.2f}", f"{s['in_degree_median']:.1f}",
                     f"{s['in_degree_zero_share']:.3f}"])
    L += ["### 1.5 Labels (citations in years y..y+k, complete windows only)",
          table(["k", "labeled papers", "by year", "mean citations", "median", "share zero", "top-10 % rate",
                 "in-graph refs mean", "median", "share with none"], rows)]
    h = st["hetero"]
    L += ["### 1.6 Heterogeneous graph and venue graph",
          table(["quantity", "value"], [[k, v] for k, v in h.items()] +
                [[f"venue graph {k}", v] for k, v in st["venue_graph"].items()])]
    return L


def section_exp1(cfg: Config) -> list[str]:
    r = load(cfg, "exp1.json")
    if not r:
        return []
    L = ["## 2. Sub-topic 1: future impact with GAT (random split 70/15/15, regression on log1p citations)"]
    for k in cfg.horizons:
        s = r["summary"].get(f"k{k}", {})
        L += [f"### k = {k}", table(["model", "Spearman", "R²", "MAE (log)", "runs"],
                                    [[MODEL_NAMES.get(m, m), pm(v.get("spearman")), pm(v.get("r2")), pm(v.get("mae")),
                                      v["spearman"]["n"]] for m, v in s.items()])]
    rows = []
    for key, a in r.get("attention", {}).items():
        for kind in ("trained", "untrained", "trained_layer1"):
            if kind in a:
                x = a[kind]
                rows.append([key.replace("|", " "), kind, pm(x["per_paper_cit_mean"]), pm(x["per_paper_cit_share_pos"]),
                             pm(x["pooled_cit"]), pm(x["per_paper_age_mean"]), pm(x["per_paper_age_share_pos"]),
                             pm(x["pooled_age"]), pm(x["n_targets"], 0)])
    L += ["### Attention analysis (test papers with >= 3 in-graph references; self-loops excluded)",
          "Per-paper = Spearman between attention and the reference attribute within each citing paper, "
          "averaged over papers; pooled = Spearman between degree-normalised attention and the attribute over all edges.",
          table(["model k", "weights", "per-paper ρ(att, ref citations)", "share ρ>0", "pooled ρ(cit)",
                 "per-paper ρ(att, ref age)", "share ρ>0", "pooled ρ(age)", "papers"], rows)]
    for key, a in r.get("attention", {}).items():
        if "self_attention_test_mean" in a:
            L.append(f"- {key.replace('|', ' ')}: mean self-loop attention on test papers = {pm(a['self_attention_test_mean'])}")
    if r.get("examples"):
        L.append("### Examples (GAT, attention averaged over seeds; highest-impact labeled papers)")
        for e in r["examples"]:
            ck = [c for c in e if c.startswith("citations_")][0]
            L.append(f"- **{e['title']}** ({e['year']}, {e['venue']}; {e[ck]} citations, "
                     f"{e['n_refs_in_graph']} references in graph, uniform attention = {1 / e['n_refs_in_graph']:.3f})")
            for lab in ("top3", "bottom3"):
                for x in e[lab]:
                    L.append(f"    - {lab[:-1]}: {x['title']} ({x['year']}; {x['citations_before_citing_year']} "
                             f"citations before citing year; attention {x['attention']:.3f})")
    return L


def section_exp2(cfg: Config) -> list[str]:
    r = load(cfg, "exp2.json")
    if not r:
        return []
    L = ["## 3. Sub-topic 2: inductive cold start with GraphSAGE (temporal split)"]
    for k in cfg.horizons:
        sp, tg = r["split"][f"k{k}"], r["train_graph"].get(f"k{k}") or {}
        s = r["summary"].get(f"k{k}", {})
        L += [f"### k = {k}: train < {sp['val_year']}, validation = {sp['val_year']}, test >= {sp['test_from']} "
              f"(training graph: {tg.get('nodes')} nodes, {tg.get('edges')} edges; test papers absent)",
              table(["variant", "Spearman", "R²", "MAE (log)", "runs"],
                    [[m, pm(v.get("spearman")), pm(v.get("r2")), pm(v.get("mae")), v["spearman"]["n"]]
                     for m, v in s.items()])]
    lp = r.get("lab_predictions", [])
    if lp:
        cols = [c for c in lp[0] if c.endswith("_pred_citations") or c.endswith("_percentile") or c.startswith("actual")]
        L += ["### Lab papers (deployment models trained on all labeled years; mean over seeds)",
              "Percentile = share of labeled papers whose actual log citations are below the prediction. "
              "`extrapolated` = publication year outside the labeled years.",
              table(["title", "year"] + cols + ["extrapolated"],
                    [[x["title"][:80], x["year"]] + [x.get(c) for c in cols] + [x["extrapolated_year"]] for x in lp])]
    return L


def section_exp3(cfg: Config) -> list[str]:
    r = load(cfg, "exp3.json")
    if not r:
        return []
    L = ["## 4. Sub-topic 3: heterogeneous graph, top-10 % detection (random split)"]
    for k in cfg.horizons:
        s = r["summary"].get(f"k{k}", {})
        L += [f"### k = {k}", table(["model", "ROC-AUC", "AP", "precision@10%", "runs"],
                                    [[MODEL_NAMES.get(m, m), pm(v.get("roc_auc")), pm(v.get("ap")), pm(v.get("p_at_10")),
                                      v["roc_auc"]["n"]] for m, v in s.items()])]
    return L


def section_exp4(cfg: Config) -> list[str]:
    r = load(cfg, "exp4.json")
    if not r:
        return []
    inj = r["injection"]
    L = ["## 5. Sub-topic 4: citation cartels and self-citation (venue graph, injected anomalies)",
         f"Venue graph: {r['graph']['venues']} venues, {r['graph']['edges']} weighted edges, {r['graph']['links']} "
         f"citation links. Injection per draw: {inj['n_cartels']} cartels of {inj['cartel_size']} venues, "
         f"{inj['n_selfcit']} self-citing venues; intensity = share of the venue's outgoing citations added; "
         f"{inj['draws']} draws."]
    for lvl, s in r["summary"].items():
        dets = sorted({k.split("|")[0] for k in s})
        rows = [[d] + [pm(s.get(f"{d}|{t}", {}).get(m)) for t in ("cartel", "self", "all") for m in ("roc_auc", "ap")]
                for d in dets]
        L += [f"### intensity {lvl} ({inj['intensities'][lvl]})",
              table(["detector", "cartel AUC", "cartel AP", "self AUC", "self AP", "all AUC", "all AP"], rows)]
    if r.get("size_correlation"):
        L += ["### Real graph: Spearman correlation of each detector's score with venue size",
              table(["detector", "ρ(score, log out-links)", "ρ(score, log in-links)"],
                    [[d, f"{v['log_out']:.3f}", f"{v['log_in']:.3f}"] for d, v in r["size_correlation"].items()])]
    if r.get("cases_for_inspection"):
        ref = r["real_graph_reference"]
        L += ["### Real graph: venues ranked highest by DOMINANT (cases for inspection, not accusations)",
              "Reference values over all venues: " + "; ".join(
                  f"{k} median {v['median']:.3f}, p90 {v['p90']:.3f}" for k, v in ref.items()),
              table(["venue", "rank pct", "self ratio", "reciprocity", "out links", "top partner", "top partner share"],
                    [[c["venue"], f"{c['dominant_rank_pct']:.3f}", f"{c['self_ratio']:.3f}", f"{c['reciprocity']:.3f}",
                      c["out_links"], c["top_partner"], f"{c['top_partner_share']:.3f}"]
                     for c in r["cases_for_inspection"]])]
    return L


def section_exp5(cfg: Config) -> list[str]:
    r = load(cfg, "exp5.json")
    if not r:
        return []
    L = ["## 6. Sub-topic 5: text vs structure (k = 3, random split)"]
    reg, cls = r["summary"].get("reg", {}), r["summary"].get("cls", {})
    keys = sorted(set(reg) | set(cls), key=lambda s: (s.split("|")[0], s.split("|")[1], s.split("|")[2]))
    L.append(table(["model", "text", "bibliometric", "Spearman", "R²", "MAE", "ROC-AUC", "AP", "precision@10%"],
                   [[*k.split("|"), pm(reg.get(k, {}).get("spearman")), pm(reg.get(k, {}).get("r2")), pm(reg.get(k, {}).get("mae")),
                     pm(cls.get(k, {}).get("roc_auc")), pm(cls.get(k, {}).get("ap")), pm(cls.get(k, {}).get("p_at_10"))]
                    for k in keys]))
    # contributions (differences of means)
    rows = []
    for bib in ("nobiblio", "biblio"):
        for text in ("none", "scibert", "specter"):
            for m in ("sage", "gat"):
                a, b = reg.get(f"{m}|{text}|{bib}"), reg.get(f"mlp|{text}|{bib}")
                ca, cb = cls.get(f"{m}|{text}|{bib}"), cls.get(f"mlp|{text}|{bib}")
                if a and b and ca and cb:
                    rows.append([f"graph: {m} − mlp", text, bib, f"{a['spearman']['mean'] - b['spearman']['mean']:+.3f}",
                                 f"{ca['roc_auc']['mean'] - cb['roc_auc']['mean']:+.3f}"])
        for m in ("mlp", "sage", "gat"):
            for text in ("scibert", "specter"):
                a, b = reg.get(f"{m}|{text}|{bib}"), reg.get(f"{m}|none|{bib}")
                ca, cb = cls.get(f"{m}|{text}|{bib}"), cls.get(f"{m}|none|{bib}")
                if a and b and ca and cb:
                    rows.append([f"text: {text} − none", m, bib, f"{a['spearman']['mean'] - b['spearman']['mean']:+.3f}",
                                 f"{ca['roc_auc']['mean'] - cb['roc_auc']['mean']:+.3f}"])
    L += ["### Contributions (difference of mean scores)", table(["contrast", "setting", "bibliometric",
                                                                  "Δ Spearman", "Δ ROC-AUC"], rows)]
    return L


def section_exploratory(cfg: Config, dim: int = 64) -> list[str]:
    """Post hoc analysis decided after seeing exp5: text embeddings reduced to ``dim`` PCA components."""
    e1, e3, e5 = (load(cfg, f"exp{e}_pca{dim}.json") for e in (1, 3, 5))
    if not (e1 or e3 or e5):
        return []
    any_r = e1 or e3 or e5
    ev = any_r.get("pca_explained_variance", {})
    L = [f"## 9. Exploratory (post hoc): text embeddings reduced to {dim} PCA components",
         "Decided after inspecting sub-topic 5 (the 768-dimensional text inputs appeared to overfit). PCA is fitted "
         "without labels on all paper nodes. Explained variance: " +
         ", ".join(f"{m} {v:.3f}" for m, v in ev.items()) + ". Main results above remain the reference."]
    m1 = load(cfg, "exp1.json")
    if e1 and m1:
        rows = []
        for k in cfg.horizons:
            for m, v in e1["summary"].get(f"k{k}", {}).items():
                base = m1["summary"][f"k{k}"].get(m, {})
                rows.append([k, MODEL_NAMES.get(m, m), pm(base.get("spearman")), pm(v.get("spearman")),
                             pm(base.get("r2")), pm(v.get("r2"))])
        L += ["### Sub-topic 1 with PCA-64 text", table(["k", "model", "Spearman (768-d)", f"Spearman ({dim}-d)",
                                                         "R² (768-d)", f"R² ({dim}-d)"], rows)]
    m3 = load(cfg, "exp3.json")
    if e3 and m3:
        rows = []
        for k in cfg.horizons:
            for m, v in e3["summary"].get(f"k{k}", {}).items():
                base = m3["summary"][f"k{k}"].get(m, {})
                rows.append([k, MODEL_NAMES.get(m, m), pm(base.get("roc_auc")), pm(v.get("roc_auc")),
                             pm(base.get("ap")), pm(v.get("ap"))])
        L += ["### Sub-topic 3 with PCA-64 text", table(["k", "model", "ROC-AUC (768-d)", f"ROC-AUC ({dim}-d)",
                                                         "AP (768-d)", f"AP ({dim}-d)"], rows)]
    if e5:
        reg, cls = e5["summary"].get("reg", {}), e5["summary"].get("cls", {})
        keys = sorted(k for k in set(reg) | set(cls) if k.split("|")[1] != "none")
        L += ["### Sub-topic 5 with PCA-64 text (runs without text are identical to the main grid)",
              table(["model", "text", "bibliometric", "Spearman", "R²", "ROC-AUC", "AP"],
                    [[*k.split("|"), pm(reg.get(k, {}).get("spearman")), pm(reg.get(k, {}).get("r2")),
                      pm(cls.get(k, {}).get("roc_auc")), pm(cls.get(k, {}).get("ap"))] for k in keys])]
        rows = []
        for bib in ("nobiblio", "biblio"):
            for text in ("scibert", "specter"):
                for m in ("sage", "gat"):
                    a, b = reg.get(f"{m}|{text}|{bib}"), reg.get(f"mlp|{text}|{bib}")
                    ca, cb = cls.get(f"{m}|{text}|{bib}"), cls.get(f"mlp|{text}|{bib}")
                    if a and b and ca and cb:
                        rows.append([f"graph: {m} − mlp", text, bib,
                                     f"{a['spearman']['mean'] - b['spearman']['mean']:+.3f}",
                                     f"{ca['roc_auc']['mean'] - cb['roc_auc']['mean']:+.3f}"])
        L += ["### Graph contribution with PCA-64 text (difference of mean scores)",
              table(["contrast", "text", "bibliometric", "Δ Spearman", "Δ ROC-AUC"], rows)]
    return L


def section_ogb(cfg: Config) -> list[str]:
    r = load(get_config("full"), "ogbn_arxiv.json")
    if not r:
        return []
    pub = r["published"]
    rows = [[m, f"{v['mean']:.2f} ± {v['std']:.2f}", len(v["test_acc"]),
             f"{pub[m][0]:.2f} ± {pub[m][1]:.2f}" if m in pub else "–"] for m, v in r["runs"].items()]
    return ["## 10. Implementation check on ogbn-arxiv (test accuracy %)",
            f"Reference values: {pub['source']}; our models keep their residual architecture (not a reproduction).",
            table(["model", "ours (mean ± std)", "seeds", "published"], rows)]


def section_runtime(cfg: Config) -> list[str]:
    rows = []
    for e in range(1, 6):
        r = load(cfg, f"exp{e}.json")
        if not r:
            continue
        runs = r.get("runs", [])
        t = [x["time_s"] for x in runs if "time_s" in x]
        mem = [x["peak_gpu_mb"] for x in runs if x.get("peak_gpu_mb") is not None]
        rows.append([e, r.get("device"), r.get("workers"), len(runs), r.get("wall_time_s"),
                     f"{sum(t) / max(len(t), 1):.1f}", f"{max(mem):.0f}" if mem else "–"])
    env = load(cfg, "env.json") or {}
    return ["## 7. Run time and environment",
            table(["exp", "device", "workers", "runs", "wall time (s)", "mean run (s)", "peak GPU MB / run"], rows),
            "", "Environment: " + ", ".join(f"{k} {v}" for k, v in env.items())]


LEAKAGE = """## 8. Leakage control and residual leakage

Controls implemented in the code (asserted in `build_graph.py`, tested in `tests/`):
- citation edges go only from a reference to the citing paper, and edges whose reference is published after the
  citing paper are dropped; a paper never receives information from the papers that cite it;
- author, venue and reference statistics attached to a paper published in year y count only citations received
  before y (`total − Σ counts_by_year[t ≥ y]`); for years before 2012 (first covered year) the value as of 2011 is
  used, which is still earlier than every target (targets are published ≥ 2017);
- labels: citations in years y..y+k, papers kept only if y+k ≤ 2025; lab papers are never training labels;
- cold start: test papers (and every paper published from the test year on) are removed from the training graph;
- heterogeneous graph: author/venue/keyword nodes receive messages only from papers published before 2017;
  HAN metapath neighbours satisfy year(Q) < 2017 (T0 rule) or year(Q) < year(P) (temporal rule);
  the "HGT naive" row uses all reverse edges and is reported only to show how much leakage inflates scores.

Residual leakage (not removable with a 2026 snapshot):
- OpenAlex author disambiguation, topics and keywords are computed today, with information posterior to publication;
- SPECTER was trained on citation triplets of papers published up to about 2019–2020, so its embeddings may encode
  citation information of part of the 2017–2020 papers (SciBERT is trained on text only);
- the sample contains only works still indexed in 2026 (survivorship), with reference lists as currently recorded;
- the label window includes citations received in the publication year itself;
- reference-node selection uses the number of citers among sampled papers (a structural, label-free criterion);
- bf16 autocast is used for all models except HGT (pyg-lib `segment_matmul` does not accept bf16 inputs).
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = get_config(args.profile)
    parts = [f"# Axis 1 – Citation graphs and GNNs: results summary ({cfg.profile} run)",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `summarize.py` from `data/` and `results/`."]
    for fn in (section_data, section_exp1, section_exp2, section_exp3, section_exp4, section_exp5, section_runtime):
        parts += fn(cfg) + [""]
    parts.append(LEAKAGE)
    parts += section_exploratory(cfg)
    parts += section_ogb(cfg)
    out = cfg.results_dir / "summary.md"
    out.write_text("\n\n".join(p for p in parts if p is not None))
    print(out)


if __name__ == "__main__":
    main()
