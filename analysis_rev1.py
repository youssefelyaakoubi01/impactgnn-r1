"""Descriptive analyses of the revision that do not train models: topic-level composition and reference coverage
of the sample, and reference coverage of every graph variant.

``python analysis_rev1.py``  ->  ``results/rev1/tables/topics.csv``, ``results/rev1/stats/composition.json``
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from config import ROOT, get_config
from graph_variants import edge_mask, node_mask

RES = ROOT / "results" / "rev1"
VARIANTS = ["submitted", "cap30k", "cap45k", "cap60k", "uncapped", "cut2022", "cut2022lso", "cut2022sym", "cut2021sym"]


def main() -> None:
    cfg = get_config("full")
    g = torch.load(cfg.processed_dir / "rev1" / "graph.pt", weights_only=False)
    topics = {t["topic_id"]: t for t in json.loads((cfg.raw_dir / "topics.json").read_text())}
    sample = pd.read_parquet(cfg.raw_dir / "sample.parquet", columns=["sampled_topic", "topic_name", "topic_id"])
    names = dict(zip(sample.topic_id, sample.topic_name))
    n = len(g["ids"])
    ei = g["edge_index"]
    years = g["year"].numpy()
    lab = g["m3"].numpy().astype(bool)
    origin = np.asarray(g["origin"])
    n_refs = g["n_refs"].numpy()

    def in_deg(variant):
        em = edge_mask(g, variant)
        return np.bincount(ei[1, em].numpy(), minlength=n), int(node_mask(g, variant).sum()), int(em.sum())

    deg_sub, _, _ = in_deg("submitted")
    df = pd.DataFrame({"topic": g["sampled_topic"], "group": g["topic_group"], "field": g["field"], "year": years,
                       "c3": g["c3"].numpy(), "top3": g["top3"].numpy(), "deg": deg_sub, "n_refs": n_refs})
    df = df[lab & (origin == "sample")]
    df["coverage"] = df.deg / df.n_refs.clip(lower=1)
    rows = []
    for t, d in df.groupby("topic"):
        rows.append({"topic_id": t, "topic": names.get(t, topics.get(t, {}).get("name", "")),
                     "group": d.group.iloc[0], "field": d.field.mode().iloc[0], "n_labelled": len(d),
                     "n_2022": int((d.year == 2022).sum()), "median_citations_3y": float(d.c3.median()),
                     "top10_rate": float(d.top3.mean()), "n_top10": int(d.top3.sum()),
                     "median_in_graph_refs": float(d.deg.median()), "mean_coverage": float(d.coverage.mean()),
                     "share_no_in_graph_ref": float((d.deg == 0).mean())})
    tab = pd.DataFrame(rows).sort_values(["group", "topic"])
    (RES / "tables").mkdir(parents=True, exist_ok=True)
    (RES / "stats").mkdir(parents=True, exist_ok=True)
    tab.to_csv(RES / "tables" / "topics.csv", index=False)

    q = lambda s: {"min": float(s.min()), "q25": float(s.quantile(0.25)), "median": float(s.median()),  # noqa: E731
                   "q75": float(s.quantile(0.75)), "max": float(s.max())}
    comp = {"n_topics": int(len(tab)), "by_group": tab.groupby("group").n_labelled.sum().to_dict(),
            "fields": df.field.value_counts().to_dict(), "n_fields": int(df.field.nunique()),
            "per_topic": {c: q(tab[c]) for c in ("n_labelled", "median_citations_3y", "top10_rate", "n_top10",
                                                  "median_in_graph_refs", "mean_coverage", "share_no_in_graph_ref")},
            "top10_rate_by_field": df.groupby("field").top3.mean().round(4).to_dict(),
            "labelled_by_year": {int(y): int(c) for y, c in df.year.value_counts().sort_index().items()},
            "variants": {}}
    tr, te = lab & (years <= 2018), lab & (years == 2022) & (origin == "sample")
    all_lab = lab & (origin == "sample")
    for v in VARIANTS:
        deg, nn, ne = in_deg(v)
        f = lambda m: {"median": float(np.median(deg[m])), "mean": float(deg[m].mean()),  # noqa: E731
                       "zero_share": float((deg[m] == 0).mean()),
                       "coverage": float((deg[m] / np.maximum(n_refs[m], 1)).mean())}
        comp["variants"][v] = {"nodes": nn, "edges": ne, "labelled": f(all_lab), "train_2017_2018": f(tr),
                               "test_2022": f(te)}
    # selection of the reference nodes: role of citers published after a prediction date, and of the cap tie-break
    ref = origin == "ref"
    sub = g["submitted"].numpy().astype(bool)
    shared = ref & ~g["sel_lab_ref"].numpy().astype(bool)
    cb = g["sel_citers_before"].numpy()
    cut = {T: j for j, T in enumerate(g["cutoffs"])}
    comp["selection"] = {
        "shared_reference_nodes_submitted": int((shared & sub).sum()),
        "fail_two_citers_before_2021": int((shared & sub & (cb[:, cut[2021]] < 2)).sum()),
        "fail_two_citers_before_2022": int((shared & sub & (cb[:, cut[2022]] < 2)).sum()),
        "no_citer_before_2021": int((shared & sub & (cb[:, cut[2021]] == 0)).sum()),
        "cut_by_cap": int((shared & ~sub).sum()),
        "cut_by_cap_published_2019_or_later": float((years[shared & ~sub] >= 2019).mean()),
        "kept_two_citers_published_2019_or_later": float(
            (years[shared & sub & (g["sel_n_citers"].numpy() == 2)] >= 2019).mean()),
        "median_year_cut": float(np.median(years[shared & ~sub])),
        "median_year_kept_two_citers": float(np.median(years[shared & sub & (g["sel_n_citers"].numpy() == 2)])),
    }
    # sizes of the protocols and number of labelled seed papers
    from protocols import make_split

    comp["labelled_seeds_k3"] = int((lab & (origin == "seed_open")).sum())
    comp["protocols"] = {}
    for name, job in (("mature_2022", dict(protocol="mature", T=2022, k=3, split_seed=0)),
                      ("mature_2021", dict(protocol="mature", T=2021, k=3, split_seed=0)),
                      ("retro_k3", dict(protocol="retro", T=2021, k=3)), ("retro_k5", dict(protocol="retro", T=2020, k=5)),
                      ("kfold_k3", dict(protocol="kfold", k=3, rep=0, fold=0)),
                      ("kfold_k5", dict(protocol="kfold", k=5, rep=0, fold=0))):
        sp = make_split(g, job, cfg)
        comp["protocols"][name] = {"train": int(len(sp["tr"])), "validation": int(len(sp["va"])),
                                   "test": int(len(sp["te"])), "train_plus_validation": int(len(sp["tr"]) + len(sp["va"]))}
    # meaning of the yearly citation counts of author and venue records (Section 3.2)
    sample = pd.read_parquet(cfg.raw_dir / "sample.parquet")
    authors = pd.read_parquet(cfg.raw_dir / "authors.parquet")
    venues = pd.read_parquet(cfg.raw_dir / "venues.parquet")
    A = {r.id: (dict(zip(r.cby_years, r.cby_works)), dict(zip(r.cby_years, r.cby_cited))) for r in authors.itertuples()}
    n_single = n_total = 0
    for pp in sample.itertuples():
        for aid in pp.author_ids:
            if aid in A and A[aid][0].get(pp.year, 0) == 1:
                n_single += 1
                n_total += A[aid][1].get(pp.year, 0) == pp.cited_by_count
    count = lambda df: (sum(len(r) for r in df.cby_works),  # noqa: E731
                        sum(int(w == 0 and c > 0) for ws, cs in zip(df.cby_works, df.cby_cited) for w, c in zip(ws, cs)))
    comp["record_semantics"] = {"author_years_single_sampled_work": n_single, "equal_to_total_citations": n_total,
                                "share": round(n_total / n_single, 4), "author_year_entries": count(authors)[0],
                                "author_entries_citations_without_works": count(authors)[1],
                                "venue_year_entries": count(venues)[0],
                                "venue_entries_citations_without_works": count(venues)[1]}
    (RES / "stats" / "composition.json").write_text(json.dumps(comp, indent=1, default=int))
    print(json.dumps({k: comp[k] for k in ("n_topics", "by_group", "n_fields", "per_topic", "selection")}, indent=1))


if __name__ == "__main__":
    main()
