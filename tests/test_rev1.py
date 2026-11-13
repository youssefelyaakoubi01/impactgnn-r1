"""Tests of the revision code: label maturity, fit rows, cutoff-restricted graphs, time-safe entity statistics and
the paired statistics. They use small synthetic graphs, so they run without the collected data."""

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

import stats as S
from build_graph import YearlyCounts
from build_rev1 import GRID0, before_table, clip_years
from config import Config
from experiments_rev1 import fit_mask, pca_fit, refagg_ingraph
from graph_variants import edge_mask, node_mask
from protocols import make_split


def toy_graph(n_per_year: int = 40, seed: int = 0) -> dict:
    """Sampled papers 2017-2022, a few seeds and older reference nodes."""
    rng = np.random.default_rng(seed)
    years, origin, topic, group = [], [], [], []
    for y in range(2017, 2023):
        for i in range(n_per_year):
            years.append(y)
            origin.append("sample")
            topic.append(f"T{i % 8}")
            group.append("lab" if i % 8 < 4 else "open")
    n_base = len(years)
    for i in range(60):
        years.append(int(rng.integers(2000, 2020)))
        origin.append("ref")
        topic.append("")
        group.append("")
    years = np.array(years)
    cfg = Config()
    m3 = np.array([o == "sample" and y + 3 <= cfg.last_complete_year for o, y in zip(origin, years)])
    m5 = np.array([o == "sample" and y + 5 <= cfg.last_complete_year for o, y in zip(origin, years)])
    n = len(years)
    return {"year": torch.as_tensor(years), "origin": origin, "sampled_topic": topic, "topic_group": group,
            "m3": torch.as_tensor(m3), "m5": torch.as_tensor(m5), "n_base": n_base, "n": n}


# ---------------------------------------------------------------- protocols
@pytest.mark.parametrize("T", [2021, 2022])
def test_mature_labels_are_closed_before_the_prediction_date(T):
    g, cfg = toy_graph(), Config()
    sp = make_split(g, dict(protocol="mature", T=T, k=3, split_seed=1), cfg)
    years = g["year"].numpy()
    fit_years = years[np.concatenate([sp["tr"], sp["va"]])]
    assert (fit_years + 3 <= T - 1).all()                 # window y .. y+3 ends before the prediction date
    assert (years[sp["te"]] >= T).all()
    assert not sp["train_nodes"][years >= T].any()        # no paper published from T on in the training graph
    assert len(sp["te"]) and len(sp["va"]) and len(sp["tr"])


def test_retro_split_has_immature_labels():
    g, cfg = toy_graph(), Config()
    sp = make_split(g, dict(protocol="retro", T=2021, k=3), cfg)
    years = g["year"].numpy()
    assert (years[sp["tr"]] + 3 > 2020).any()             # the reason it is called retrospective


def test_kfold_partitions_every_labelled_paper_once():
    g, cfg = toy_graph(), Config()
    seen = []
    for fold in range(6):
        sp = make_split(g, dict(protocol="kfold", k=3, rep=0, fold=fold), cfg)
        assert not set(sp["te"]) & set(sp["tr"]) and not set(sp["te"]) & set(sp["va"])
        seen.append(sp["te"])
    seen = np.concatenate(seen)
    assert len(seen) == len(set(seen)) == int(g["m3"].sum())
    a = make_split(g, dict(protocol="kfold", k=3, rep=0, fold=2), cfg)["te"]
    b = make_split(g, dict(protocol="kfold", k=3, rep=1, fold=2), cfg)["te"]
    assert set(a) != set(b)                               # repetitions use different partitions


def test_topicfold_keeps_topics_together():
    g, cfg = toy_graph(), Config()
    topic = np.asarray(g["sampled_topic"])
    for fold in range(6):
        sp = make_split(g, dict(protocol="topicfold", k=3, rep=0, fold=fold), cfg)
        assert not set(topic[sp["te"]]) & set(topic[np.concatenate([sp["tr"], sp["va"]])])


def test_fit_rows_exclude_validation_test_and_future_nodes():
    g, cfg = toy_graph(), Config()
    sp = make_split(g, dict(protocol="mature", T=2022, k=3, split_seed=0), cfg)
    vn = torch.ones(g["n"], dtype=torch.bool)
    tr, va, te = (torch.as_tensor(sp[k]) for k in ("tr", "va", "te"))
    fit = fit_mask(vn, torch.as_tensor(sp["train_nodes"]), tr, va, te)
    assert not fit[va].any() and not fit[te].any()
    assert not fit[g["year"] >= 2022].any()
    assert fit[tr].all()
    x = torch.randn(g["n"], 16)
    x[te] += 100.0                                         # a shift in the test rows must not move the fit
    mean, comp, _ = pca_fit(x, fit, 4)
    assert mean.abs().max() < 5


# ---------------------------------------------------------------- graph variants
def variant_graph() -> dict:
    """Papers P0 (2018), P1 (2019), P2 (2022) and references R0..R3.
    R0: cited by P0, P1 (two citers before 2022)         -> node of every cutoff variant
    R1: cited by P0, P2 (one citer before 2022, one after) -> only through the symmetric rule, for P2
    R2: cited by P2 and a 2024 seed P3                    -> never available at T = 2022
    R3: cited by P0, P1, P2."""
    origin = ["sample", "sample", "sample", "seed_open", "ref", "ref", "ref", "ref"]
    year = torch.tensor([2018, 2019, 2022, 2024, 2010, 2011, 2012, 2013])
    edges = [(4, 0), (4, 1), (5, 0), (5, 2), (6, 2), (6, 3), (7, 0), (7, 1), (7, 2)]
    citers = {4: [2018, 2019], 5: [2018, 2022], 6: [2022, 2024], 7: [2018, 2019, 2022]}
    cutoffs = list(range(2017, 2025))
    before = torch.zeros(8, len(cutoffs), dtype=torch.long)
    for r, ys in citers.items():
        before[r] = torch.tensor([sum(y < T for y in ys) for T in cutoffs])
    return {"origin": origin, "year": year, "edge_index": torch.tensor(edges).T, "cutoffs": cutoffs,
            "sel_citers_before": before, "sel_lab_ref": torch.zeros(8, dtype=torch.bool),
            "sel_lab_first_year": torch.full((8,), 9999), "submitted": torch.ones(8, dtype=torch.bool),
            "sel_rank_fair": torch.tensor([-1, -1, -1, -1, 1, 2, 3, 0])}


def kept(g, variant):
    em = edge_mask(g, variant)
    return {tuple(e) for e in g["edge_index"][:, em].T.tolist()}


def test_cutoff_variant_uses_only_citers_before_the_date():
    g = variant_graph()
    nm = node_mask(g, "cut2022")
    assert nm[[4, 7]].all() and not nm[[5, 6]].any()      # R1 and R2 need a citer published from 2022 on
    assert kept(g, "cut2022") == {(4, 0), (4, 1), (7, 0), (7, 1), (7, 2)}


def test_leave_self_out_and_symmetric_rules():
    g = variant_graph()
    # leave-self-out: a training paper does not count for its own references. R0 and R3 have two citers before
    # 2022, i.e. one without the citing training paper, so their links to P0 and P1 go; P2 (2022) was never counted
    assert kept(g, "cut2022lso") == {(7, 2)}
    # symmetric: the 2022 paper counts for its own references, so R1 (one earlier citer) is linked to it
    sym = kept(g, "cut2022sym")
    assert (5, 2) in sym and (5, 0) not in sym            # ... but R1 is not linked to the training paper P0
    assert (6, 2) not in sym                              # R2 has no citer before 2022
    assert {(4, 0), (4, 1), (7, 0), (7, 1), (7, 2)} <= sym


def test_cap_variant_takes_the_best_ranked_references():
    g = variant_graph()
    nm = node_mask(g, "cap6k")                            # budget = 6000 - 4 base papers: every reference fits
    assert nm.all()


def test_ingraph_aggregates_follow_the_edge_mask():
    g = variant_graph()
    g["edge_attr"] = torch.arange(18, dtype=torch.float32).reshape(9, 2)
    g["n_refs"] = torch.full((8,), 10.0)
    full = refagg_ingraph(g, torch.ones(9, dtype=torch.bool))
    cut = refagg_ingraph(g, edge_mask(g, "cut2022"))
    assert full[2, 5] == pytest.approx(np.log1p(3)) and cut[2, 5] == pytest.approx(np.log1p(1))


# ---------------------------------------------------------------- time-safe entity statistics
def test_entity_citations_before_a_year_count_only_earlier_citations():
    """One author with two works: W0 (2015) cited 5 times in 2016, 7 in 2018, 9 in 2020; W1 (2019) cited 4 times
    in 2021. Before 2019 the author had 12 citations; the submitted rule (yearly counts keyed by the publication
    year of the cited work) gives 21, i.e. it includes the 2020 citations of W0."""
    totals, years, counts = [21, 4], [[2016, 2018, 2020], [2021]], [[5, 7, 9], [4]]
    yc = YearlyCounts(totals, clip_years(years), counts, 2012)
    ent, wyear, widx = np.array([0, 0]), np.array([2015, 2019]), np.array([0, 1])
    cit, cnt = before_table(ent, wyear, widx, yc, 1, 2012)
    col = lambda y: y - GRID0  # noqa: E731
    assert cit[0, col(2017)] == 5 and cnt[0, col(2017)] == 1
    assert cit[0, col(2019)] == 12 and cnt[0, col(2019)] == 1      # W1 is not published before 2019
    assert cit[0, col(2022)] == 25 and cnt[0, col(2022)] == 2
    hind = YearlyCounts([25], [[2015, 2019]], [[21, 4]], 2012)     # author record: counts by publication year
    assert hind.before([0], 2019)[0] == 21                         # the defect the exact statistic removes


def test_artefact_years_are_not_counted_as_past():
    yc = YearlyCounts([10], clip_years([[2015, 10000]]), [[4, 6]], 2012)
    assert yc.before([0], 2017)[0] == 4                            # the 6 citations dated 10000 are removed


# ---------------------------------------------------------------- statistics
def test_vectorised_metrics_match_reference_implementations():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.2).astype(float)
    p = rng.normal(size=(3, 300)) + y
    assert np.allclose(S.auc_rows(y[None], p), [roc_auc_score(y, r) for r in p])
    assert np.allclose(S.ap_rows(y[None], p), [average_precision_score(y, r) for r in p])
    t = rng.normal(size=300)
    assert np.allclose(S.spearman_rows(t[None], p), [spearmanr(t, r).correlation for r in p])


def test_holm_adjustment():
    assert S.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_corrected_t_test_is_wider_than_the_naive_one():
    import pandas as pd

    rows = []
    rng = np.random.default_rng(1)
    for rep in range(2):
        for fold in range(6):
            base = rng.normal(0.6, 0.02)
            for seed in range(2):
                rows.append(dict(arm="a", protocol="kfold", rep=rep, fold=fold, init_seed=seed, m_spearman=base + 0.01))
                rows.append(dict(arm="b", protocol="kfold", rep=rep, fold=fold, init_seed=seed,
                                 m_spearman=base + rng.normal(0, 0.004)))
    c = S.contrast_kfold(pd.DataFrame(rows), "a", "b", "spearman")
    d = pd.DataFrame(rows).groupby(["arm", "rep", "fold"]).m_spearman.mean().unstack(0)
    naive_se = (d.a - d.b).std(ddof=1) / np.sqrt(12)
    assert (c["ci"][1] - c["ci"][0]) > 2 * 2.2 * naive_se * 1.5
    assert c["diff"] == pytest.approx((d.a - d.b).mean())


def test_a_job_run_twice_is_read_once_with_its_last_record(tmp_path, monkeypatch):
    """Two processes can run the same job; the stored predictions are those of the last run, so the readers of the
    run records must keep the last line of an identifier (tables and paired statistics then use the same run)."""
    import json

    import experiments_rev1 as E

    (tmp_path / "runs").mkdir()
    rec = lambda i, v: {"id": i, "job": {"arm": "gat"}, "epochs": 1, "time_s": 0.0, "n_train": 1, "n_test": 1,  # noqa: E731
                        "metrics": {"spearman": v}, "val_metrics": {}}
    lines = [rec("a", 0.10), rec("b", 0.20), rec("a", 0.30)]
    (tmp_path / "runs" / "toy.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    monkeypatch.setattr(E, "RES", tmp_path)
    monkeypatch.setattr(S, "RES", tmp_path)
    got = {r["id"]: r["metrics"]["spearman"] for r in E.read_runs("toy")}
    assert got == {"a": 0.30, "b": 0.20}
    df = S.runs("toy")
    assert sorted(df.id) == ["a", "b"] and float(df.set_index("id").m_spearman["a"]) == 0.30
