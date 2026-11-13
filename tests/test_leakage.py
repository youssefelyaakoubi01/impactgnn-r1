"""Unit checks of the label window, time-safe counts and the leakage rules of the graphs."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import subgraph

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_graph import YearlyCounts, check_hetero, citation_edges, label_complete, label_counts, metapath_edges  # noqa: E402
from config import get_config  # noqa: E402
from experiments import inject  # noqa: E402

CFG = get_config("small")


def test_label_window_inclusive():
    years, counts = [2018, 2019, 2021, 2022], [5, 3, 2, 7]
    assert label_counts(years, counts, 2018, 3) == 5 + 3 + 2          # 2018..2021
    assert label_counts(years, counts, 2018, 3, inclusive=False) == 8  # 2018..2020
    assert label_complete(2022, 3, CFG) and not label_complete(2023, 3, CFG)
    assert label_complete(2020, 5, CFG) and not label_complete(2021, 5, CFG)
    assert not label_complete(2016, 3, CFG)  # before T0


def test_counts_before_year():
    yc = YearlyCounts([100], [[2012, 2015, 2020]], [[10, 20, 30]], first_year=2012)
    # 40 citations happened before 2012 (100 - 60 covered)
    assert yc.before([0], [2016])[0] == 100 - 30
    assert yc.before([0], [2015])[0] == 100 - 50
    assert yc.before([0], [2005])[0] == 40      # clamped to "before 2012"
    assert yc.window([0], [2015], [2020])[0] == 50


def test_edges_never_from_later_paper():
    nodes = pd.DataFrame({"id": ["A", "B", "C"], "year": [2010, 2018, 2015],
                          "refs": [[], ["A", "C"], ["B"]]})  # C (2015) "cites" B (2018): must be dropped
    meta_counts = YearlyCounts([5, 3, 1], [[], [], []], [[], [], []], first_year=2012)
    ei, attr, dropped = citation_edges(nodes, meta_counts, {"A": 0, "B": 1, "C": 2}, CFG)
    years = nodes.year.to_numpy()
    assert dropped == 1
    assert (years[ei[0]] <= years[ei[1]]).all()
    assert set(map(tuple, ei.T)) == {(0, 1), (2, 1)}
    assert attr.shape == (2, 2)


def test_metapath_rules():
    rng = np.random.default_rng(0)
    years = rng.integers(2005, 2023, 400)
    groups = [rng.choice(400, size=rng.integers(2, 30), replace=False) for _ in range(80)]
    for rule in ("t0", "temporal"):
        e = metapath_edges(groups, years, rule, CFG.t0, 5, rng)
        assert e.shape[1] > 0
        if rule == "t0":
            assert (years[e[0]] < CFG.t0).all()
        else:
            assert (years[e[0]] < years[e[1]]).all()
        assert (np.bincount(e[1], minlength=400) <= 5).all()
    het = {f"rev_safe_{n}": np.zeros((2, 0), int) for n in ("author", "venue", "keyword")}
    for n in ("author", "venue", "keyword"):
        het[f"mp_t0_{n}"] = metapath_edges(groups, years, "t0", CFG.t0, 5, rng)
        het[f"mp_temporal_{n}"] = metapath_edges(groups, years, "temporal", CFG.t0, 5, rng)
    check_hetero(het, years, CFG.t0)


def test_coldstart_training_graph_excludes_test_papers():
    years = torch.tensor([2010, 2018, 2019, 2021, 2022, 2020])
    ei = torch.tensor([[0, 1, 2, 0, 1, 5], [1, 2, 3, 4, 5, 3]])
    T = 2021
    keep = years < T
    sub_ei, _ = subgraph(keep, ei, None, relabel_nodes=False, num_nodes=6)
    assert not torch.isin(sub_ei, torch.nonzero(~keep).squeeze(1)).any()
    assert sub_ei.shape[1] == 3


def test_injection_adds_intended_weight():
    W = np.ones((60, 60))
    out = W.sum(1)
    W2, out2, labels, groups, selfers = inject(W, out, CFG, 0.5, np.random.default_rng(1))
    for grp in groups:
        i, j = grp[0], grp[1]
        assert W2[i, j] - W[i, j] == max(1, round(0.5 * out[i] / (CFG.cartel_size - 1)))
    for i in selfers:
        assert W2[i, i] - W[i, i] == max(1, round(0.5 * out[i]))
    assert (labels == 1).sum() == CFG.n_cartels * CFG.cartel_size and (labels == 2).sum() == CFG.n_selfcit
    untouched = labels == 0
    assert np.allclose(W2[np.ix_(untouched, untouched)], 1)
