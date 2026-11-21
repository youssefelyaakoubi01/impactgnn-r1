"""Paired comparisons and grouped metrics computed from the per-paper predictions of the revision runs.

Two designs, matching the protocols:

- **Fixed test set** (``mature``, ``retro``): two arms are run on the same validation partitions and seeds and
  predict the same test papers. The statistic is the difference of the metric averaged over the paired runs; its
  interval comes from a paired bootstrap over the test papers (the same resampled papers for both arms and all
  runs). A second bootstrap resamples whole topics, which accounts for the dependence between papers of a topic.
- **Repeated K-fold** (``kfold``, ``topicfold``): the metric is averaged over the initialisation seeds of each
  test fold; the folds give paired differences, tested with the corrected resampled t-test of Nadeau and Bengio
  (the training sets of different folds overlap, so the ordinary paired t-test is too optimistic).

A difference is called *equivalent* when its 95 % interval lies inside the margin (default +-0.01), and
*different* when the interval excludes zero. ``holm`` adjusts the p-values of a family of primary contrasts.
"""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats as sps

from config import ROOT

RES = ROOT / "results" / "rev1"
MARGIN = 0.01
N_BOOT = 2000


# ====================================================================== loading
def runs(grid: str) -> pd.DataFrame:
    """One row per run of a grid: job fields, test metrics (``m_*``), validation metrics (``v_*``)."""
    rows = []
    for line in (RES / "runs" / f"{grid}.jsonl").read_text().splitlines():
        r = json.loads(line)
        row = dict(r["job"], id=r["id"], epochs=r["epochs"], time_s=r["time_s"], n_train=r["n_train"],
                   n_test=r["n_test"])
        row.update({f"m_{k}": v for k, v in (r.get("metrics") or {}).items()})
        row.update({f"v_{k}": v for k, v in (r.get("val_metrics") or {}).items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    # a job run by two processes has two lines: the last one is the run whose predictions are stored
    return df.drop_duplicates("id", keep="last").reset_index(drop=True) if len(df) else df


@lru_cache(maxsize=4096)
def preds(run_id: str) -> dict:
    z = np.load(RES / "preds" / f"{run_id}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


@lru_cache(maxsize=1)
def node_table() -> pd.DataFrame:
    """Identifier, year, origin, field and sampled topic of every paper node (cached from the rev1 graph)."""
    path = RES / "nodes.parquet"
    if not path.exists():
        import torch

        g = torch.load(ROOT / "data" / "processed" / "full" / "rev1" / "graph.pt", weights_only=False)
        deg = np.bincount(g["edge_index"][1].numpy()[(g["submitted"][g["edge_index"][0]]
                                                       & g["submitted"][g["edge_index"][1]]).numpy()],
                          minlength=len(g["ids"]))
        pd.DataFrame({"id": g["ids"], "title": g["titles"], "year": g["year"].numpy(), "origin": g["origin"],
                      "field": g["field"],
                      "subfield": g["subfield"], "topic": g["sampled_topic"], "topic_group": g["topic_group"],
                      "n_refs": g["n_refs"].numpy(), "in_graph_refs": deg, "c3": g["c3"].numpy(),
                      "c5": g["c5"].numpy(), "m3": g["m3"].numpy().astype(bool),
                      "m5": g["m5"].numpy().astype(bool)}).to_parquet(path)
    return pd.read_parquet(path).set_index("id")


def node_table_holdout(g: dict) -> pd.DataFrame:
    """Identifier -> sampled topic for the graph that contains the fresh holdout papers."""
    return pd.DataFrame({"id": g["ids"], "topic": g["sampled_topic"]}).set_index("id")


# ====================================================================== vectorised metrics (rows = runs)
def spearman_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    ry, rp = sps.rankdata(y, axis=-1), sps.rankdata(p, axis=-1)
    ry, rp = ry - ry.mean(-1, keepdims=True), rp - rp.mean(-1, keepdims=True)
    return (ry * rp).sum(-1) / np.sqrt((ry ** 2).sum(-1) * (rp ** 2).sum(-1))


def auc_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    r = sps.rankdata(p, axis=-1)
    n_pos = y.sum(-1)
    n_neg = y.shape[-1] - n_pos
    return ((r * y).sum(-1) - n_pos * (n_pos + 1) / 2) / np.maximum(n_pos * n_neg, 1)


def ap_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    order = np.argsort(-p, axis=-1, kind="stable")
    ys = np.take_along_axis(np.broadcast_to(y, p.shape), order, -1)
    prec = np.cumsum(ys, -1) / np.arange(1, ys.shape[-1] + 1)
    return (prec * ys).sum(-1) / np.maximum(ys.sum(-1), 1)


METRICS = {"spearman": spearman_rows, "roc_auc": auc_rows, "ap": ap_rows}


def primary_metric(task: str) -> str:
    return "spearman" if task == "reg" else "roc_auc"


# ====================================================================== fixed test set: paired bootstrap
PAIR_KEYS = ["split_seed", "init_seed"]


def _stack(df: pd.DataFrame, arm: str, part: str = "te"):
    """Predictions of the runs of an arm (rows sorted by the pairing key), targets and paper identifiers."""
    d = df[df.arm == arm].copy()
    for k in PAIR_KEYS:
        if k not in d:
            d[k] = 0
    d = d.fillna({k: 0 for k in PAIR_KEYS}).sort_values(PAIR_KEYS)
    P = np.stack([preds(i)[f"{part}_p"] for i in d.id])
    first = preds(d.id.iloc[0])
    return P, first[f"{part}_y"], first[f"{part}_ids"], list(zip(*(d[k] for k in PAIR_KEYS)))


def contrast_fixed(df: pd.DataFrame, a: str, b: str, metric: str, n_boot: int = N_BOOT, seed: int = 0,
                   cluster: bool = False, margin: float = MARGIN, subset: np.ndarray | None = None,
                   topic_of: pd.Series | None = None, resample_runs: bool = False) -> dict:
    """Arm ``a`` minus arm ``b`` on a common test set; ``cluster`` resamples topics instead of papers;
    ``subset`` restricts the test papers (boolean mask). ``resample_runs`` also resamples the paired runs, so that
    the interval includes the variability of training; the runs are drawn with a second generator, so the papers
    drawn are those of the interval without this option."""
    Pa, y, ids, ka = _stack(df, a)
    Pb, yb, idb, kb = _stack(df, b)
    assert ka == kb, f"runs of {a} and {b} are not paired"
    assert np.array_equal(ids, idb) and np.array_equal(y, yb), "different test sets"
    if subset is not None:
        Pa, Pb, y, ids = Pa[:, subset], Pb[:, subset], y[subset], ids[subset]
    fn = METRICS[metric]
    ma, mb = fn(y[None], Pa), fn(y[None], Pb)
    rng = np.random.default_rng(seed)
    rng_runs = np.random.default_rng(seed + 1)
    n, n_runs = len(y), len(Pa)
    if cluster:
        topic = (node_table().topic if topic_of is None else topic_of).loc[ids].to_numpy()
        groups = [np.flatnonzero(topic == t) for t in np.unique(topic)]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        if cluster:
            idx = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        else:
            idx = rng.integers(n, size=n)
        d = fn(y[idx][None], Pa[:, idx]) - fn(y[idx][None], Pb[:, idx])
        boots[i] = d[rng_runs.integers(n_runs, size=n_runs)].mean() if resample_runs else d.mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"a": a, "b": b, "metric": metric, "mean_a": float(ma.mean()), "mean_b": float(mb.mean()),
            "sd_a": float(ma.std(ddof=1)), "sd_b": float(mb.std(ddof=1)), "diff": float((ma - mb).mean()),
            "ci": [float(lo), float(hi)], "p": float(min(1.0, max(p, 1.0 / n_boot))), "n_runs": int(len(ma)),
            "n_test": int(n), "equivalent": bool(lo > -margin and hi < margin), "different": bool(lo > 0 or hi < 0),
            "design": ("topic" if cluster else "paper") + (" and run bootstrap" if resample_runs else " bootstrap")}


# ====================================================================== repeated K-fold: corrected t-test
def fold_metrics(df: pd.DataFrame, arm: str, metric: str) -> pd.Series:
    """Metric per test fold (index: rep, fold), averaged over the initialisation seeds."""
    d = df[df.arm == arm]
    return d.groupby(["rep", "fold"])[f"m_{metric}"].mean()


def contrast_kfold(df: pd.DataFrame, a: str, b: str, metric: str, margin: float = MARGIN) -> dict:
    fa, fb = fold_metrics(df, a, metric), fold_metrics(df, b, metric)
    assert fa.index.equals(fb.index), f"folds of {a} and {b} differ"
    d = (fa - fb).to_numpy()
    j = len(d)
    k = df[df.arm == a].fold.nunique()
    var = d.var(ddof=1)
    se = np.sqrt((1.0 / j + 1.0 / (k - 1)) * var)        # n_test / n_train = 1 / (k - 1) for K-fold
    t = d.mean() / se if se > 0 else 0.0
    crit = sps.t.ppf(0.975, j - 1)
    lo, hi = d.mean() - crit * se, d.mean() + crit * se
    return {"a": a, "b": b, "metric": metric, "mean_a": float(fa.mean()), "mean_b": float(fb.mean()),
            "sd_a": float(fa.std(ddof=1)), "sd_b": float(fb.std(ddof=1)), "diff": float(d.mean()),
            "ci": [float(lo), float(hi)], "p": float(2 * sps.t.sf(abs(t), j - 1)), "n_folds": int(j),
            "share_positive": float((d > 0).mean()), "equivalent": bool(lo > -margin and hi < margin),
            "different": bool(lo > 0 or hi < 0), "design": "corrected resampled t-test"}


def contrast(df: pd.DataFrame, a: str, b: str, metric: str, **kw) -> dict:
    proto = df[df.arm == a].protocol.iloc[0]
    if proto in ("kfold", "topicfold"):
        return contrast_kfold(df, a, b, metric, **{k: v for k, v in kw.items() if k == "margin"})
    return contrast_fixed(df, a, b, metric, **kw)


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


# ====================================================================== summaries
def arm_summary(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Mean and standard deviation of the test metrics per arm. For K-fold protocols the unit is the test fold
    (seed-averaged); for fixed test sets it is the run."""
    out = []
    for arm, d in df.groupby("arm", sort=False):
        row = {"arm": arm, "n": len(d)}
        for m in metrics:
            if f"m_{m}" not in d:
                continue
            v = d.groupby(["rep", "fold"])[f"m_{m}"].mean() if d.protocol.iloc[0] in ("kfold", "topicfold") else d[f"m_{m}"]
            row[m], row[f"{m}_sd"] = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out.append(row)
    return pd.DataFrame(out).set_index("arm")


def oof(df: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Out-of-fold (or test) predictions of an arm, averaged over the runs that predict each paper."""
    acc = defaultdict(list)
    y = {}
    for i in df[df.arm == arm].id:
        z = preds(i)
        for pid, p, t in zip(z["te_ids"], z["te_p"], z["te_y"]):
            acc[pid].append(p)
            y[pid] = t
    return pd.DataFrame({"p": {k: float(np.mean(v)) for k, v in acc.items()}, "y": y})


def by_group(df: pd.DataFrame, arm: str, col: str, metric: str, min_n: int = 30, min_pos: int = 5) -> pd.DataFrame:
    """Metric of an arm within each group of papers (topic, field, year, ...), from the averaged predictions."""
    o = oof(df, arm).join(node_table()[[col]])
    rows = []
    for gname, d in o.groupby(col):
        if len(d) < min_n or (metric != "spearman" and (d.y.sum() < min_pos or d.y.sum() == len(d))):
            continue
        rows.append({col: gname, "n": len(d), metric: float(METRICS[metric](d.y.to_numpy()[None], d.p.to_numpy()[None])[0])})
    return pd.DataFrame(rows).set_index(col)
