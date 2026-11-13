"""Evaluation protocols of the revision. ``make_split`` returns the training, validation and test papers and the
mask of the nodes that exist when the model is trained.

``mature``     prediction date T. Training and validation papers are those whose citation window y .. y+k closed
               before T (y <= T-1-k), so every label used for fitting was observable at T; a year-stratified random
               share of them is the validation set (``split_seed``). The test papers are the sampled papers
               published from T on. The training graph contains only papers published before T.
``retro``      the temporal split of the submitted article (train < T-1, validation = T-1, test >= T). The citation
               windows of the training and validation papers extend beyond T, so this is a retrospective evaluation.
``kfold``      repeated stratified K-fold over all labelled papers (transductive, like the random split of the
               submitted article): ``rep`` and ``fold`` choose the test fold, 15 % of the rest is the validation set.
``topicfold``  K-fold over the sampled topics: every paper of a topic is in the same fold (cross-topic test).
``group``      train on one topic group (laboratory-related or open-corpus topics), test on the other.
``legacy``     the random 70/15/15 split of the submitted code (``seed``), kept to reproduce the submitted numbers.
``deploy``     laboratory case study of the submitted article: all labelled years for training, the last one for
               validation, the laboratory papers (held out of the graph) as targets.
"""

from __future__ import annotations

import numpy as np

from config import Config

N_FOLDS = 6
VAL_SHARE = 0.15


def _origin(g: dict) -> np.ndarray:
    return np.asarray(g["origin"])


def _strat_take(idx: np.ndarray, years: np.ndarray, share: float, rng: np.random.Generator) -> np.ndarray:
    """Year-stratified random subset of ``idx`` (boolean mask over ``idx``)."""
    take = np.zeros(len(idx), dtype=bool)
    for y in np.unique(years[idx]):
        pos = np.flatnonzero(years[idx] == y)
        rng.shuffle(pos)
        take[pos[:round(share * len(pos))]] = True
    return take


def _folds(idx: np.ndarray, years: np.ndarray, rng: np.random.Generator, k: int = N_FOLDS) -> np.ndarray:
    fold = np.zeros(len(idx), dtype=np.int64)
    for y in np.unique(years[idx]):
        pos = np.flatnonzero(years[idx] == y)
        rng.shuffle(pos)
        fold[pos] = np.arange(len(pos)) % k
    return fold


def make_split(g: dict, job: dict, cfg: Config) -> dict:
    k, proto = job["k"], job["protocol"]
    years = g["year"].cpu().numpy()
    lab = g[f"m{k}"].bool().cpu().numpy()
    origin = _origin(g)
    sample = origin == "sample"
    n = len(years)
    train_nodes = np.ones(n, dtype=bool)
    idx_lab = np.flatnonzero(lab)

    if proto == "legacy":
        from experiments import random_split

        tr, va, te = (t.cpu().numpy() for t in random_split(g, k, job["seed"], cfg.split))
    elif proto == "kfold":
        fold = _folds(idx_lab, years, np.random.default_rng(1000 + job["rep"]))
        te = idx_lab[fold == job["fold"]]
        rest = idx_lab[fold != job["fold"]]
        v = _strat_take(rest, years, VAL_SHARE, np.random.default_rng(100_000 + 100 * job["rep"] + job["fold"]))
        tr, va = rest[~v], rest[v]
    elif proto == "topicfold":
        idx = np.flatnonzero(lab & sample)
        topic = np.asarray(g["sampled_topic"])[idx]
        topics = np.unique(topic)
        perm = np.random.default_rng(2000 + job["rep"]).permutation(len(topics))
        fold_of = {t: i % N_FOLDS for i, t in zip(range(len(topics)), topics[perm])}
        fold = np.array([fold_of[t] for t in topic])
        te = idx[fold == job["fold"]]
        rest = idx[fold != job["fold"]]
        v = _strat_take(rest, years, VAL_SHARE, np.random.default_rng(200_000 + 100 * job["rep"] + job["fold"]))
        tr, va = rest[~v], rest[v]
    elif proto == "group":
        grp = np.asarray(g["topic_group"])
        idx = np.flatnonzero(lab & sample)
        te = idx[grp[idx] == job["test_group"]]
        rest = idx[grp[idx] != job["test_group"]]
        v = _strat_take(rest, years, VAL_SHARE, np.random.default_rng(300_000 + job.get("split_seed", 0)))
        tr, va = rest[~v], rest[v]
    elif proto == "mature":
        T = job["T"]
        pool = np.flatnonzero(lab & (years <= T - 1 - k))
        v = _strat_take(pool, years, VAL_SHARE, np.random.default_rng(400_000 + job.get("split_seed", 0)))
        tr, va = pool[~v], pool[v]
        te = np.flatnonzero(lab & sample & (years >= T))
        train_nodes = years < T
    elif proto == "retro":
        T = job["T"]
        tr = np.flatnonzero(lab & (years < T - 1))
        va = np.flatnonzero(lab & (years == T - 1))
        te = np.flatnonzero(lab & sample & (years >= T))
        train_nodes = years < T
    elif proto == "deploy":
        ymax = int(years[lab].max())
        tr, va = np.flatnonzero(lab & (years < ymax)), np.flatnonzero(lab & (years == ymax))
        te = np.flatnonzero(origin == "seed_lab")
        train_nodes = (years <= ymax) & (origin != "seed_lab")
    else:
        raise ValueError(f"unknown protocol: {proto}")

    if job.get("train_subsample"):
        rng = np.random.default_rng(500_000 + job.get("split_seed", 0))
        tr = np.sort(rng.choice(tr, size=min(len(tr), job["train_subsample"]), replace=False))
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te)), "overlapping split"
    if proto in ("mature", "retro"):
        assert train_nodes[tr].all() and train_nodes[va].all() and not train_nodes[te].any()
    if proto == "mature":
        assert (years[np.concatenate([tr, va])] + k <= job["T"] - 1).all(), "immature label in the training set"
    return {"tr": tr, "va": va, "te": te, "train_nodes": train_nodes}
