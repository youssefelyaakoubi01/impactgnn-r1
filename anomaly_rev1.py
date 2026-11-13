"""Synthetic anomaly detection on the venue citation graph, revision (rev1).

``python anomaly_rev1.py [--draws 30]``  ->  ``results/rev1/anomaly_runs.jsonl`` and ``results/rev1/anomaly.json``

The submitted experiment injected reciprocal cartels and self-citation, which are exactly what the reciprocity and
self-citation heuristics measure. This version keeps those two patterns and adds patterns that the heuristics are
not built for, a detector that is not tied to one pattern, and size-matched controls:

patterns (one per injected graph; ``f`` is the injected share of a venue's outgoing citations)
  ``cartel``          5 groups of 4 venues add reciprocal citations                       (submitted)
  ``selfcit``         10 venues add citations to themselves                               (submitted)
  ``cartel_rewire``   as ``cartel`` but the citations are moved from existing links, so the venue sizes do not change
  ``cartel_large``    2 groups of 12 venues, same budget per venue, i.e. weak links between many partners
  ``cartel_camo``     as ``cartel``, plus the same volume sent to random outside venues (dilutes the reciprocal share)
  ``ring``            5 one-directional rings of 4 venues (A -> B -> C -> D -> A): no reciprocal link is created
  ``stacking``        10 donor venues each send citations to one recipient venue (one-directional citation stacking)

detectors
  ``heur_self_ratio``, ``heur_reciprocity``, ``heur_max``    the heuristics of the submitted article
  ``null_excess``     largest Poisson z-score of a venue's citation links (outgoing or incoming, self-links
                      included) against the degree-corrected null model  E_ij = out_i * in_j / total
  ``dominant_*``, ``gae_*``, ``iforest_*``                   as submitted (content / content + structure features)

For each detector and pattern: ROC-AUC and average precision against all other venues, the prevalence (chance
level of the average precision), and the ROC-AUC against controls matched on size (the ten venues closest in
outgoing citations to each anomalous venue). The injection draw and the detector seed are separate.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from config import ROOT, get_config, resolve_device, setup_logging
from experiments import adjacency_target, venue_dense, venue_structure
from models import fit_detector

logger = logging.getLogger("anomaly_rev1")
RES = ROOT / "results" / "rev1"
INTENSITIES = {"mild": 0.10, "medium": 0.25, "strong": 0.50}
PATTERNS = ["cartel", "selfcit", "cartel_rewire", "cartel_large", "cartel_camo", "ring", "stacking"]


# ====================================================================== injections
def _move(W: np.ndarray, i: int, targets: list[int], amount: float, rng) -> None:
    """Budget-preserving: take ``amount`` citations from the existing out-links of venue i (proportionally, outside
    ``targets`` and i itself) and give them to ``targets`` in equal parts."""
    row = W[i].copy()
    row[targets] = 0
    row[i] = 0
    avail = row.sum()
    if avail <= 0:
        return
    amount = min(amount, avail)
    W[i] -= row * (amount / avail)
    W[i, targets] += amount / len(targets)


def inject(W: np.ndarray, out_total: np.ndarray, pattern: str, f: float, rng: np.random.Generator):
    """Returns the modified matrix, the new out totals and the 0/1 labels of the venues."""
    n = W.shape[0]
    W2, out2 = W.astype(np.float64).copy(), out_total.astype(np.float64).copy()
    labels = np.zeros(n, dtype=np.int64)
    perm = rng.permutation(n)

    def groups(n_groups, size):
        return [perm[i * size:(i + 1) * size] for i in range(n_groups)]

    if pattern in ("cartel", "cartel_rewire", "cartel_camo", "cartel_large"):
        gs = groups(2, 12) if pattern == "cartel_large" else groups(5, 4)
        used = sum(len(g) for g in gs)
        outside = perm[used:]
        for grp in gs:
            m = len(grp)
            for i in grp:
                others = [j for j in grp if j != i]
                budget = max(m - 1, round(f * out_total[i]))
                if pattern == "cartel_rewire":
                    _move(W2, i, others, budget, rng)
                else:
                    W2[i, others] += budget / (m - 1)
                    out2[i] += budget
                    if pattern == "cartel_camo":
                        tgt = rng.choice(outside, size=8, replace=False)
                        W2[i, tgt] += budget / 8
                        out2[i] += budget
            labels[grp] = 1
    elif pattern == "selfcit":
        for i in perm[:10]:
            add = max(1, round(f * out_total[i]))
            W2[i, i] += add
            out2[i] += add
            labels[i] = 1
    elif pattern == "ring":
        for grp in groups(5, 4):
            for a, i in enumerate(grp):
                j = grp[(a + 1) % len(grp)]
                add = max(1, round(f * out_total[i]))
                W2[i, j] += add
                out2[i] += add
            labels[grp] = 1
    elif pattern == "stacking":
        for a in range(10):
            donor, recipient = perm[2 * a], perm[2 * a + 1]
            add = max(1, round(f * out_total[donor]))
            W2[donor, recipient] += add
            out2[donor] += add
            labels[[donor, recipient]] = 1
    else:
        raise ValueError(pattern)
    return W2, out2, labels


# ====================================================================== detectors
def null_excess(W: np.ndarray) -> np.ndarray:
    """Largest Poisson z-score of the links of each venue against E_ij = out_i * in_j / total (graph-internal
    totals). It is high for any pair exchanging far more citations than the sizes of the two venues predict."""
    out_i, in_j, tot = W.sum(1), W.sum(0), max(W.sum(), 1.0)
    E = np.outer(out_i, in_j) / tot
    z = (W - E) / np.sqrt(E + 1.0)
    return np.maximum(z.max(1), z.max(0))


def scores_for(W: np.ndarray, out_total: np.ndarray, content: np.ndarray, seed: int, device) -> dict[str, np.ndarray]:
    st = venue_structure(W, out_total)
    S = np.column_stack(list(st.values()))
    A = torch.as_tensor(adjacency_target(W), dtype=torch.float32, device=device)
    rank = lambda v: pd.Series(v).rank(pct=True).to_numpy()  # noqa: E731
    scores = {"heur_self_ratio": st["self_ratio"], "heur_reciprocity": st["reciprocity"],
              "heur_max": np.maximum(rank(st["self_ratio"]), rank(st["reciprocity"])), "null_excess": null_excess(W)}
    for fs, X in (("content", content), ("full", np.column_stack([content, S]))):
        Xs = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
        xt = torch.as_tensor(Xs, dtype=torch.float32, device=device)
        scores[f"dominant_{fs}"] = fit_detector("dominant", xt, A, seed).cpu().numpy()
        scores[f"gae_{fs}"] = fit_detector("gae", xt, A, seed).cpu().numpy()
        scores[f"iforest_{fs}"] = -IsolationForest(n_estimators=300, random_state=seed).fit(Xs).score_samples(Xs)
    return scores


def matched_controls(labels: np.ndarray, size: np.ndarray, n_per: int = 10) -> np.ndarray:
    """Indices of the non-anomalous venues closest in size to each anomalous venue (without replacement)."""
    neg = np.flatnonzero(labels == 0)
    order = neg[np.argsort(size[neg])]
    s_sorted = size[order]
    taken = np.zeros(len(order), dtype=bool)
    for i in np.flatnonzero(labels == 1):
        pos = np.searchsorted(s_sorted, size[i])
        lo, hi, got = pos - 1, pos, 0
        while got < n_per and (lo >= 0 or hi < len(order)):
            dl = abs(s_sorted[lo] - size[i]) if lo >= 0 else np.inf
            dh = abs(s_sorted[hi] - size[i]) if hi < len(order) else np.inf
            j = lo if dl <= dh else hi
            if dl <= dh:
                lo -= 1
            else:
                hi += 1
            if not taken[j]:
                taken[j] = True
                got += 1
    return order[taken]


def evaluate(scores: dict, labels: np.ndarray, size: np.ndarray) -> dict:
    ctrl = matched_controls(labels, size)
    pos = np.flatnonzero(labels == 1)
    sub = np.concatenate([pos, ctrl])
    out = {}
    for det, s in scores.items():
        s = np.nan_to_num(s)
        out[det] = {"roc_auc": float(roc_auc_score(labels, s)), "ap": float(average_precision_score(labels, s)),
                    "roc_auc_matched": float(roc_auc_score(labels[sub], s[sub]))}
    return out


# ====================================================================== external check on the real graph
def external_check(cfg, device, n_seeds: int = 5, n_perm: int = 5000) -> dict:
    """Do the venues that Clarivate suppressed from the Journal Citation Reports (2018-2025 releases; reversed
    suppressions excluded) rank high under each detector on the real, un-injected venue graph? Descriptive only:
    the graph is built from a topic sample, and a suppression is an editorial decision, not a ground truth."""
    vg = torch.load(cfg.processed_dir / "venue_graph.pt", weights_only=False)
    W, out_total = venue_dense(vg), vg["out_total"].numpy().astype(np.float64)
    content = np.column_stack([vg["content"].numpy(), np.log1p(vg["works_count"].numpy())])
    issn = pd.read_parquet(cfg.raw_dir / "rev1_venue_issn.parquet")
    of_issn = {x: r.id for r in issn.itertuples() for x in set([r.issn_l] + list(r.issn)) if x}
    jcr = pd.read_csv(ROOT / "external" / "jcr_suppressed.csv", dtype=str).fillna("")
    jcr = jcr[(jcr.jcr_year.astype(int) <= 2025) & ~jcr.reason.str.contains("reversed")]
    vid = {v: i for i, v in enumerate(vg["venue_ids"])}
    reason = {}
    for r in jcr.itertuples():
        for x in r.issn.replace(" ", "").split(";"):
            if x in of_issn and of_issn[x] in vid:
                kind = "stacking" if "stacking" in r.reason else "self-citation"
                reason.setdefault(vid[of_issn[x]], set()).add(kind)
    labels = np.zeros(len(vid), dtype=np.int64)
    labels[list(reason)] = 1
    per_seed = [scores_for(W, out_total, content, 2000 + s, device) for s in range(n_seeds)]
    rank = lambda v: pd.Series(np.nan_to_num(v)).rank(pct=True).to_numpy()  # noqa: E731
    scores = {d: np.mean([rank(ps[d]) for ps in per_seed], 0) for d in per_seed[0]}
    size = np.log1p(out_total)
    ctrl = matched_controls(labels, size)
    sub = np.concatenate([np.flatnonzero(labels == 1), ctrl])
    rng = np.random.default_rng(0)
    out = {"n_suppressed_in_graph": int(labels.sum()), "n_venues": int(len(labels)),
           "by_reason": {k: int(sum(k in v for v in reason.values())) for k in ("self-citation", "stacking")},
           "median_out_links_suppressed": float(np.median(out_total[labels == 1])),
           "median_out_links_all": float(np.median(out_total)), "detectors": {}}
    for det, s in scores.items():
        auc = roc_auc_score(labels, s)
        null = np.array([roc_auc_score(rng.permutation(labels), s) for _ in range(n_perm)])
        out["detectors"][det] = {
            "roc_auc": float(auc), "p_perm": float(max((null >= auc).mean(), 1 / n_perm)),
            "roc_auc_matched": float(roc_auc_score(labels[sub], s[sub])),
            "median_percentile": float(np.median(s[labels == 1])),
            "in_top_decile": int((s[labels == 1] >= 0.9).sum()),
            "roc_auc_self": float(roc_auc_score([1 if "self-citation" in reason.get(i, ()) else 0
                                                 for i in range(len(labels))], s))}
    (RES / "anomaly_external.json").write_text(json.dumps(out, indent=1))
    return out


# ====================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--profile", default="full")
    ap.add_argument("--external", action="store_true", help="only the check against JCR-suppressed journals")
    args = ap.parse_args()
    cfg = get_config(args.profile)
    setup_logging("anomaly_rev1", cfg)
    device = resolve_device(cfg)
    if args.external:
        RES.mkdir(parents=True, exist_ok=True)
        res = external_check(cfg, device)
        logger.info("external check: %d suppressed journals among %d venues", res["n_suppressed_in_graph"],
                    res["n_venues"])
        for det, m in res["detectors"].items():
            logger.info("  %-18s AUC %.3f (p %.3f)  matched %.3f  median percentile %.2f  top decile %d", det,
                        m["roc_auc"], m["p_perm"], m["roc_auc_matched"], m["median_percentile"], m["in_top_decile"])
        return
    vg = torch.load(cfg.processed_dir / "venue_graph.pt", weights_only=False)
    W, out_total = venue_dense(vg), vg["out_total"].numpy().astype(np.float64)
    content = np.column_stack([vg["content"].numpy(), np.log1p(vg["works_count"].numpy())])
    RES.mkdir(parents=True, exist_ok=True)
    path = RES / "anomaly_runs.jsonl"
    done = set()
    if path.exists():
        done = {(r["pattern"], r["intensity"], r["draw"]) for r in map(json.loads, path.read_text().splitlines())}
    t0, n = time.time(), 0
    with open(path, "a") as fh:
        for draw in range(args.draws):
            for pattern in PATTERNS:
                for lvl, f in INTENSITIES.items():
                    if (pattern, lvl, draw) in done:
                        continue
                    rng = np.random.default_rng([20261112, draw, PATTERNS.index(pattern)])   # same venues at every level
                    W2, out2, labels = inject(W, out_total, pattern, f, rng)
                    scores = scores_for(W2, out2, content, 1000 + draw, device)
                    rec = {"pattern": pattern, "intensity": lvl, "f": f, "draw": draw, "n_anomalous": int(labels.sum()),
                           "prevalence": float(labels.mean()), "metrics": evaluate(scores, labels, np.log1p(out2))}
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    n += 1
            logger.info("draw %d/%d done (%d runs, %.0f s)", draw + 1, args.draws, n, time.time() - t0)

    runs = [json.loads(line) for line in path.read_text().splitlines()]
    acc = defaultdict(lambda: defaultdict(list))
    prev = {}
    for r in runs:
        prev[r["pattern"]] = r["prevalence"]
        for det, mv in r["metrics"].items():
            for m, v in mv.items():
                acc[(r["pattern"], r["intensity"], det)][m].append(v)
    summary = defaultdict(lambda: defaultdict(dict))
    for (pattern, lvl, det), ms in acc.items():
        summary[pattern][lvl][det] = {
            m: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v),
                "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]} for m, v in ms.items()}
    out = {"summary": summary, "prevalence": prev, "draws": args.draws, "intensities": INTENSITIES,
           "patterns": PATTERNS, "graph": {"venues": int(W.shape[0]), "links": float(W.sum())},
           "real_graph": {"median_out_links": float(np.median(out_total)),
                          "size_vs_reciprocity_spearman": float(pd.Series(venue_structure(W, out_total)["reciprocity"])
                                                                .corr(pd.Series(np.log1p(out_total)), method="spearman"))}}
    (RES / "anomaly.json").write_text(json.dumps(out, indent=1))
    logger.info("anomaly: %d runs -> %s", len(runs), RES / "anomaly.json")


if __name__ == "__main__":
    main()
