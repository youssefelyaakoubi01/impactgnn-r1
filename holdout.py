"""Confirmatory holdout of the revision: frozen models evaluated once on a fresh sample of 2022 papers.

Order of operations (the first two steps are committed and tagged before the third one is run):

1. ``python experiments_rev1.py --grid frozen``     train the arms on the primary protocol and save every model with
                                                    its scalers and projections (``results/rev1/checkpoints``)
2. ``python holdout.py --register``                 write ``prereg/frozen.json``: checkpoint digests, contrasts,
                                                    metric, equivalence margin
3. ``python collect_rev1.py --step fresh`` then ``python build_rev1.py --holdout``
4. ``python holdout.py --evaluate``                 predict the fresh papers with the saved models (nothing is
                                                    fitted) -> ``results/rev1/runs/holdout.jsonl``,
                                                    ``results/rev1/holdout.json``

The fresh papers are 50 articles of 2022 per topic of the original sample, drawn with another seed and disjoint
from every work already collected. Their citation labels do not exist locally before step 3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math

import numpy as np
import torch

import stats as S
from config import ROOT, get_config, resolve_device, setup_logging
from experiments import cls_metrics, reg_metrics
from experiments_rev1 import RES, load, predict_frozen, read_runs

PREREG = ROOT / "prereg"
CONTRASTS = [  # (task, arm a, arm b, role)
    ("reg", "sage", "mlp", "primary"), ("reg", "gat", "mlp", "primary"), ("reg", "mlp+refagg", "mlp", "primary"),
    ("cls", "hgt_roll", "mlp", "primary"), ("cls", "sage", "mlp", "primary"),
    ("cls", "hgt_roll|specter|d768", "mlp|specter|d768", "primary"),
    ("cls", "hgt_roll|specter|d64", "mlp|specter|d64", "primary"),
    ("cls", "mlp|specter|d64", "mlp|specter|d768", "secondary"),
    ("cls", "hgt_roll|specter|d64", "hgt_roll|specter|d768", "secondary"),
]


def digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register() -> None:
    runs = read_runs("frozen")
    PREREG.mkdir(exist_ok=True)
    reg = {
        "protocol": "mature labels, prediction date 2022, graph cut2022sym; models trained on cohorts 2017-2018",
        "test_set": "fresh sample: 48 topics x 50 articles of 2022, OpenAlex sample seed 20261113, disjoint from "
                    "every work already collected; label = citations 2022-2025; top-10 % class with the threshold of "
                    "the original 2022 cohort",
        "metric": {"reg": "spearman", "cls": "roc_auc"}, "equivalence_margin": S.MARGIN,
        "statistic": "difference of the metric averaged over the paired runs (5 validation partitions x 3 seeds); "
                     "95 % interval from a paired bootstrap over the fresh papers (2000 resamples) and from a "
                     "bootstrap over topics; Holm adjustment over the primary contrasts",
        "contrasts": [dict(task=t, a=a, b=b, role=r) for t, a, b, r in CONTRASTS],
        "rule": "the models, scalers and projections below are applied unchanged; the evaluation is run once and "
                "reported whatever the outcome",
        "checkpoints": [{"id": r["id"], "arm": r["job"]["arm"], "task": r["job"]["task"],
                         "split_seed": r["job"]["split_seed"], "init_seed": r["job"]["init_seed"],
                         "sha256": digest(RES / "checkpoints" / f"{r['id']}.pt"),
                         "test_metric_original_2022_cohort": r["metrics"][S.primary_metric(r["job"]["task"])]}
                        for r in runs],
    }
    (PREREG / "frozen.json").write_text(json.dumps(reg, indent=1))
    print(f"registered {len(runs)} checkpoints -> {PREREG / 'frozen.json'}")


def evaluate(device_name: str | None = None) -> None:
    cfg = get_config("full", device=device_name)
    setup_logging("holdout", cfg)
    device = resolve_device(cfg)
    out_runs = RES / "runs" / "holdout.jsonl"
    assert not out_runs.exists(), "the holdout has already been evaluated; it is run once"
    reg = json.loads((PREREG / "frozen.json").read_text())
    G = load(cfg, device, sub="rev1_holdout")
    g = G["g"]
    fresh = torch.as_tensor([o == "fresh" for o in g["origin"]], device=device) & g["m3_fresh"].bool()
    idx = fresh.nonzero().squeeze(1)
    ids = np.asarray(g["ids"])[idx.cpu().numpy()]
    jobs = {r["id"]: r["job"] for r in read_runs("frozen")}
    with open(out_runs, "w") as fh:
        for c in reg["checkpoints"]:
            path = RES / "checkpoints" / f"{c['id']}.pt"
            assert digest(path) == c["sha256"], f"checkpoint {c['id']} differs from the registered one"
            ck = torch.load(path, weights_only=False)
            p = predict_frozen(ck, G, device)[idx].cpu().numpy().astype(np.float32)
            task = c["task"]
            y = (g["y3"] if task == "reg" else g["top3"])[idx].float().cpu().numpy()
            hid = "h" + c["id"]
            np.savez_compressed(RES / "preds" / f"{hid}.npz", te_ids=ids, te_p=p, te_y=y, va_ids=ids[:0], va_p=p[:0],
                                va_y=y[:0])
            m = reg_metrics(y, p) if task == "reg" else cls_metrics(y, p, cfg.top_share)
            fh.write(json.dumps({"id": hid, "job": dict(jobs[c["id"]], protocol="holdout"), "metrics": m,
                                 "val_metrics": {}, "epochs": 0, "time_s": 0, "n_train": 0, "n_test": int(len(y))},
                                default=float) + "\n")
    df = S.runs("holdout")
    res = {"n_fresh": int(len(ids)), "n_topics": int(S.node_table_holdout(g).loc[ids, "topic"].nunique()),
           "positives": int(g["top3"][idx].sum()), "contrasts": []}
    for task, a, b, role in CONTRASTS:
        d = df[df.task == task]
        metric = S.primary_metric(task)
        c = S.contrast_fixed(d, a, b, metric)
        c["topic_ci"] = S.contrast_fixed(d, a, b, metric, cluster=True,
                                         topic_of=S.node_table_holdout(g).topic)["ci"]
        c.update(task=task, role=role)
        res["contrasts"].append(c)
    prim = [c for c in res["contrasts"] if c["role"] == "primary"]
    for c, padj in zip(prim, S.holm([c["p"] for c in prim])):
        c["p_holm"] = padj
    (RES / "holdout.json").write_text(json.dumps(res, indent=1))
    for c in res["contrasts"]:
        print(f"{c['task']} {c['a']:26s} - {c['b']:20s} {c['metric']}: {c['mean_a']:.3f} vs {c['mean_b']:.3f}  "
              f"diff {c['diff']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]  p {c['p']:.3f}")


def describe() -> None:
    """Facts about the fresh sample and the holdout graph that the article reports (no model is evaluated):
    collection effort, reference coverage of the fresh papers, and the check that the saved models reproduce
    their predictions for the original test papers on the enlarged graph."""
    import pandas as pd

    from graph_variants import edge_mask

    cfg = get_config("full", device="cpu")
    device = torch.device("cpu")
    G = load(cfg, device, sub="rev1_holdout")
    g = G["g"]
    raw = cfg.raw_dir
    origin = np.asarray(g["origin"])
    fresh = origin == "fresh"
    years = g["year"].numpy()
    em = edge_mask(g, "cut2022sym")
    deg = np.bincount(g["edge_index"][1, em].numpy(), minlength=len(origin))
    old_te = g["m3"].bool().numpy() & (years == 2022) & (origin == "sample")
    report = json.loads((raw / "collect_report.json").read_text())["steps"]
    pages = sum(json.loads(p.read_text())["pages"] for p in (raw / "author_works_fresh").glob("part-*.json"))
    out = {"fresh_papers": int(fresh.sum()), "requests_sample_refs_authors_venues": report["rev1_fresh"]["requests_sent"],
           "requests_author_works": int(pages),
           "new_references_with_metadata": int(len(pd.read_parquet(raw / "fresh_refs.parquet", columns=["id"]))),
           "new_authors": int(len(pd.read_parquet(raw / "fresh_authors.parquet", columns=["id"]))),
           "reference_nodes_added": int((np.arange(len(origin)) >= len(origin) - (origin == "ref").sum()).sum() * 0
                                        + ((origin == "ref") & (np.arange(len(origin)) > np.flatnonzero(fresh).max())).sum()),
           "nodes": int(len(origin)), "edges": int(g["edge_index"].shape[1]),
           "in_graph_refs_fresh": {"median": float(np.median(deg[fresh])), "mean": float(deg[fresh].mean()),
                                   "zero_share": float((deg[fresh] == 0).mean())},
           "in_graph_refs_original_test": {"median": float(np.median(deg[old_te])), "mean": float(deg[old_te].mean()),
                                           "zero_share": float((deg[old_te] == 0).mean())},
           "reproduction_original_test": {}}
    idx = {i: n for n, i in enumerate(g["ids"])}
    seen = set()
    for r in read_runs("frozen"):
        arm = (r["job"]["arm"], r["job"]["task"])
        if arm in seen:
            continue
        seen.add(arm)
        ck = torch.load(RES / "checkpoints" / f"{r['id']}.pt", weights_only=False)
        p = predict_frozen(ck, G, device).numpy()
        z = np.load(RES / "preds" / f"{r['id']}.npz")
        te = [idx[i] for i in z["te_ids"]]
        out["reproduction_original_test"][f"{arm[0]}|{arm[1]}"] = {
            "max_abs_diff": float(np.abs(p[te] - z["te_p"]).max()), "correlation": float(np.corrcoef(p[te], z["te_p"])[0, 1])}
    rep = out["reproduction_original_test"]
    out["frozen_models"] = len(read_runs("frozen"))
    out["hgt_min_correlation_floor4"] = math.floor(min(v["correlation"] for k, v in rep.items() if k.startswith("hgt"))
                                                   * 1e4) / 1e4
    out["homogeneous_max_abs_diff"] = max(v["max_abs_diff"] for k, v in rep.items() if not k.startswith("hgt"))
    (RES / "stats" / "holdout_build.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--device", default=None, help="cuda (default) or cpu; inference only, so cpu is practical")
    ap.add_argument("--describe", action="store_true", help="facts about the fresh sample and the holdout graph")
    a = ap.parse_args()
    if a.describe:
        describe()
    if a.register:
        register()
    if a.evaluate:
        evaluate(a.device)
