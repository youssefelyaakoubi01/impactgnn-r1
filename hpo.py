"""Hyperparameter selection of the revision.

``python experiments_rev1.py --grid hpo``   runs the search (same budget for every model family)
``python hpo.py``                           selects one configuration per model family, task and protocol family

Selection uses validation data only: the mean validation metric (Spearman correlation for regression, ROC-AUC for
classification) over three data partitions. The first trial of every family is the configuration of the submitted
article, so the selected configuration is never worse than it on the validation data.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from experiments_rev1 import read_runs
from grids_rev1 import HPO_FILE

METRIC = {"reg": "spearman", "cls": "roc_auc"}


def select() -> dict:
    trials = defaultdict(lambda: defaultdict(list))
    spec = {}
    for r in read_runs("hpo"):
        j = r["job"]
        _, fam, task, pf, n = j["arm"].split("|")
        key = f"{fam}|{task}|{pf}"
        trials[key][int(n)].append(r["val_metrics"][METRIC[task]])
        spec[(key, int(n))] = {"hp": j["hp"], "text_dim": j["text_dim"]}
    best = {}
    for key, tr in sorted(trials.items()):
        complete = {n: v for n, v in tr.items() if len(v) == 3}
        means = {n: float(np.mean(v)) for n, v in complete.items()}
        n_best = max(means, key=means.get)
        best[key] = dict(spec[(key, n_best)], trial=n_best, val=means[n_best], val_default=means.get(0),
                         n_trials=len(complete),
                         val_all=[round(means[n], 4) for n in sorted(means)])
    HPO_FILE.parent.mkdir(parents=True, exist_ok=True)
    HPO_FILE.write_text(json.dumps(best, indent=1))
    return best


if __name__ == "__main__":
    for key, b in select().items():
        print(f"{key:18s} trial {b['trial']:2d}/{b['n_trials']}  val {b['val']:.4f} (submitted configuration "
              f"{b['val_default']:.4f})  text_dim {b['text_dim']}  {b['hp']}")
