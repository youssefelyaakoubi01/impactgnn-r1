# Confirmatory holdout: what changed after the registration

`frozen.json` and `README.md` in this directory are unchanged since the tag `v2-frozen` (commit `d51be96`,
2026-11-13 03:23:46 UTC). This file records what happened between the registration and the evaluation. Times are
taken from the git history and from `logs/collect_rev1_full.log` and `logs/build_rev1_full.log`.

| Time (UTC, 2026-11-13) | Event |
|---|---|
| 03:23:46 | Registration committed and tagged `v2-frozen`: 165 saved models with their digests, the contrasts, the metrics and the margin |
| 03:23:55 | Download of the fresh sample starts (`collect_rev1.py --step fresh`) |
| 03:38:51 | Holdout graph built a first time (78,468 nodes, 904,201 edges) |
| 03:42:42 | Holdout graph built a second time, with the change described below |
| 03:43:28 | The change is committed (`24327d0`) |
| 03:55:24 | Result of the single evaluation committed (`08fba0c`) |

## One change to the code

`build_rev1.py --holdout` was modified after the registration and after the fresh sample had been downloaded, and
before the evaluation. In the first build, adding the fresh papers changed the features of nodes that were already
in the training graph: a reference node whose author also wrote a fresh paper gained an author record, and the
references of the fresh papers entered the aggregates of older papers. With the change, the nodes of the training
graph keep exactly the features they had when the models were trained, and only the fresh papers and the reference
nodes they bring in receive new features. The same commit adds a `--device` option to `holdout.py`, which does not
affect the predictions.

What did not change: the saved models, their digests, the feature scalers and text projections, the contrasts, the
metric of each task, the equivalence margin and the definition of the test set.

## What the records show and what they do not

- `holdout.py --evaluate` refuses to run when `results/rev1/runs/holdout.jsonl` exists, and that file holds one
  record per registered model (165). The evaluation reported in the article is this single run, made on the second
  build of the graph.
- `logs/holdout_full.log` is empty: the output of the evaluation was printed to the terminal and was not saved. The
  result is in `results/rev1/holdout.json`, and the per-paper predictions of the 165 models are in the archive.
- No evaluation on the first build is recorded. The logs cannot prove that none was started; the commit message of
  `24327d0` states that the change was made before the evaluation was run.

The article reports this change in Section 4.4 ("Confirmatory holdout").
