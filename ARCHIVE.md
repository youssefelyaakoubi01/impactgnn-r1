# Archive of the revision

Snapshot of the development repository at commit `065f38bf0a25954f41321319ef175a4ed37f3e20`, without its history; the tags and branches named in `README.md` belong to that repository.

| Path | Content |
|---|---|
| repository root | code, tests, table and figure scripts, result files (`results/`), registration of the confirmatory holdout (`prereg/`) |
| `results/rev1/runs/` | one line per run: configuration, seeds, hyperparameters and metrics |
| `results/rev1/preds/` | per-paper predictions of every run (test and validation), one file per run identifier |
| `results/rev1/checkpoints/` | the registered models of the confirmatory holdout, with their scalers and projections |
| `archive/identifiers/` | OpenAlex identifiers of every paper node, with its year, topic, role, labels and the graph variants it belongs to; identifiers of the fresh holdout sample |
| `archive/splits/` | training, validation and test identifiers of every partition used by a run |
| `archive/environment/` | pinned packages with the wheel indexes, Python / PyTorch / CUDA / GPU versions |
| `SHA256SUMS` | checksum of every file |

Not included: the downloaded OpenAlex records (CC0; the collection scripts download them again), the manuscript and the Scopus export used for the related work.

Withheld, in line with the ethical considerations of the article: the names of the venues that an anomaly detector ranked highest on the real graph, and the predictions for the laboratory papers of the case study (the listings of the initial pipeline and the prediction files of the grid `deploy`, so `make_tables.py lab` cannot be run from this tree). The four laboratory papers with a complete citation window are in `results/rev1/tables/lab.md`, as in the article.
