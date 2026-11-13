# impactgnn: Axis 1, citation graphs and GNNs

Temporally constrained evaluation of graph neural networks for scientific impact assessment on OpenAlex data:
(1) impact prediction with GAT and attention analysis, (2) inductive cold start with GraphSAGE,
(3) heterogeneous graph with HAN/HGT, (4) citation cartels and self-citation, (5) text vs structure.

## Setup

Python 3.12 virtual environment (`.venv`) with CUDA torch 2.14.1 (cu130), torch_geometric 2.8.1 and pyg-lib 0.9.0;
pinned versions in `requirements.txt`. The OpenAlex API keys are read from `.env` (`OPENALEX_API_KEY`,
`OPENALEX_API_KEY_2`, ...; git-ignored); the collector switches to the next key when one is exhausted. An optional
directory of earlier PDF matches can be given with `IMPACTGNN_PRIOR_DIR`; it is not needed to repeat the results.

## Pipeline

```bash
.venv/bin/python collect.py --profile full        # OpenAlex matching and downloads (cached in data/cache)
.venv/bin/python build_graph.py --profile full    # embeddings (GPU), time-safe features, graphs
.venv/bin/python experiments.py --profile full --exp all --jobs 3
.venv/bin/python summarize.py --profile full      # results/summary.md
.venv/bin/python experiments.py --profile full --exp 1,3,5 --text-dim 64   # exploratory analysis (post hoc)
.venv/bin/python benchmarks.py                    # ogbn-arxiv implementation check
.venv/bin/streamlit run app.py                    # interface
```

`./start_full.sh` runs collect -> build -> experiments -> summary detached from the terminal (log in `logs/`).
`--profile small` runs the whole pipeline on a pilot (4 topics x 100 papers). Unit tests: `.venv/bin/python -m pytest tests`.

## Revision (rev1)

The revision of the article (branch `revision-r1`) keeps the results of the submitted pipeline (`results/*.json`;
the code as submitted is at the tag `v1-submitted`, and `models.py`, `collect.py` and `config.py` have been extended
since) and adds its own pipeline, driven by `./run_rev1.sh <step>`:

| Step | Scripts | Output |
|---|---|---|
| `data` | `collect_rev1.py`, `build_rev1.py`, `analysis_rev1.py` | works of every author (citations by citing year), augmented graph `data/processed/full/rev1/`, first-hop graph, composition statistics |
| `hpo` | `experiments_rev1.py --grid hpo`, `hpo.py` | `results/rev1/hpo/best.json` (24 configurations per model family, task and protocol) |
| `grids` | `experiments_rev1.py --grid ...` (`grids_rev1.py`) | `results/rev1/runs/<grid>.jsonl`, per-paper predictions in `results/rev1/preds/` |
| `anomaly` | `anomaly_rev1.py` | `results/rev1/anomaly.json`, `anomaly_external.json` |
| `freeze`, `holdout` | `holdout.py`, `prereg/` | registered models (tag `v2-frozen`), fresh 2022 sample, `results/rev1/holdout.json` |
| `paper` | `make_tables.py`, `paper/figures.py`, `paper/include_tables.py`, `paper/check_numbers.py`, `paper/md2tex.py` | `results/rev1/tables/`, `paper/paper.pdf` and `paper/supplement.pdf` (changes in red), their `_clean` copies, `paper/response_to_reviewers.pdf` |

What the revision changes in the method:

- **Author and venue statistics.** The yearly citation counts of an OpenAlex author or source are keyed by the
  publication year of the cited works, so "total minus the years from y on" is not the number of citations received
  before y. `build_rev1.py` recomputes it from the works of each author (`exact`), keeps the submitted statistics as
  `hind`, and offers a `strict` variant without citation counts.
- **Protocols** (`protocols.py`): `mature` (labels closed before the prediction date; primary), `retro` (temporal
  split of the submitted version), `kfold` / `topicfold` / `group` (retrospective), `legacy` (reproduces the
  submitted split).
- **Graphs** (`graph_variants.py`): the submitted graph, uncapped and capped graphs, and graphs whose reference
  nodes are selected from papers published before a prediction date; year-specific entity snapshots for HGT.
- **Statistics** (`stats.py`): paired differences on identical test papers, bootstrap intervals over papers and over
  topics, corrected resampled t-test for K-fold, equivalence margin, Holm adjustment.
- Split and initialisation seeds are separate; scalers, PCA and TF-IDF are fitted on the training graph; every run
  stores its predictions; all tables of the article are generated.

Tests: `.venv/bin/python -m pytest tests` (leakage rules of the submitted version and of the revision).
The contact address for the OpenAlex polite pool is read from the environment variable `OPENALEX_MAILTO` (optional).

## Paper

`paper/axis1.md` is the source text of the article and of its supplement (the part between `<!--supplement-->` and
`<!--/supplement-->`); `paper/md2tex.py` generates `paper/paper.tex` and `paper/supplement.tex`, with the changes of
the revision in red, and their `_clean` copies (IJASCA-like layout,
`references.bib` written by `paper/verify_refs.py` from Crossref/arXiv metadata). `paper/figures.py` draws the
figures from `results/`, and `paper/check_numbers.py` checks that every number of the paper, or of the response
letter when it is given `paper/response_to_reviewers.tex`, appears in the pipeline outputs. `paper/refs_md.py keys|numbers` switches the citations of `axis1.md` between `[@key]` (for
editing) and numbers, and regenerates its reference list from the verified metadata in the same order as the PDF.
`paper/scopus/` holds the Scopus export used for the related work and the screening decision for each record
(`screening.csv`).
