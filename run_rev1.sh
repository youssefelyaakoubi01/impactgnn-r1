#!/usr/bin/env bash
# Experiments of the revision (rev1), in order. Every step resumes: a run that is already in
# results/rev1/runs/<grid>.jsonl is not repeated, so the script can be interrupted and started again.
#
#   ./run_rev1.sh data       downloads of the revision and graphs          (network, then GPU)
#   ./run_rev1.sh hpo        hyperparameter search and selection           (GPU, ~3 h)
#   ./run_rev1.sh grids      all experiment grids                          (GPU, ~10 h)
#   ./run_rev1.sh anomaly    synthetic anomalies and the external check    (GPU, ~1 h)
#   ./run_rev1.sh freeze     train and register the models of the confirmatory holdout
#   ./run_rev1.sh holdout    fresh sample, holdout graph, single evaluation (network, then GPU)
#   ./run_rev1.sh paper      tables, figures, checks, the article, the supplement and the letter
#
# The holdout steps are run once: "freeze" is committed and tagged before "holdout" downloads the fresh sample.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
GRIDS=mature_main,mature_text,mature_coldstart,mature_graphs,mature_biblio,mature_audit,mature_origin2021,mature_pca,mature_field,mature_default,retro,deploy,kfold_main,kfold_default,kfold_text,kfold_pca,kfold_biblio,kfold_caps,kfold_attention,kfold_field,topicfold,kfold_firsthop,mature_firsthop

case "${1:-}" in
  data)
    ./run_collect_rev1.sh cutrefs,issn,authorworks
    $PY build_rev1.py --profile full
    $PY build_rev1.py --profile full --firsthop
    $PY analysis_rev1.py ;;
  hpo)
    $PY experiments_rev1.py --grid hpo
    $PY hpo.py ;;
  grids)
    # the heterogeneous models need most of a 20 GB GPU: they run in their own pass
    $PY experiments_rev1.py --grid "$GRIDS" --models mlp,sage,gat
    $PY experiments_rev1.py --grid "$GRIDS" --models han,hgt ;;
  anomaly)
    $PY anomaly_rev1.py --draws 30
    $PY anomaly_rev1.py --external ;;
  freeze)
    $PY experiments_rev1.py --grid frozen
    $PY holdout.py --register ;;
  holdout)
    $PY collect_rev1.py --profile full --step fresh
    $PY build_rev1.py --profile full --holdout
    $PY holdout.py --evaluate ;;
  paper)
    $PY make_tables.py 2>&1 | tee logs/make_tables_full.out
    $PY paper/figures.py
    $PY paper/include_tables.py
    $PY paper/refs_md.py numbers
    $PY paper/check_numbers.py 2>&1 | tee logs/check_numbers.out
    $PY paper/md2tex.py
    $PY paper/check_marks.py 2>&1 | tee logs/check_marks.out      # every change of the article is in red
    # the article and the supplement read each other's labels: article, supplement, then both again
    (cd paper && for pass in 1 2; do for doc in paper supplement paper_clean supplement_clean; do
       latexmk -g -pdf -interaction=nonstopmode "$doc.tex" > /dev/null || { echo "LaTeX failed: $doc"; exit 1; }
     done; done
     latexmk -g -pdf -interaction=nonstopmode response_to_reviewers.tex > /dev/null)
    $PY paper/check_numbers.py paper/response_to_reviewers.tex 2>&1 | tee logs/check_numbers_letter.out
    $PY paper/refs_md.py keys ;;
  *)
    sed -n '2,14p' "$0"; exit 1 ;;
esac
