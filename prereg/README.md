# Confirmatory holdout: registration

This directory fixes, before any fresh data are collected, what the confirmatory holdout of the revision tests.

- `frozen.json` is written by `python holdout.py --register` after the arms of the grid `frozen` have been trained.
  It lists the SHA-256 digest of every saved model (weights, feature scalers, text projections), the contrasts, the
  metric of each task, the equivalence margin and the test-set definition.
- The commit that adds `frozen.json` is tagged `v2-frozen`. The fresh sample is downloaded only afterwards
  (`python collect_rev1.py --step fresh`), and `python holdout.py --evaluate` refuses to run a second time.

What is tested
- Models: trained on the primary protocol (labels of the 2017-2018 cohorts, closed before the prediction date
  2022; graph restricted to 2022). Nothing is fitted on the fresh papers.
- Test set: 50 articles of 2022 per topic of the original sample (48 topics), OpenAlex `sample` with seed
  20261113, same filters as the original sample, disjoint from every work already collected. Labels: citations
  2022-2025; top-10 % class with the threshold of the original 2022 cohort.
- Primary contrasts (paired, same fresh papers):
  1. GraphSAGE - MLP, Spearman (regression)
  2. GAT - MLP, Spearman (regression)
  3. MLP with all-reference aggregates - MLP, Spearman (regression)
  4. HGT (rolling rule) - MLP, ROC-AUC (classification)
  5. GraphSAGE - MLP, ROC-AUC (classification)
  6. HGT - MLP with 768-dimensional SPECTER text and the submitted hyperparameters, ROC-AUC
  7. HGT - MLP with 64 SPECTER components and the submitted hyperparameters, ROC-AUC
  Contrasts 6 and 7 test the exploratory finding of the submitted version: a gain of the heterogeneous model over
  the MLP with 768-dimensional text that is no longer detectable with 64 components.
- Secondary contrasts: 64 components against 768 dimensions for the MLP and for HGT.
- Statistic: difference of the metric averaged over the 15 paired runs of each arm; 95 % interval from a paired
  bootstrap over the fresh papers (2,000 resamples) and from a bootstrap over topics; equivalence margin 0.01;
  Holm adjustment over the seven primary contrasts.
- The evaluation is run once and reported whatever the outcome.
