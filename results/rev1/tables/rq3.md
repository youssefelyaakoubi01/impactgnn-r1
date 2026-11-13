**Table @rq3.** Detection of top-10 % papers: mean ± std over runs (primary protocol) or over test folds (retrospective K-fold). AP: average precision; P@10 %: precision among the 10 % highest scores. Δ is the paired difference in ROC-AUC with the MLP and its 95 % interval. Tuned hyperparameters, SciBERT text. Primary analysis for the rows of the primary protocol; the K-fold rows are retrospective.

| Protocol | Model | ROC-AUC | AP | P@10 % | Δ ROC-AUC vs MLP |
|---|---|---|---|---|---|
| Primary (15 runs) | MLP | 0.796 ± 0.013 | 0.353 ± 0.034 | 0.393 ± 0.018 | reference |
|  | MLP + in-graph aggregates | 0.817 ± 0.010 | 0.386 ± 0.028 | 0.418 ± 0.021 | +0.021 [+0.012, +0.029]ᵇ |
|  | MLP + all-reference aggregates | 0.811 ± 0.013 | 0.364 ± 0.030 | 0.405 ± 0.013 | +0.015 [+0.009, +0.021]ᵇ |
|  | GraphSAGE | 0.801 ± 0.010 | 0.340 ± 0.019 | 0.374 ± 0.009 | +0.005 [−0.002, +0.013] |
|  | GAT | 0.807 ± 0.015 | 0.345 ± 0.035 | 0.366 ± 0.019 | +0.010 [+0.001, +0.020]ᵇ |
|  | HAN (temporal rule) | 0.808 ± 0.013 | 0.357 ± 0.032 | 0.378 ± 0.021 | +0.012 [+0.001, +0.022]ᵇ |
|  | HGT (T0 rule) | 0.813 ± 0.008 | 0.364 ± 0.018 | 0.404 ± 0.016 | +0.017 [+0.006, +0.027]ᵇ |
|  | HGT (rolling rule) | 0.813 ± 0.007 | 0.369 ± 0.019 | 0.403 ± 0.023 | +0.017 [+0.008, +0.027]ᵇ |
|  | HGT (naive) | 0.813 ± 0.008 | 0.364 ± 0.019 | 0.406 ± 0.020 | +0.017 [+0.006, +0.027]ᵇ |
| K-fold, *k* = 3 (12 folds) | MLP | 0.843 ± 0.009 | 0.421 ± 0.030 | 0.430 ± 0.029 | reference |
|  | GraphSAGE | 0.852 ± 0.009 | 0.426 ± 0.028 | 0.438 ± 0.033 | +0.010 [+0.003, +0.017]ᵇ |
|  | GAT | 0.846 ± 0.007 | 0.417 ± 0.026 | 0.430 ± 0.022 | +0.003 [−0.004, +0.011] |
|  | HAN (temporal rule) | 0.853 ± 0.009 | 0.435 ± 0.031 | 0.440 ± 0.031 | +0.011 [+0.002, +0.019]ᵇ |
|  | HAN (T0 rule) | 0.851 ± 0.009 | 0.427 ± 0.036 | 0.437 ± 0.033 | +0.009 [+0.0001, +0.017]ᵇ |
|  | HGT (T0 rule) | 0.859 ± 0.009 | 0.453 ± 0.034 | 0.458 ± 0.029 | +0.017 [+0.012, +0.022]ᵇ |
|  | HGT (rolling rule) | 0.862 ± 0.010 | 0.463 ± 0.029 | 0.461 ± 0.028 | +0.019 [+0.014, +0.025]ᵇ |
|  | HGT (naive) | 0.860 ± 0.008 | 0.453 ± 0.033 | 0.456 ± 0.033 | +0.018 [+0.013, +0.022]ᵇ |
| K-fold, *k* = 5 (12 folds) | MLP | 0.835 ± 0.013 | 0.403 ± 0.032 | 0.404 ± 0.035 | reference |
|  | GraphSAGE | 0.850 ± 0.015 | 0.427 ± 0.049 | 0.448 ± 0.043 | +0.015 [+0.003, +0.026]ᵇ |
|  | GAT | 0.836 ± 0.012 | 0.386 ± 0.049 | 0.413 ± 0.048 | 0.000 [−0.019, +0.020] |
|  | HAN (temporal rule) | 0.850 ± 0.013 | 0.432 ± 0.045 | 0.438 ± 0.046 | +0.015 [+0.004, +0.025]ᵇ |
|  | HAN (T0 rule) | 0.848 ± 0.013 | 0.428 ± 0.046 | 0.442 ± 0.052 | +0.013 [+0.0001, +0.025]ᵇ |
|  | HGT (T0 rule) | 0.858 ± 0.016 | 0.451 ± 0.036 | 0.455 ± 0.046 | +0.023 [+0.007, +0.039]ᵇ |
|  | HGT (rolling rule) | 0.855 ± 0.016 | 0.449 ± 0.038 | 0.451 ± 0.038 | +0.020 [+0.005, +0.036]ᵇ |
|  | HGT (naive) | 0.857 ± 0.014 | 0.451 ± 0.038 | 0.453 ± 0.041 | +0.022 [+0.009, +0.036]ᵇ |

ᵇ Different: the 95 % interval excludes zero. The other differences are inconclusive. "Aggregates": seven summary statistics of the references (Section 4.3), computed over the in-graph references or over all references. The naive HGT lets every paper send messages to its authors, venue and keywords. Registered contrasts (Holm-adjusted p-values in Section 5.10): HGT (rolling rule) and GraphSAGE against the MLP in the primary protocol; the other differences are exploratory. Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; HAN: heterogeneous graph attention network; HGT: heterogeneous graph transformer; ROC-AUC: area under the ROC curve; AP: average precision; P@10 %: precision among the 10 % highest-scored papers; K-fold: 6-fold cross-validation repeated twice (retrospective); *k*: horizon in years; T0 rule: author, venue and keyword nodes aggregate only the papers published before 2017; rolling rule: year-specific author, venue and keyword nodes that aggregate only earlier papers; temporal rule: a metapath neighbour is kept only if it is earlier than the target paper; naive: no temporal rule on the author, venue and keyword nodes; Δ: paired difference (first model minus second model, or model minus reference).
