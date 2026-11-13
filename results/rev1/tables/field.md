**Table @field.** Top-10 % class defined within publication year and field (*k* = 3; primary protocol: mean ± std over 15 runs; retrospective K-fold: over 12 test folds). Exploratory analysis.

| Model | Primary: ROC-AUC | Primary: AP | Primary: Δ ROC-AUC vs MLP | K-fold: ROC-AUC | K-fold: AP | K-fold: Δ ROC-AUC vs MLP |
|---|---|---|---|---|---|---|
| MLP | 0.738 ± 0.016 | 0.319 ± 0.028 | reference | 0.816 ± 0.008 | 0.392 ± 0.023 | reference |
| GraphSAGE | 0.734 ± 0.010 | 0.309 ± 0.019 | −0.004 [−0.012, +0.004] | 0.822 ± 0.008 | 0.397 ± 0.025 | +0.007 [+0.001, +0.012]ᵇ |
| GAT | 0.741 ± 0.015 | 0.309 ± 0.031 | +0.003 [−0.007, +0.012] | 0.816 ± 0.009 | 0.384 ± 0.019 | 0.000 [−0.008, +0.008]ᵃ |
| HAN (temporal rule) | 0.745 ± 0.011 | 0.327 ± 0.019 | +0.007 [−0.005, +0.017] | 0.827 ± 0.008 | 0.406 ± 0.027 | +0.011 [+0.006, +0.016]ᵇ |
| HGT (T0 rule) | – | – | – | 0.830 ± 0.009 | 0.422 ± 0.029 | +0.015 [+0.006, +0.023]ᵇ |
| HGT (rolling rule) | 0.750 ± 0.007 | 0.339 ± 0.012 | +0.012 [+0.002, +0.022]ᵇ | 0.833 ± 0.011 | 0.433 ± 0.026 | +0.018 [+0.010, +0.025]ᵇ |

ᵃ Equivalent: the 95 % interval lies within ±0.01. ᵇ Different: the 95 % interval excludes zero. Other differences are inconclusive. Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; HAN: heterogeneous graph attention network; HGT: heterogeneous graph transformer; ROC-AUC: area under the ROC curve; AP: average precision; K-fold: 6-fold cross-validation repeated twice (retrospective); T0 rule: author, venue and keyword nodes aggregate only the papers published before 2017; rolling rule: year-specific author, venue and keyword nodes that aggregate only earlier papers; temporal rule: a metapath neighbour is kept only if it is earlier than the target paper; Δ: paired difference (first model minus second model, or model minus reference).
