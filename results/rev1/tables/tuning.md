**Table @tuning.** Initial and tuned hyperparameters at *k* = 3 (primary protocol: mean ± std over 15 runs; retrospective K-fold: over 12 test folds). Δ is the paired difference with the MLP under the same setting. Exploratory analysis.

| Protocol, metric | Model | Initial configuration | Δ vs MLP (initial) | Tuned configuration | Δ vs MLP (tuned) |
|---|---|---|---|---|---|
| Primary, Spearman | MLP | 0.603 ± 0.010 | reference | 0.639 ± 0.005 | reference |
|  | MLP + all-reference aggregates | 0.619 ± 0.013 | +0.016 [+0.010, +0.021]ᵇ | 0.646 ± 0.004 | +0.007 [+0.001, +0.014]ᵇ |
|  | GraphSAGE | 0.600 ± 0.011 | −0.003 [−0.011, +0.005] | 0.624 ± 0.009 | −0.016 [−0.025, −0.006]ᵇ |
|  | GAT | 0.594 ± 0.012 | −0.009 [−0.021, +0.004] | 0.615 ± 0.009 | −0.024 [−0.034, −0.013]ᵇ |
| Primary, ROC-AUC | MLP | 0.771 ± 0.013 | reference | 0.796 ± 0.013 | reference |
|  | GraphSAGE | 0.764 ± 0.024 | −0.006 [−0.012, −0.0005]ᵇ | 0.801 ± 0.010 | +0.005 [−0.002, +0.013] |
|  | GAT | 0.773 ± 0.026 | +0.003 [−0.009, +0.015] | 0.807 ± 0.015 | +0.010 [+0.001, +0.020]ᵇ |
|  | HAN (temporal rule) | 0.789 ± 0.020 | +0.018 [+0.011, +0.026]ᵇ | 0.808 ± 0.013 | +0.012 [+0.001, +0.022]ᵇ |
|  | HGT (T0 rule) | 0.789 ± 0.021 | +0.019 [+0.010, +0.027]ᵇ | 0.813 ± 0.008 | +0.017 [+0.006, +0.027]ᵇ |
|  | HGT (rolling rule) | 0.787 ± 0.024 | +0.016 [+0.008, +0.024]ᵇ | 0.813 ± 0.007 | +0.017 [+0.008, +0.027]ᵇ |
| K-fold, Spearman | MLP | 0.658 ± 0.012 | reference | 0.671 ± 0.014 | reference |
|  | MLP + all-reference aggregates | 0.677 ± 0.012 | +0.019 [+0.014, +0.024]ᵇ | 0.685 ± 0.012 | +0.014 [+0.009, +0.019]ᵇ |
|  | GraphSAGE | 0.665 ± 0.011 | +0.007 [+0.002, +0.011]ᵇ | 0.682 ± 0.013 | +0.012 [+0.006, +0.017]ᵇ |
|  | GAT | 0.662 ± 0.011 | +0.005 [−0.0004, +0.010]ᵃ | 0.685 ± 0.012 | +0.015 [+0.008, +0.022]ᵇ |
| K-fold, ROC-AUC | MLP | 0.817 ± 0.011 | reference | 0.843 ± 0.009 | reference |
|  | GraphSAGE | 0.826 ± 0.011 | +0.010 [+0.002, +0.018]ᵇ | 0.852 ± 0.009 | +0.010 [+0.003, +0.017]ᵇ |
|  | GAT | 0.829 ± 0.011 | +0.013 [+0.006, +0.019]ᵇ | 0.846 ± 0.007 | +0.003 [−0.004, +0.011] |
|  | HAN (temporal rule) | 0.832 ± 0.014 | +0.015 [+0.005, +0.026]ᵇ | 0.853 ± 0.009 | +0.011 [+0.002, +0.019]ᵇ |
|  | HGT (T0 rule) | 0.831 ± 0.011 | +0.015 [+0.009, +0.021]ᵇ | 0.859 ± 0.009 | +0.017 [+0.012, +0.022]ᵇ |
|  | HGT (rolling rule) | 0.835 ± 0.011 | +0.018 [+0.012, +0.025]ᵇ | 0.862 ± 0.010 | +0.019 [+0.014, +0.025]ᵇ |

ᵃ Equivalent: the 95 % interval lies within ±0.01. ᵇ Different: the 95 % interval excludes zero. Other differences are inconclusive. Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; HAN: heterogeneous graph attention network; HGT: heterogeneous graph transformer; ROC-AUC: area under the ROC curve; K-fold: 6-fold cross-validation repeated twice (retrospective); T0 rule: author, venue and keyword nodes aggregate only the papers published before 2017; rolling rule: year-specific author, venue and keyword nodes that aggregate only earlier papers; temporal rule: a metapath neighbour is kept only if it is earlier than the target paper; Δ: paired difference (first model minus second model, or model minus reference).
