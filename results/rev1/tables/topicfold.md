**Table @topicfold.** Generalisation across topics at *k* = 3 (mean ± std over 12 test folds, or over runs for the two splits between groups of topics). Folds within topics: papers of a topic are in training and test folds; folds over topics: all papers of a topic are in the same fold. Δ is the paired difference with the MLP on the same split. Exploratory analysis.

| Split | Model | Metric | Test value | Δ vs MLP |
|---|---|---|---|---|
| Folds within topics | MLP | Spearman | 0.671 ± 0.014 | reference |
|  | MLP + all-reference aggregates | Spearman | 0.685 ± 0.012 | +0.014 [+0.009, +0.019]ᵇ |
|  | GraphSAGE | Spearman | 0.682 ± 0.013 | +0.012 [+0.006, +0.017]ᵇ |
|  | GAT | Spearman | 0.685 ± 0.012 | +0.015 [+0.008, +0.022]ᵇ |
| Folds over topics | MLP | Spearman | 0.635 ± 0.023 | reference |
|  | MLP + all-reference aggregates | Spearman | 0.657 ± 0.025 | +0.022 [+0.011, +0.033]ᵇ |
|  | GraphSAGE | Spearman | 0.651 ± 0.024 | +0.016 [+0.001, +0.031]ᵇ |
|  | GAT | Spearman | 0.655 ± 0.023 | +0.020 [+0.006, +0.035]ᵇ |
| Folds within topics | MLP | ROC-AUC | 0.843 ± 0.009 | reference |
|  | GraphSAGE | ROC-AUC | 0.852 ± 0.009 | +0.010 [+0.003, +0.017]ᵇ |
|  | HGT (rolling rule) | ROC-AUC | 0.862 ± 0.010 | +0.019 [+0.014, +0.025]ᵇ |
| Folds over topics | MLP | ROC-AUC | 0.810 ± 0.026 | reference |
|  | GraphSAGE | ROC-AUC | 0.822 ± 0.023 | +0.012 [+0.004, +0.021]ᵇ |
|  | HGT (rolling rule) | ROC-AUC | 0.833 ± 0.022 | +0.023 [+0.011, +0.036]ᵇ |
| Trained on the open-corpus topics, tested on the laboratory topics | MLP | Spearman | 0.595 ± 0.010 | reference |
|  | GraphSAGE | Spearman | 0.604 ± 0.009 | +0.009 [+0.001, +0.017]ᵇ |
|  | GAT | Spearman | 0.611 ± 0.010 | +0.016 [+0.007, +0.025]ᵇ |
| Trained on the laboratory topics, tested on the open-corpus topics | MLP | Spearman | 0.610 ± 0.010 | reference |
|  | GraphSAGE | Spearman | 0.621 ± 0.012 | +0.011 [+0.004, +0.018]ᵇ |
|  | GAT | Spearman | 0.614 ± 0.015 | +0.003 [−0.006, +0.013] |

ᵇ Different: the 95 % interval excludes zero. The other differences are inconclusive. Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; HGT: heterogeneous graph transformer; ROC-AUC: area under the ROC curve; rolling rule: year-specific author, venue and keyword nodes that aggregate only earlier papers; Δ: paired difference (first model minus second model, or model minus reference).
