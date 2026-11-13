**Table @graphs.** Reference selection and the primary protocol (*k* = 3, mean ± std over 15 runs). Δ is the paired difference with the MLP, which does not use the graph. Exploratory analysis.

| Graph | Model | Metric | Test value | Δ vs MLP |
|---|---|---|---|---|
| None (the MLP does not use the graph) | MLP | Spearman | 0.639 ± 0.005 | reference |
|  | MLP | ROC-AUC | 0.796 ± 0.013 | reference |
| Initial (all years, cap by identifier) | MLP + in-graph aggregates | Spearman | 0.640 ± 0.004 | +0.001 [−0.006, +0.008]ᵃ |
|  | GraphSAGE | Spearman | 0.619 ± 0.005 | −0.020 [−0.029, −0.011]ᵇ |
|  | GAT | Spearman | 0.619 ± 0.005 | −0.020 [−0.030, −0.010]ᵇ |
|  | HGT (rolling rule) | ROC-AUC | 0.814 ± 0.004 | +0.018 [+0.009, +0.028]ᵇ |
| Restricted to 2022, strict | MLP + in-graph aggregates | Spearman | 0.625 ± 0.005 | −0.014 [−0.023, −0.006]ᵇ |
|  | GraphSAGE | Spearman | 0.622 ± 0.006 | −0.018 [−0.028, −0.007]ᵇ |
|  | GAT | Spearman | 0.615 ± 0.004 | −0.024 [−0.035, −0.014]ᵇ |
|  | HGT (rolling rule) | ROC-AUC | 0.805 ± 0.010 | +0.008 [−0.001, +0.019] |
| Restricted to 2022, leave-self-out | MLP + in-graph aggregates | Spearman | 0.635 ± 0.003 | −0.004 [−0.011, +0.002] |
|  | GraphSAGE | Spearman | 0.626 ± 0.006 | −0.013 [−0.022, −0.003]ᵇ |
|  | GAT | Spearman | 0.621 ± 0.006 | −0.018 [−0.028, −0.008]ᵇ |
|  | HGT (rolling rule) | ROC-AUC | 0.804 ± 0.010 | +0.007 [−0.003, +0.018] |
| Restricted to 2022 (primary) | MLP + in-graph aggregates | Spearman | 0.642 ± 0.006 | +0.003 [−0.004, +0.010] |
|  | GraphSAGE | Spearman | 0.624 ± 0.009 | −0.016 [−0.025, −0.006]ᵇ |
|  | GAT | Spearman | 0.615 ± 0.009 | −0.024 [−0.034, −0.013]ᵇ |
|  | HGT (rolling rule) | ROC-AUC | 0.813 ± 0.007 | +0.017 [+0.008, +0.027]ᵇ |

ᵃ Equivalent: the 95 % interval lies within ±0.01. ᵇ Different: the 95 % interval excludes zero. Other differences are inconclusive. Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; HGT: heterogeneous graph transformer; ROC-AUC: area under the ROC curve; rolling rule: year-specific author, venue and keyword nodes that aggregate only earlier papers; Δ: paired difference (first model minus second model, or model minus reference).
