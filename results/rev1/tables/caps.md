**Table @caps.** Sensitivity to the node cap in the retrospective K-fold (*k* = 3, Spearman correlation, mean ± std over 12 test folds). Δ is the paired difference with the MLP, which does not use the graph. Exploratory analysis.

| Graph | MLP + in-graph aggregates | GraphSAGE | Δ GraphSAGE − MLP | GAT | Δ GAT − MLP |
|---|---|---|---|---|---|
| Cap 30,000 | 0.684 ± 0.013 | 0.679 ± 0.014 | +0.009 [+0.001, +0.017]ᵇ | 0.681 ± 0.012 | +0.010 [+0.002, +0.018]ᵇ |
| Cap 45,000 | 0.686 ± 0.013 | 0.681 ± 0.014 | +0.010 [+0.003, +0.017]ᵇ | 0.682 ± 0.011 | +0.011 [+0.003, +0.019]ᵇ |
| Cap 60,000 (ties by hash) | 0.689 ± 0.013 | 0.683 ± 0.014 | +0.012 [+0.006, +0.018]ᵇ | 0.684 ± 0.011 | +0.013 [+0.005, +0.021]ᵇ |
| Initial graph | 0.687 ± 0.011 | 0.682 ± 0.013 | +0.012 [+0.006, +0.017]ᵇ | 0.685 ± 0.012 | +0.015 [+0.008, +0.022]ᵇ |
| Uncapped | 0.689 ± 0.011 | 0.683 ± 0.012 | +0.012 [+0.005, +0.020]ᵇ | 0.684 ± 0.012 | +0.014 [+0.007, +0.021]ᵇ |

ᵇ Different: the 95 % interval excludes zero. The other differences are inconclusive. Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; Δ: paired difference (first model minus second model, or model minus reference).
