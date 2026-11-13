**Table @edgedrop.** Share of the reference links into sampled papers that is kept (retrospective K-fold, *k* = 3, Spearman correlation, mean ± std over 12 test folds). Exploratory analysis.

| Reference links kept | GraphSAGE | Δ GraphSAGE − MLP | GAT | Δ GAT − MLP |
|---|---|---|---|---|
| 100 % | 0.682 ± 0.013 | +0.012 [+0.006, +0.017]ᵇ | 0.685 ± 0.012 | +0.015 [+0.008, +0.022]ᵇ |
| 75 % | 0.680 ± 0.013 | +0.009 [+0.003, +0.016]ᵇ | 0.682 ± 0.012 | +0.011 [+0.004, +0.019]ᵇ |
| 50 % | 0.678 ± 0.015 | +0.007 [+0.001, +0.013]ᵇ | 0.680 ± 0.011 | +0.009 [+0.002, +0.016]ᵇ |
| 25 % | 0.671 ± 0.014 | 0.000 [−0.009, +0.010]ᵃ | 0.676 ± 0.010 | +0.006 [−0.002, +0.013] |
| 0 % | 0.637 ± 0.012 | −0.034 [−0.045, −0.022]ᵇ | 0.673 ± 0.012 | +0.002 [−0.002, +0.006]ᵃ |

ᵃ Equivalent: the 95 % interval lies within ±0.01. ᵇ Different: the 95 % interval excludes zero. Other differences are inconclusive. Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; Δ: paired difference (first model minus second model, or model minus reference).
