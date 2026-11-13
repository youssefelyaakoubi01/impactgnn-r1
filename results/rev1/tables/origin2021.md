**Table @origin2021.** Second prediction date (2021): training labels from the 2017 cohort, test cohorts 2021 and 2022 (*k* = 3, mean ± std over 15 runs). Exploratory analysis.

| Model | Metric | Test value | Δ vs MLP |
|---|---|---|---|
| MLP | Spearman | 0.629 ± 0.002 | reference |
| MLP + all-reference aggregates | Spearman | 0.639 ± 0.008 | +0.009 [+0.005, +0.014]ᵇ |
| GraphSAGE | Spearman | 0.609 ± 0.009 | −0.020 [−0.028, −0.013]ᵇ |
| GAT | Spearman | 0.610 ± 0.006 | −0.019 [−0.026, −0.012]ᵇ |
| MLP | ROC-AUC | 0.787 ± 0.011 | reference |
| GraphSAGE | ROC-AUC | 0.783 ± 0.016 | −0.005 [−0.010, +0.001]ᵃ |
| GAT | ROC-AUC | 0.790 ± 0.014 | +0.003 [−0.005, +0.010]ᵃ |
| HGT (rolling rule) | ROC-AUC | 0.778 ± 0.015 | −0.009 [−0.020, +0.001] |

ᵃ Equivalent: the 95 % interval lies within ±0.01. ᵇ Different: the 95 % interval excludes zero. Other differences are inconclusive. Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; HGT: heterogeneous graph transformer; ROC-AUC: area under the ROC curve; rolling rule: year-specific author, venue and keyword nodes that aggregate only earlier papers; Δ: paired difference (first model minus second model, or model minus reference).
