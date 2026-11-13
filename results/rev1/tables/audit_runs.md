**Table @audit_runs.** Sensitivity to metadata whose state at the prediction date is uncertain (primary protocol, *k* = 3, mean ± std over 15 runs). Δ is the paired difference with the same model on all features. Exploratory analysis.

| Configuration | Spearman | Δ Spearman | ROC-AUC | Δ ROC-AUC |
|---|---|---|---|---|
| MLP, all features | 0.639 ± 0.005 | reference | 0.796 ± 0.013 | reference |
| MLP without author features | 0.613 ± 0.007 | −0.027 [−0.036, −0.017]ᵇ | 0.786 ± 0.013 | −0.011 [−0.023, +0.002] |
| MLP without venue features | 0.610 ± 0.005 | −0.030 [−0.038, −0.021]ᵇ | 0.787 ± 0.011 | −0.009 [−0.017, +0.0002] |
| MLP, base features and text only | 0.574 ± 0.008 | −0.065 [−0.080, −0.050]ᵇ | 0.772 ± 0.011 | −0.024 [−0.042, −0.007]ᵇ |
| GraphSAGE, all features | 0.624 ± 0.009 | reference | 0.801 ± 0.010 | reference |
| GraphSAGE without author features | 0.607 ± 0.006 | −0.017 [−0.025, −0.008]ᵇ | 0.796 ± 0.009 | −0.005 [−0.018, +0.007] |
| HGT (rolling rule), all node types | – | – | 0.813 ± 0.007 | reference |
| HGT without keyword nodes | – | – | 0.820 ± 0.010 | +0.006 [+0.0003, +0.012]ᵇ |
| HGT without author nodes and features | – | – | 0.799 ± 0.005 | −0.014 [−0.026, −0.003]ᵇ |
| HGT with papers and venues only | – | – | 0.807 ± 0.005 | −0.006 [−0.020, +0.006] |

ᵇ Different: the 95 % interval excludes zero. The other differences are inconclusive. Abbreviations: MLP: multilayer perceptron; HGT: heterogeneous graph transformer; ROC-AUC: area under the ROC curve; rolling rule: year-specific author, venue and keyword nodes that aggregate only earlier papers; Δ: paired difference (first model minus second model, or model minus reference).
