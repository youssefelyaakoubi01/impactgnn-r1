**Table @holdout.** Confirmatory holdout: frozen models evaluated once on 2,400 fresh papers of 2022 (239 in the top-10 % class). Paired differences with 95 % intervals from a bootstrap over papers and over topics; Holm-adjusted p-values for the registered primary contrasts. Confirmatory analysis.

| Metric | Contrast | Role | First model | Second model | Δ [95 % interval, papers] | 95 % interval, topics | Holm p |
|---|---|---|---|---|---|---|---|
| Spearman | GraphSAGE − MLP | primary | 0.597 | 0.609 | −0.012 [−0.022, −0.003]ᵇ | [−0.023, −0.002] | 0.016 |
| Spearman | GAT − MLP | primary | 0.593 | 0.609 | −0.016 [−0.027, −0.006]ᵇ | [−0.029, −0.003] | 0.010 |
| Spearman | MLP + all-reference aggregates − MLP | primary | 0.619 | 0.609 | +0.010 [+0.002, +0.018]ᵇ | [+0.001, +0.021] | 0.030 |
| ROC-AUC | HGT (rolling) − MLP | primary | 0.804 | 0.788 | +0.016 [+0.005, +0.027]ᵇ | [+0.004, +0.027] | 0.016 |
| ROC-AUC | GraphSAGE − MLP | primary | 0.789 | 0.788 | +0.001 [−0.008, +0.010]ᵃ | [−0.011, +0.012] | 0.837 |
| ROC-AUC | HGT (rolling) − MLP, 768 dimensions (SPECTER, initial hyperparameters) | primary | 0.772 | 0.734 | +0.039 [+0.028, +0.051]ᵇ | [+0.025, +0.054] | 0.004 |
| ROC-AUC | HGT (rolling) − MLP, 64 components (SPECTER, initial hyperparameters) | primary | 0.798 | 0.775 | +0.023 [+0.011, +0.035]ᵇ | [+0.011, +0.035] | 0.004 |
| ROC-AUC | MLP: 64 components − 768 dimensions (SPECTER, initial hyperparameters) | secondary | 0.775 | 0.734 | +0.041 [+0.019, +0.062]ᵇ | [+0.022, +0.060] | – |
| ROC-AUC | HGT (rolling): 64 components − 768 dimensions (SPECTER, initial hyperparameters) | secondary | 0.798 | 0.772 | +0.025 [+0.010, +0.040]ᵇ | [+0.012, +0.039] | – |

ᵃ Equivalent: the 95 % interval lies within ±0.01. ᵇ Different: the 95 % interval excludes zero. Other differences are inconclusive. Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; HGT: heterogeneous graph transformer; ROC-AUC: area under the ROC curve; Holm p: p-value adjusted with Holm's method; rolling rule: year-specific author, venue and keyword nodes that aggregate only earlier papers; Δ: paired difference (first model minus second model, or model minus reference).
