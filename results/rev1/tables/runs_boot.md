**Table @runs_boot.** Paired differences of the primary protocol (*k* = 3, 15 runs) with two 95 % intervals: the bootstrap over the test papers used in the article, which keeps the 15 trained models fixed, and a bootstrap that resamples the test papers and the 15 runs. Exploratory analysis.

| Metric | Contrast | Role | Δ [95 % interval, papers] | Δ [95 % interval, papers and runs] |
|---|---|---|---|---|
| Spearman | GraphSAGE − MLP | registered | −0.016 [−0.025, −0.006]ᵇ | −0.016 [−0.026, −0.005]ᵇ |
| Spearman | GAT − MLP | registered | −0.024 [−0.034, −0.013]ᵇ | −0.024 [−0.035, −0.012]ᵇ |
| Spearman | MLP + all-reference aggregates − MLP | registered | +0.007 [+0.001, +0.014]ᵇ | +0.007 [+0.0002, +0.014]ᵇ |
| ROC-AUC | HGT (rolling rule) − MLP | registered | +0.017 [+0.008, +0.027]ᵇ | +0.017 [+0.005, +0.029]ᵇ |
| ROC-AUC | GraphSAGE − MLP | registered | +0.005 [−0.002, +0.013] | +0.005 [−0.004, +0.015] |
| ROC-AUC | GAT − MLP | exploratory | +0.010 [+0.001, +0.020]ᵇ | +0.010 [−0.001, +0.023] |
| ROC-AUC | HAN (temporal rule) − MLP | exploratory | +0.012 [+0.001, +0.022]ᵇ | +0.012 [−0.0004, +0.024] |
| ROC-AUC | HGT without keyword nodes − HGT | exploratory | +0.006 [+0.0003, +0.012]ᵇ | +0.006 [−0.002, +0.014] |
| ROC-AUC | HGT without author nodes and features − MLP without author features | exploratory | +0.014 [+0.002, +0.025]ᵇ | +0.014 [+0.0002, +0.026]ᵇ |

ᵇ Different: the 95 % interval excludes zero. The other differences are inconclusive. Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; HAN: heterogeneous graph attention network; HGT: heterogeneous graph transformer; ROC-AUC: area under the ROC curve; rolling rule: year-specific author, venue and keyword nodes that aggregate only earlier papers; temporal rule: a metapath neighbour is kept only if it is earlier than the target paper; Δ: paired difference (first model minus second model, or model minus reference).
