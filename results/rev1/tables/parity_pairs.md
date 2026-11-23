**Table @parity_pairs.** Paired differences in Spearman correlation for the feature-parity comparisons (first model minus second model, 95 % intervals). Exploratory analysis.

| Model | Compared with | Primary protocol | Retrospective K-fold |
|---|---|---|---|
| GraphSAGE | MLP + in-graph aggregates | −0.019 [−0.029, −0.008]ᵇ | −0.005 [−0.0102, −0.0001]ᵇ |
| GAT | MLP + in-graph aggregates | −0.027 [−0.038, −0.015]ᵇ | −0.002 [−0.007, +0.003]ᵃ |
| GraphSAGE + all-reference aggregates | MLP + all-reference aggregates | −0.010 [−0.017, −0.002]ᵇ | +0.002 [−0.003, +0.007]ᵃ |
| GAT + all-reference aggregates | MLP + all-reference aggregates | −0.018 [−0.026, −0.009]ᵇ | +0.004 [−0.003, +0.011] |
| GraphSAGE | MLP + all-reference aggregates | −0.023 [−0.033, −0.011]ᵇ | −0.003 [−0.006, +0.001]ᵃ |
| GAT | MLP + all-reference aggregates | −0.031 [−0.042, −0.019]ᵇ | 0.000 [−0.006, +0.007]ᵃ |
| MLP + all-reference aggregates | MLP + in-graph aggregates | +0.004 [−0.004, +0.012] | −0.003 [−0.006, +0.0005]ᵃ |
| GraphSAGE, every reference as a node | MLP + all-reference aggregates | −0.014 [−0.025, −0.003]ᵇ | +0.007 [+0.002, +0.013]ᵇ |
| GAT, every reference as a node | MLP + all-reference aggregates | −0.015 [−0.026, −0.005]ᵇ | +0.002 [−0.003, +0.007]ᵃ |
| GraphSAGE, every reference as a node | GraphSAGE | +0.009 [−0.001, +0.019] | +0.010 [+0.003, +0.016]ᵇ |
| GAT, every reference as a node | GAT | +0.016 [+0.007, +0.025]ᵇ | +0.002 [−0.005, +0.009]ᵃ |
| GraphSAGE, every reference as a node | MLP | −0.007 [−0.018, +0.004] | +0.021 [+0.014, +0.029]ᵇ |
| GAT, every reference as a node | MLP | −0.008 [−0.019, +0.003] | +0.016 [+0.009, +0.023]ᵇ |

ᵃ Equivalent: the 95 % interval lies within ±0.01. ᵇ Different: the 95 % interval excludes zero. Other differences are inconclusive. Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; K-fold: 6-fold cross-validation repeated twice (retrospective).
