**Table @rq1.** Regression of log(1 + citations): mean ± std over runs (primary protocol) or over test folds (retrospective K-fold). Δ is the paired difference in Spearman correlation with the MLP and its 95 % interval. Primary: labels closed at the prediction date; K-fold: retrospective. Tuned hyperparameters, SciBERT text. Primary analysis for the rows of the primary protocol; the K-fold rows are retrospective.

| Protocol | Model | Spearman | R² | MAE | Δ Spearman vs MLP |
|---|---|---|---|---|---|
| Primary (15 runs) | MLP | 0.639 ± 0.005 | 0.367 ± 0.014 | 0.784 ± 0.009 | reference |
|  | MLP + in-graph aggregates | 0.642 ± 0.006 | 0.397 ± 0.009 | 0.768 ± 0.006 | +0.003 [−0.004, +0.0102] |
|  | MLP + all-reference aggregates | 0.646 ± 0.004 | 0.363 ± 0.014 | 0.782 ± 0.010 | +0.007 [+0.001, +0.014]ᵇ |
|  | GraphSAGE | 0.624 ± 0.009 | 0.303 ± 0.032 | 0.821 ± 0.018 | −0.016 [−0.025, −0.006]ᵇ |
|  | GraphSAGE + edge features | 0.630 ± 0.005 | 0.318 ± 0.019 | 0.814 ± 0.010 | −0.010 [−0.018, −0.001]ᵇ |
|  | GAT | 0.615 ± 0.009 | 0.290 ± 0.052 | 0.831 ± 0.029 | −0.024 [−0.034, −0.013]ᵇ |
|  | GAT + edge features | 0.612 ± 0.012 | 0.265 ± 0.046 | 0.845 ± 0.027 | −0.027 [−0.038, −0.016]ᵇ |
| K-fold, *k* = 3 (12 folds) | MLP | 0.671 ± 0.014 | 0.444 ± 0.022 | 0.754 ± 0.019 | reference |
|  | MLP + in-graph aggregates | 0.687 ± 0.011 | 0.472 ± 0.018 | 0.735 ± 0.016 | +0.017 [+0.012, +0.021]ᵇ |
|  | MLP + all-reference aggregates | 0.685 ± 0.012 | 0.466 ± 0.017 | 0.740 ± 0.015 | +0.014 [+0.009, +0.019]ᵇ |
|  | GraphSAGE | 0.682 ± 0.013 | 0.457 ± 0.018 | 0.744 ± 0.015 | +0.012 [+0.006, +0.017]ᵇ |
|  | GraphSAGE + edge features | 0.685 ± 0.012 | 0.465 ± 0.019 | 0.740 ± 0.017 | +0.014 [+0.008, +0.020]ᵇ |
|  | GAT | 0.685 ± 0.012 | 0.468 ± 0.016 | 0.737 ± 0.018 | +0.015 [+0.008, +0.022]ᵇ |
|  | GAT + edge features | 0.680 ± 0.011 | 0.460 ± 0.015 | 0.743 ± 0.016 | +0.009 [+0.003, +0.016]ᵇ |
| K-fold, *k* = 5 (12 folds) | MLP | 0.672 ± 0.019 | 0.445 ± 0.028 | 0.814 ± 0.027 | reference |
|  | MLP + in-graph aggregates | 0.695 ± 0.016 | 0.480 ± 0.022 | 0.786 ± 0.021 | +0.023 [+0.014, +0.031]ᵇ |
|  | MLP + all-reference aggregates | 0.691 ± 0.017 | 0.472 ± 0.025 | 0.794 ± 0.022 | +0.019 [+0.013, +0.026]ᵇ |
|  | GraphSAGE | 0.689 ± 0.015 | 0.464 ± 0.021 | 0.798 ± 0.022 | +0.017 [+0.007, +0.027]ᵇ |
|  | GraphSAGE + edge features | 0.693 ± 0.016 | 0.478 ± 0.023 | 0.787 ± 0.024 | +0.021 [+0.014, +0.029]ᵇ |
|  | GAT | 0.692 ± 0.013 | 0.479 ± 0.017 | 0.787 ± 0.021 | +0.020 [+0.009, +0.032]ᵇ |
|  | GAT + edge features | 0.687 ± 0.013 | 0.472 ± 0.016 | 0.792 ± 0.019 | +0.015 [+0.004, +0.027]ᵇ |

ᵇ Different: the 95 % interval excludes zero. The other differences are inconclusive. "Aggregates": seven summary statistics of the references (Section 4.3), computed over the in-graph references or over all references. Registered contrasts (Holm-adjusted p-values in Section 5.10): GraphSAGE, GAT and MLP + all-reference aggregates against the MLP in the primary protocol; the other differences are exploratory. Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; R²: coefficient of determination; MAE: mean absolute error on the log scale; K-fold: 6-fold cross-validation repeated twice (retrospective); *k*: horizon in years; Δ: paired difference (first model minus second model, or model minus reference).
