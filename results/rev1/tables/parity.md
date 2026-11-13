**Table @parity.** Feature parity between the graph-free baselines and the GNNs at *k* = 3 (Spearman correlation; primary protocol: mean ± std over 15 runs; retrospective K-fold: over 12 test folds). Exploratory analysis.

| Model and reference information | Primary protocol | Retrospective K-fold |
|---|---|---|
| MLP | 0.639 ± 0.005 | 0.671 ± 0.014 |
| MLP + aggregates of the in-graph references | 0.642 ± 0.006 | 0.687 ± 0.011 |
| MLP + aggregates of all references | 0.646 ± 0.004 | 0.685 ± 0.012 |
| GraphSAGE | 0.624 ± 0.009 | 0.682 ± 0.013 |
| GraphSAGE + aggregates of all references | 0.636 ± 0.005 | 0.687 ± 0.012 |
| GAT | 0.615 ± 0.009 | 0.685 ± 0.012 |
| GAT + aggregates of all references | 0.629 ± 0.009 | 0.689 ± 0.011 |
| GraphSAGE, every reference as a node | 0.632 ± 0.006 | 0.692 ± 0.013 |
| GAT, every reference as a node | 0.631 ± 0.007 | 0.687 ± 0.011 |

Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; K-fold: 6-fold cross-validation repeated twice (retrospective).
