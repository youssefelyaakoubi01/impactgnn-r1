**Table @hpo.** Hyperparameters selected for each model family, task and protocol among 24 configurations, with the mean validation metric of the initial configuration and of the selected one (Spearman correlation for regression, ROC-AUC for classification). Every selected configuration uses 64 text components. Model selection on validation data; no test result.

| Family | Task, protocol | Hidden units | Layers | Heads | Dropout | Learning rate | Weight decay | Validation: initial | Validation: selected |
|---|---|---|---|---|---|---|---|---|---|
| GAT | classification, K-fold | 256 | 3 | 8 | 0.1 | 0.001 | 0.01 | 0.829 | 0.852 |
| GAT | classification, primary | 256 | 1 | 2 | 0.5 | 0.003 | 0.0001 | 0.791 | 0.831 |
| GAT | regression, K-fold | 256 | 2 | 8 | 0.5 | 0.01 | 0 | 0.678 | 0.699 |
| GAT | regression, primary | 64 | 2 | 2 | 0.5 | 0.01 | 0.001 | 0.630 | 0.657 |
| HAN | classification, K-fold | 64 | 3 | 8 | 0.3 | 0.001 | 0.0001 | 0.830 | 0.860 |
| HAN | classification, primary | 64 | 2 | 8 | 0.5 | 0.0003 | 0.0001 | 0.796 | 0.831 |
| HGT | classification, K-fold | 128 | 3 | 8 | 0.5 | 0.001 | 0 | 0.836 | 0.871 |
| HGT | classification, primary | 128 | 2 | 4 | 0.3 | 0.003 | 0.0001 | 0.802 | 0.840 |
| MLP | classification, K-fold | 64 | 1 | – | 0.1 | 0.001 | 0.01 | 0.813 | 0.843 |
| MLP | classification, primary | 256 | 1 | – | 0.5 | 0.003 | 0.001 | 0.777 | 0.811 |
| MLP | regression, K-fold | 128 | 3 | – | 0.1 | 0.01 | 0.0001 | 0.672 | 0.683 |
| MLP | regression, primary | 256 | 1 | – | 0.1 | 0.003 | 0.0001 | 0.619 | 0.647 |
| GraphSAGE | classification, K-fold | 256 | 1 | – | 0.5 | 0.0003 | 0.0001 | 0.828 | 0.857 |
| GraphSAGE | classification, primary | 64 | 1 | – | 0.5 | 0.0003 | 0.001 | 0.778 | 0.832 |
| GraphSAGE | regression, K-fold | 64 | 2 | – | 0.5 | 0.003 | 0 | 0.678 | 0.693 |
| GraphSAGE | regression, primary | 64 | 2 | – | 0.3 | 0.003 | 0 | 0.625 | 0.667 |

Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; HAN: heterogeneous graph attention network; HGT: heterogeneous graph transformer; K-fold: 6-fold cross-validation repeated twice (retrospective).
