**Table @textdim.** Text dimension in the retrospective K-fold at *k* = 3, with the fixed configuration (mean ± std over 12 test folds). The 64 principal components are fitted on the training graph of each fold. Exploratory analysis.

| Model | Metric | SPECTER, 768 dimensions | SPECTER, 64 components | SciBERT, 768 dimensions | SciBERT, 64 components |
|---|---|---|---|---|---|
| MLP | Spearman | 0.649 ± 0.011 | 0.665 ± 0.010 | 0.658 ± 0.012 | 0.672 ± 0.014 |
| GraphSAGE | Spearman | 0.655 ± 0.011 | 0.674 ± 0.009 | 0.665 ± 0.011 | 0.680 ± 0.013 |
| GAT | Spearman | 0.653 ± 0.012 | 0.676 ± 0.010 | 0.662 ± 0.010 | 0.683 ± 0.013 |
| MLP | ROC-AUC | 0.810 ± 0.009 | 0.841 ± 0.010 | 0.817 ± 0.011 | 0.838 ± 0.009 |
| HAN (temporal rule) | ROC-AUC | 0.825 ± 0.011 | 0.847 ± 0.009 | 0.832 ± 0.014 | 0.851 ± 0.009 |
| HGT (T0 rule) | ROC-AUC | 0.833 ± 0.013 | 0.856 ± 0.014 | 0.831 ± 0.011 | 0.859 ± 0.010 |
| HGT (rolling rule) | ROC-AUC | 0.836 ± 0.012 | 0.857 ± 0.013 | 0.835 ± 0.011 | 0.861 ± 0.010 |

Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; HAN: heterogeneous graph attention network; HGT: heterogeneous graph transformer; ROC-AUC: area under the ROC curve; T0 rule: author, venue and keyword nodes aggregate only the papers published before 2017; rolling rule: year-specific author, venue and keyword nodes that aggregate only earlier papers; temporal rule: a metapath neighbour is kept only if it is earlier than the target paper.
