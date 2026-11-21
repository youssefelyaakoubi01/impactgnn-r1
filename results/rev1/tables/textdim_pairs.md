**Table @textdim_pairs.** Paired differences in ROC-AUC between the heterogeneous models and the MLP on the same partitions, with 768-dimensional text and with 64 components (95 % intervals). Exploratory analysis.

| Protocol | Text | Contrast | 768 dimensions | 64 components |
|---|---|---|---|---|
| K-fold, fixed configuration | SPECTER | HGT (T0 rule) − MLP | +0.022 [+0.014, +0.031]ᵇ | +0.015 [+0.007, +0.023]ᵇ |
| K-fold, fixed configuration | SPECTER | HGT (rolling rule) − MLP | +0.026 [+0.018, +0.034]ᵇ | +0.016 [+0.009, +0.023]ᵇ |
| K-fold, fixed configuration | SPECTER | HAN (temporal rule) − MLP | +0.014 [+0.007, +0.022]ᵇ | +0.006 [+0.002, +0.011]ᵇ |
| K-fold, fixed configuration | SciBERT | HGT (T0 rule) − MLP | +0.015 [+0.009, +0.021]ᵇ | +0.021 [+0.017, +0.025]ᵇ |
| K-fold, fixed configuration | SciBERT | HGT (rolling rule) − MLP | +0.018 [+0.012, +0.025]ᵇ | +0.023 [+0.019, +0.028]ᵇ |
| K-fold, fixed configuration | SciBERT | HAN (temporal rule) − MLP | +0.015 [+0.005, +0.026]ᵇ | +0.013 [+0.006, +0.020]ᵇ |
| Primary protocol, tuned hyperparameters | SPECTER | HGT (rolling rule) − MLP | +0.027 [+0.016, +0.037]ᵇ | +0.014 [+0.002, +0.025]ᵇ |
| Primary protocol, tuned hyperparameters | SPECTER | HAN (temporal rule) − MLP | +0.021 [+0.012, +0.029]ᵇ | +0.012 [−0.0002, +0.023] |
| Primary protocol, tuned hyperparameters | SciBERT | HGT (rolling rule) − MLP | +0.010 [+0.002, +0.019]ᵇ | +0.017 [+0.008, +0.027]ᵇ |
| Primary protocol, tuned hyperparameters | SciBERT | HAN (temporal rule) − MLP | +0.012 [+0.004, +0.020]ᵇ | +0.012 [+0.001, +0.022]ᵇ |
| K-fold, fixed configuration | SPECTER, components fitted on all nodes | HGT (T0 rule) − MLP | – | +0.012 [+0.006, +0.018]ᵇ |

ᵇ Different: the 95 % interval excludes zero. The other differences are inconclusive. Abbreviations: MLP: multilayer perceptron; HAN: heterogeneous graph attention network; HGT: heterogeneous graph transformer; K-fold: 6-fold cross-validation repeated twice (retrospective); T0 rule: author, venue and keyword nodes aggregate only the papers published before 2017; rolling rule: year-specific author, venue and keyword nodes that aggregate only earlier papers; temporal rule: a metapath neighbour is kept only if it is earlier than the target paper.
