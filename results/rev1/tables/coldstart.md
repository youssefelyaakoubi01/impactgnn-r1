**Table @coldstart.** Cold start: test papers absent from the training graph (Spearman correlation). Primary protocol: *k* = 3, labels closed before the prediction date, mean ± std over 15 runs. Retrospective temporal split: mean ± std over 5 seeds. Primary analysis for the primary protocol; the other columns are retrospective.

| Variant | Primary protocol | Primary: Δ vs MLP, all features + reference aggregates | Retrospective, *k* = 3 | Retrospective, *k* = 5 |
|---|---|---|---|---|
| MLP: authors | 0.580 ± 0.006 | −0.066 [−0.084, −0.049]ᵇ | 0.583 ± 0.001 | 0.585 ± 0.001 |
| MLP: reference aggregates | 0.549 ± 0.003 | −0.097 [−0.119, −0.074]ᵇ | 0.584 ± 0.002 | 0.590 ± 0.001 |
| MLP: authors + reference aggregates | 0.604 ± 0.005 | −0.042 [−0.057, −0.027]ᵇ | 0.628 ± 0.001 | 0.639 ± 0.001 |
| GraphSAGE: references only (own text hidden) | 0.523 ± 0.015 | −0.123 [−0.145, −0.099]ᵇ | 0.566 ± 0.001 | 0.583 ± 0.003 |
| GraphSAGE: references + authors (own text hidden) | 0.584 ± 0.012 | −0.063 [−0.080, −0.044]ᵇ | 0.620 ± 0.004 | 0.631 ± 0.003 |
| GraphSAGE: references + authors + own text | 0.601 ± 0.008 | −0.045 [−0.058, −0.032]ᵇ | 0.622 ± 0.002 | 0.629 ± 0.002 |
| GraphSAGE: all features, edge features | 0.630 ± 0.005 | −0.017 [−0.026, −0.007]ᵇ | 0.655 ± 0.003 | 0.671 ± 0.005 |
| GraphSAGE: all features | 0.624 ± 0.009 | −0.023 [−0.033, −0.011]ᵇ | 0.652 ± 0.002 | 0.663 ± 0.005 |
| MLP: all features + reference aggregates | 0.646 ± 0.004 | reference | 0.670 ± 0.002 | 0.681 ± 0.002 |

ᵇ Different: the 95 % interval excludes zero. The other differences are inconclusive. None of these differences is a registered contrast. Abbreviations: MLP: multilayer perceptron; *k*: horizon in years; Δ: paired difference (first model minus second model, or model minus reference).
