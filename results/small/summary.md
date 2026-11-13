# Axis 1 – Citation graphs and GNNs: results summary (small run)

Generated 2026-10-25 17:13 by `summarize.py` from `data/` and `results/`.

## 1. Data

### 1.1 PDF matching (PyMuPDF + OpenAlex)

| corpus | distinct PDFs | matched | duplicate work | inherited match rejected | OpenAlex DOI printed in PDF | match method |
|---|---|---|---|---|---|---|
| lab | 37 | 35 | 0 | 0 | 28 | fbn_match: 32, doi_lookup: 3, unmatched: 2 |
| open | 1010 | 967 | 0 | 12 | 431 | manifest_openalex_id: 510, fbn_match: 448, unmatched: 43, doi_lookup: 9 |


Lab folder: 40 files, 3 exact duplicates (sha256) removed.

### 1.2 OpenAlex collection

Calls = distinct API calls of the step (each sent once, then served from the disk cache). Credits are reconstructed from the remaining-credit values printed in `logs/collect_<profile>.log` (an upward jump means a new API key with a daily limit of 10,000 credits).

| step | calls | credits consumed | seconds (last run, from cache) |
|---|---|---|---|
| seeds | – | 27 | – |
| sample | – | 6 | – |
| refs | – | 258 | – |
| nodes | – | 36 | – |
| authors | – | 100 | – |
| venues | – | 74 | – |
| total | 0 | 501 |  |

### 1.3 Topics of the labeled sample

| topic | name | group | stratum | papers sampled |
|---|---|---|---|---|
| T10246 | Mobile Ad Hoc Networks | lab |  | 100 |
| T13373 | Data Mining and Machine Learning Applications | lab |  | 100 |
| T10845 | Advanced Causal Inference Techniques | open | econ | 100 |
| T10621 | Gene Regulatory Network Analysis | open | Biochemistry, Genetics and Molecular Biology | 100 |

### 1.4 Citation graph

| quantity | value |
|---|---|
| paper nodes | 3171 |
| nodes by origin | ref: 1769, seed_open: 967, sample: 400, seed_lab: 35 |
| known works (with metadata) | 14073 |
| citation edges (reference -> citing) | 9963 |
| edges dropped (reference published after citing paper) | 135 |
| share of nodes with an abstract | 0.809 |
| author records / venue records | 4964 / 3667 |
| sample by topic group | open: 200, lab: 200 |

### 1.5 Labels (citations in years y..y+k, complete windows only)

| k | labeled papers | by year | mean citations | median | share zero | top-10 % rate | in-graph refs mean | median | share with none |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 735 | 2017: 63, 2018: 104, 2019: 119, 2020: 118, 2021: 125, 2022: 206 | 12.02 | 4.0 | 0.184 | 0.107 | 3.86 | 1.0 | 0.479 |
| 5 | 404 | 2017: 63, 2018: 104, 2019: 119, 2020: 118 | 17.53 | 6.0 | 0.119 | 0.106 | 4.25 | 1.0 | 0.418 |

### 1.6 Heterogeneous graph and venue graph

| quantity | value |
|---|---|
| authors | 4964 |
| venues | 3667 |
| keywords | 3033 |
| edges_cites | 9963 |
| edges_fwd_author | 5664 |
| edges_rev_all_author | 5664 |
| edges_rev_safe_author | 365 |
| edges_fwd_venue | 3068 |
| edges_rev_all_venue | 3068 |
| edges_rev_safe_venue | 1086 |
| edges_fwd_keyword | 13023 |
| edges_rev_all_keyword | 13023 |
| edges_rev_safe_keyword | 5664 |
| edges_mp_t0_author | 2309 |
| edges_mp_temporal_author | 1898 |
| edges_mp_t0_venue | 8071 |
| edges_mp_temporal_venue | 9496 |
| edges_mp_t0_keyword | 28235 |
| edges_mp_temporal_keyword | 28179 |
| venue graph venues | 155 |
| venue graph edges | 1441 |
| venue graph links | 4051 |
| venue graph self_links | 513 |



## 2. Sub-topic 1: future impact with GAT (random split 70/15/15, regression on log1p citations)

### k = 3

| model | Spearman | R² | MAE (log) | runs |
|---|---|---|---|---|
| MLP (no graph) | 0.218 ± 0.000 | -0.056 ± 0.000 | 0.949 ± 0.000 | 1 |
| MLP + reference aggregates | 0.285 ± 0.000 | 0.027 ± 0.000 | 0.923 ± 0.000 | 1 |
| GraphSAGE | 0.211 ± 0.000 | -0.054 ± 0.000 | 0.962 ± 0.000 | 1 |
| GraphSAGE + edge features | 0.234 ± 0.000 | -0.084 ± 0.000 | 0.972 ± 0.000 | 1 |
| GAT | 0.191 ± 0.000 | -0.076 ± 0.000 | 0.959 ± 0.000 | 1 |
| GAT + edge features | 0.277 ± 0.000 | 0.015 ± 0.000 | 0.908 ± 0.000 | 1 |

### k = 5

| model | Spearman | R² | MAE (log) | runs |
|---|---|---|---|---|
| MLP (no graph) | 0.165 ± 0.000 | -0.341 ± 0.000 | 1.034 ± 0.000 | 1 |
| MLP + reference aggregates | 0.171 ± 0.000 | -0.266 ± 0.000 | 1.031 ± 0.000 | 1 |
| GraphSAGE | 0.145 ± 0.000 | -0.093 ± 0.000 | 0.953 ± 0.000 | 1 |
| GraphSAGE + edge features | 0.219 ± 0.000 | -0.212 ± 0.000 | 0.980 ± 0.000 | 1 |
| GAT | 0.164 ± 0.000 | -0.208 ± 0.000 | 0.977 ± 0.000 | 1 |
| GAT + edge features | 0.212 ± 0.000 | -0.056 ± 0.000 | 0.952 ± 0.000 | 1 |

### Attention analysis (test papers with >= 3 in-graph references; self-loops excluded)

Per-paper = Spearman between attention and the reference attribute within each citing paper, averaged over papers; pooled = Spearman between degree-normalised attention and the attribute over all edges.

| model k | weights | per-paper ρ(att, ref citations) | share ρ>0 | pooled ρ(cit) | per-paper ρ(att, ref age) | share ρ>0 | pooled ρ(age) | papers |
|---|---|---|---|---|---|---|---|---|
| gat k3 | trained | 0.062 ± 0.000 | 0.612 ± 0.000 | 0.068 ± 0.000 | 0.057 ± 0.000 | 0.490 ± 0.000 | -0.001 ± 0.000 | 49 ± 0 |
| gat k3 | untrained | -0.029 ± 0.000 | 0.469 ± 0.000 | -0.050 ± 0.000 | 0.039 ± 0.000 | 0.531 ± 0.000 | 0.016 ± 0.000 | 49 ± 0 |
| gat k3 | trained_layer1 | -0.195 ± 0.000 | 0.327 ± 0.000 | -0.167 ± 0.000 | -0.192 ± 0.000 | 0.286 ± 0.000 | -0.206 ± 0.000 | 49 ± 0 |
| gat k5 | trained | 0.027 ± 0.000 | 0.500 ± 0.000 | -0.026 ± 0.000 | -0.010 ± 0.000 | 0.467 ± 0.000 | -0.106 ± 0.000 | 30 ± 0 |
| gat k5 | untrained | -0.045 ± 0.000 | 0.400 ± 0.000 | -0.004 ± 0.000 | 0.100 ± 0.000 | 0.533 ± 0.000 | 0.093 ± 0.000 | 30 ± 0 |
| gat k5 | trained_layer1 | -0.351 ± 0.000 | 0.200 ± 0.000 | -0.264 ± 0.000 | -0.385 ± 0.000 | 0.100 ± 0.000 | -0.317 ± 0.000 | 30 ± 0 |
| gat_e k3 | trained | 0.131 ± 0.000 | 0.653 ± 0.000 | 0.104 ± 0.000 | 0.013 ± 0.000 | 0.531 ± 0.000 | 0.060 ± 0.000 | 49 ± 0 |
| gat_e k3 | untrained | 0.216 ± 0.000 | 0.796 ± 0.000 | 0.138 ± 0.000 | 0.092 ± 0.000 | 0.612 ± 0.000 | 0.105 ± 0.000 | 49 ± 0 |
| gat_e k3 | trained_layer1 | 0.229 ± 0.000 | 0.735 ± 0.000 | 0.172 ± 0.000 | 0.341 ± 0.000 | 0.816 ± 0.000 | 0.222 ± 0.000 | 49 ± 0 |
| gat_e k5 | trained | 0.080 ± 0.000 | 0.567 ± 0.000 | 0.086 ± 0.000 | -0.019 ± 0.000 | 0.467 ± 0.000 | 0.066 ± 0.000 | 30 ± 0 |
| gat_e k5 | untrained | 0.150 ± 0.000 | 0.767 ± 0.000 | 0.116 ± 0.000 | 0.069 ± 0.000 | 0.567 ± 0.000 | 0.052 ± 0.000 | 30 ± 0 |
| gat_e k5 | trained_layer1 | -0.248 ± 0.000 | 0.233 ± 0.000 | -0.209 ± 0.000 | -0.078 ± 0.000 | 0.333 ± 0.000 | -0.067 ± 0.000 | 30 ± 0 |

- gat k3: mean self-loop attention on test papers = 0.548 ± 0.000

- gat k5: mean self-loop attention on test papers = 0.469 ± 0.000

- gat_e k3: mean self-loop attention on test papers = 0.545 ± 0.000

- gat_e k5: mean self-loop attention on test papers = 0.466 ± 0.000

### Examples (GAT, attention averaged over seeds; highest-impact labeled papers)

- **Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects** (2021, Journal of Economic Literature; 828 citations, 34 references in graph, uniform attention = 0.029)

    - top: Quasi-Experimentation: Design & Analysis Issues for Field Settings (1979; 10745 citations before citing year; attention 0.061)

    - top: Journal of The American Statistical Association (2004; 5393 citations before citing year; attention 0.049)

    - top: The Use of Matched Sampling and Regression Adjustment to Remove Bias in Observational Studies (1973; 306 citations before citing year; attention 0.047)

    - bottom: Inferring causal impact using Bayesian structural time-series models (2015; 263 citations before citing year; attention 0.018)

    - bottom: Balancing, Regression, Difference-In-Differences and Synthetic Control Methods: A Synthesis (2016; 113 citations before citing year; attention 0.019)

    - bottom: Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California’s Tobacco Control Program (2010; 2078 citations before citing year; attention 0.019)

- **Comparison of Propensity Score Methods and Covariate Adjustment** (2017, Journal of the American College of Cardiology; 194 citations, 16 references in graph, uniform attention = 0.062)

    - top: Design of Experiments (1936; 3722 citations before citing year; attention 0.081)

    - top: Propensity scores: From naïve enthusiasm to intuitive understanding (2011; 76 citations before citing year; attention 0.066)

    - top: Demystifying Double Robustness: A Comparison of Alternative Strategies for Estimating a Population Mean from Incomplete Data (2007; 383 citations before citing year; attention 0.065)

    - bottom: The central role of the propensity score in observational studies for causal effects (1983; 13020 citations before citing year; attention 0.049)

    - bottom: An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies (2011; 1294 citations before citing year; attention 0.050)

    - bottom: Regression modelling strategies for improved prognostic prediction (1984; 852 citations before citing year; attention 0.050)

- **The sound of silence: Transgene silencing in mammalian cell engineering** (2022, Cell Systems; 141 citations, 14 references in graph, uniform attention = 0.071)

    - top: Dynamics and Memory of Heterochromatin in Living Cells (2012; 314 citations before citing year; attention 0.102)

    - top: An integrated encyclopedia of DNA elements in the human genome (2012; 12379 citations before citing year; attention 0.090)

    - top: Chromatin Modifications and Their Function (2007; 8981 citations before citing year; attention 0.090)

    - bottom: A platform for rapid prototyping of synthetic gene networks in mammalian cells (2014; 83 citations before citing year; attention 0.048)

    - bottom: Dynamics of epigenetic regulation at the single-cell level (2016; 266 citations before citing year; attention 0.052)

    - bottom: Programmable protein circuits in living cells (2018; 123 citations before citing year; attention 0.053)

- **Chemically induced proximity in biology and medicine** (2018, Science; 129 citations, 18 references in graph, uniform attention = 0.056)

    - top: Complex transcriptional modulation with orthogonal and inducible dCas9 regulators (2016; 35 citations before citing year; attention 0.081)

    - top: Transcription control by the ENL YEATS domain in acute leukaemia (2017; 22 citations before citing year; attention 0.078)

    - top: Dynamics and Memory of Heterochromatin in Living Cells (2012; 191 citations before citing year; attention 0.068)

    - bottom: A Rapid, Reversible, and Tunable Method to Regulate Protein Function in Living Cells Using Synthetic Small Molecules (2006; 439 citations before citing year; attention 0.034)

    - bottom: Rapid and reversible epigenome editing by endogenous chromatin regulators (2017; 1 citations before citing year; attention 0.036)

    - bottom: Manipulation of nuclear architecture through CRISPR-mediated chromosomal looping (2017; 6 citations before citing year; attention 0.036)

- **Locally Robust Semiparametric Estimation** (2022, Econometrica; 99 citations, 31 references in graph, uniform attention = 0.032)

    - top: Targeted Learning (2011; 368 citations before citing year; attention 0.198)

    - top: Contributions to a General Asymptotic Statistical Theory (1982; 133 citations before citing year; attention 0.100)

    - top: High-dimensional inference for the average treatment effect under model misspecification using penalized bias-reduced double-robust estimation (2021; 3 citations before citing year; attention 0.045)

    - bottom: Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher (1987; 1446 citations before citing year; attention 0.014)

    - bottom: Twicing Kernels and a Small Bias Property of Semiparametric Estimators (2004; 80 citations before citing year; attention 0.015)

    - bottom: The influence function of semiparametric estimators (2022; 2 citations before citing year; attention 0.017)



## 3. Sub-topic 2: inductive cold start with GraphSAGE (temporal split)

### k = 3: train < 2020, validation = 2020, test >= 2021 (training graph: 1920 nodes, 7489 edges; test papers absent)

| variant | Spearman | R² | MAE (log) | runs |
|---|---|---|---|---|
| mlp_authors | 0.324 ± 0.000 | -0.116 ± 0.000 | 1.037 ± 0.000 | 1 |
| mlp_refagg | 0.308 ± 0.000 | -0.026 ± 0.000 | 1.026 ± 0.000 | 1 |
| mlp_authors_refagg | 0.340 ± 0.000 | 0.044 ± 0.000 | 0.986 ± 0.000 | 1 |
| sage_refs | 0.360 ± 0.000 | 0.067 ± 0.000 | 0.996 ± 0.000 | 1 |
| sage_refs_authors | 0.527 ± 0.000 | 0.225 ± 0.000 | 0.880 ± 0.000 | 1 |
| sage_refs_authors_text | 0.329 ± 0.000 | 0.061 ± 0.000 | 0.997 ± 0.000 | 1 |
| sage_full | 0.361 ± 0.000 | 0.104 ± 0.000 | 0.969 ± 0.000 | 1 |
| sage_plain_full | 0.333 ± 0.000 | 0.078 ± 0.000 | 0.986 ± 0.000 | 1 |
| mlp_full | 0.312 ± 0.000 | 0.053 ± 0.000 | 1.015 ± 0.000 | 1 |

### k = 5: train < 2019, validation = 2019, test >= 2020 (training graph: 1720 nodes, 6812 edges; test papers absent)

| variant | Spearman | R² | MAE (log) | runs |
|---|---|---|---|---|
| mlp_authors | 0.489 ± 0.000 | 0.178 ± 0.000 | 0.973 ± 0.000 | 1 |
| mlp_refagg | 0.512 ± 0.000 | 0.224 ± 0.000 | 0.928 ± 0.000 | 1 |
| mlp_authors_refagg | 0.514 ± 0.000 | 0.189 ± 0.000 | 0.952 ± 0.000 | 1 |
| sage_refs | 0.406 ± 0.000 | 0.123 ± 0.000 | 0.996 ± 0.000 | 1 |
| sage_refs_authors | 0.467 ± 0.000 | 0.126 ± 0.000 | 0.987 ± 0.000 | 1 |
| sage_refs_authors_text | 0.190 ± 0.000 | -0.045 ± 0.000 | 1.111 ± 0.000 | 1 |
| sage_full | 0.214 ± 0.000 | -0.046 ± 0.000 | 1.135 ± 0.000 | 1 |
| sage_plain_full | 0.264 ± 0.000 | -0.058 ± 0.000 | 1.111 ± 0.000 | 1 |
| mlp_full | 0.209 ± 0.000 | -0.107 ± 0.000 | 1.122 ± 0.000 | 1 |

### Lab papers (deployment models trained on all labeled years; mean over seeds)

Not published (see the ethical considerations of the article).

## 4. Sub-topic 3: heterogeneous graph, top-10 % detection (random split)

### k = 3

| model | ROC-AUC | AP | precision@10% | runs |
|---|---|---|---|---|
| MLP (no graph) | 0.492 ± 0.000 | 0.329 ± 0.000 | 0.250 ± 0.000 | 1 |
| GraphSAGE | 0.476 ± 0.000 | 0.326 ± 0.000 | 0.250 ± 0.000 | 1 |
| GAT | 0.532 ± 0.000 | 0.252 ± 0.000 | 0.250 ± 0.000 | 1 |
| HAN (T0 rule) | 0.484 ± 0.000 | 0.327 ± 0.000 | 0.250 ± 0.000 | 1 |
| HAN (temporal rule) | 0.476 ± 0.000 | 0.326 ± 0.000 | 0.250 ± 0.000 | 1 |
| HGT (T0 rule) | 0.532 ± 0.000 | 0.249 ± 0.000 | 0.250 ± 0.000 | 1 |
| HGT naive (leaky, reference only) | 0.529 ± 0.000 | 0.247 ± 0.000 | 0.250 ± 0.000 | 1 |

### k = 5

| model | ROC-AUC | AP | precision@10% | runs |
|---|---|---|---|---|
| MLP (no graph) | 0.662 ± 0.000 | 0.241 ± 0.000 | 0.286 ± 0.000 | 1 |
| GraphSAGE | 0.522 ± 0.000 | 0.325 ± 0.000 | 0.143 ± 0.000 | 1 |
| GAT | 0.577 ± 0.000 | 0.138 ± 0.000 | 0.143 ± 0.000 | 1 |
| HAN (T0 rule) | 0.599 ± 0.000 | 0.370 ± 0.000 | 0.286 ± 0.000 | 1 |
| HAN (temporal rule) | 0.583 ± 0.000 | 0.357 ± 0.000 | 0.286 ± 0.000 | 1 |
| HGT (T0 rule) | 0.645 ± 0.000 | 0.364 ± 0.000 | 0.286 ± 0.000 | 1 |
| HGT naive (leaky, reference only) | 0.645 ± 0.000 | 0.364 ± 0.000 | 0.286 ± 0.000 | 1 |



## 5. Sub-topic 4: citation cartels and self-citation (venue graph, injected anomalies)

Venue graph: 155 venues, 1441 weighted edges, 4051 citation links. Injection per draw: 5 cartels of 4 venues, 10 self-citing venues; intensity = share of the venue's outgoing citations added; 1 draws.

### intensity mild (0.1)

| detector | cartel AUC | cartel AP | self AUC | self AP | all AUC | all AP |
|---|---|---|---|---|---|---|
| dominant_content | 0.634 ± 0.000 | 0.203 ± 0.000 | 0.356 ± 0.000 | 0.059 ± 0.000 | 0.541 ± 0.000 | 0.217 ± 0.000 |
| dominant_full | 0.666 ± 0.000 | 0.215 ± 0.000 | 0.410 ± 0.000 | 0.064 ± 0.000 | 0.580 ± 0.000 | 0.229 ± 0.000 |
| gae_content | 0.579 ± 0.000 | 0.208 ± 0.000 | 0.487 ± 0.000 | 0.121 ± 0.000 | 0.549 ± 0.000 | 0.258 ± 0.000 |
| gae_full | 0.582 ± 0.000 | 0.204 ± 0.000 | 0.475 ± 0.000 | 0.080 ± 0.000 | 0.546 ± 0.000 | 0.238 ± 0.000 |
| heur_max | 0.748 ± 0.000 | 0.298 ± 0.000 | 0.829 ± 0.000 | 0.209 ± 0.000 | 0.775 ± 0.000 | 0.388 ± 0.000 |
| heur_reciprocity | 0.840 ± 0.000 | 0.414 ± 0.000 | 0.545 ± 0.000 | 0.132 ± 0.000 | 0.741 ± 0.000 | 0.412 ± 0.000 |
| heur_self_ratio | 0.405 ± 0.000 | 0.125 ± 0.000 | 0.882 ± 0.000 | 0.274 ± 0.000 | 0.564 ± 0.000 | 0.256 ± 0.000 |
| iforest_content | 0.537 ± 0.000 | 0.150 ± 0.000 | 0.296 ± 0.000 | 0.055 ± 0.000 | 0.457 ± 0.000 | 0.174 ± 0.000 |
| iforest_full | 0.568 ± 0.000 | 0.200 ± 0.000 | 0.284 ± 0.000 | 0.054 ± 0.000 | 0.474 ± 0.000 | 0.208 ± 0.000 |

### intensity medium (0.25)

| detector | cartel AUC | cartel AP | self AUC | self AP | all AUC | all AP |
|---|---|---|---|---|---|---|
| dominant_content | 0.673 ± 0.000 | 0.219 ± 0.000 | 0.373 ± 0.000 | 0.061 ± 0.000 | 0.573 ± 0.000 | 0.230 ± 0.000 |
| dominant_full | 0.704 ± 0.000 | 0.231 ± 0.000 | 0.527 ± 0.000 | 0.080 ± 0.000 | 0.645 ± 0.000 | 0.254 ± 0.000 |
| gae_content | 0.554 ± 0.000 | 0.183 ± 0.000 | 0.451 ± 0.000 | 0.097 ± 0.000 | 0.520 ± 0.000 | 0.227 ± 0.000 |
| gae_full | 0.592 ± 0.000 | 0.221 ± 0.000 | 0.432 ± 0.000 | 0.082 ± 0.000 | 0.538 ± 0.000 | 0.249 ± 0.000 |
| heur_max | 0.807 ± 0.000 | 0.342 ± 0.000 | 0.940 ± 0.000 | 0.408 ± 0.000 | 0.851 ± 0.000 | 0.498 ± 0.000 |
| heur_reciprocity | 0.881 ± 0.000 | 0.471 ± 0.000 | 0.545 ± 0.000 | 0.132 ± 0.000 | 0.769 ± 0.000 | 0.450 ± 0.000 |
| heur_self_ratio | 0.392 ± 0.000 | 0.119 ± 0.000 | 0.970 ± 0.000 | 0.551 ± 0.000 | 0.585 ± 0.000 | 0.343 ± 0.000 |
| iforest_content | 0.537 ± 0.000 | 0.150 ± 0.000 | 0.296 ± 0.000 | 0.055 ± 0.000 | 0.457 ± 0.000 | 0.174 ± 0.000 |
| iforest_full | 0.582 ± 0.000 | 0.181 ± 0.000 | 0.390 ± 0.000 | 0.062 ± 0.000 | 0.518 ± 0.000 | 0.203 ± 0.000 |

### intensity strong (0.5)

| detector | cartel AUC | cartel AP | self AUC | self AP | all AUC | all AP |
|---|---|---|---|---|---|---|
| dominant_content | 0.688 ± 0.000 | 0.230 ± 0.000 | 0.353 ± 0.000 | 0.059 ± 0.000 | 0.577 ± 0.000 | 0.236 ± 0.000 |
| dominant_full | 0.725 ± 0.000 | 0.234 ± 0.000 | 0.623 ± 0.000 | 0.097 ± 0.000 | 0.691 ± 0.000 | 0.273 ± 0.000 |
| gae_content | 0.482 ± 0.000 | 0.148 ± 0.000 | 0.397 ± 0.000 | 0.064 ± 0.000 | 0.453 ± 0.000 | 0.181 ± 0.000 |
| gae_full | 0.468 ± 0.000 | 0.133 ± 0.000 | 0.376 ± 0.000 | 0.061 ± 0.000 | 0.437 ± 0.000 | 0.169 ± 0.000 |
| heur_max | 0.831 ± 0.000 | 0.394 ± 0.000 | 0.963 ± 0.000 | 0.561 ± 0.000 | 0.875 ± 0.000 | 0.580 ± 0.000 |
| heur_reciprocity | 0.903 ± 0.000 | 0.529 ± 0.000 | 0.545 ± 0.000 | 0.132 ± 0.000 | 0.784 ± 0.000 | 0.489 ± 0.000 |
| heur_self_ratio | 0.372 ± 0.000 | 0.112 ± 0.000 | 0.984 ± 0.000 | 0.722 ± 0.000 | 0.576 ± 0.000 | 0.388 ± 0.000 |
| iforest_content | 0.537 ± 0.000 | 0.150 ± 0.000 | 0.296 ± 0.000 | 0.055 ± 0.000 | 0.457 ± 0.000 | 0.174 ± 0.000 |
| iforest_full | 0.598 ± 0.000 | 0.211 ± 0.000 | 0.515 ± 0.000 | 0.078 ± 0.000 | 0.571 ± 0.000 | 0.237 ± 0.000 |

### Real graph: venues ranked highest by DOMINANT (cases for inspection, not accusations)

Not published (see the ethical considerations of the article).

## 6. Sub-topic 5: text vs structure (k = 3, random split)

| model | text | bibliometric | Spearman | R² | MAE | ROC-AUC | AP | precision@10% |
|---|---|---|---|---|---|---|---|---|
| gat | none | biblio | 0.449 ± 0.000 | 0.188 ± 0.000 | 0.825 ± 0.000 | 0.580 ± 0.000 | 0.277 ± 0.000 | 0.250 ± 0.000 |
| gat | none | nobiblio | 0.420 ± 0.000 | 0.174 ± 0.000 | 0.822 ± 0.000 | 0.696 ± 0.000 | 0.338 ± 0.000 | 0.333 ± 0.000 |
| gat | scibert | biblio | 0.330 ± 0.000 | -0.009 ± 0.000 | 0.906 ± 0.000 | 0.542 ± 0.000 | 0.336 ± 0.000 | 0.250 ± 0.000 |
| gat | scibert | nobiblio | 0.281 ± 0.000 | -0.001 ± 0.000 | 0.925 ± 0.000 | 0.563 ± 0.000 | 0.258 ± 0.000 | 0.250 ± 0.000 |
| gat | specter | biblio | 0.193 ± 0.000 | -0.076 ± 0.000 | 0.960 ± 0.000 | 0.530 ± 0.000 | 0.251 ± 0.000 | 0.250 ± 0.000 |
| gat | specter | nobiblio | 0.216 ± 0.000 | -0.091 ± 0.000 | 0.970 ± 0.000 | 0.475 ± 0.000 | 0.177 ± 0.000 | 0.167 ± 0.000 |
| mlp | none | biblio | 0.435 ± 0.000 | 0.136 ± 0.000 | 0.846 ± 0.000 | 0.611 ± 0.000 | 0.179 ± 0.000 | 0.167 ± 0.000 |
| mlp | none | nobiblio | 0.445 ± 0.000 | 0.215 ± 0.000 | 0.805 ± 0.000 | 0.731 ± 0.000 | 0.348 ± 0.000 | 0.250 ± 0.000 |
| mlp | scibert | biblio | 0.288 ± 0.000 | 0.033 ± 0.000 | 0.937 ± 0.000 | 0.582 ± 0.000 | 0.346 ± 0.000 | 0.250 ± 0.000 |
| mlp | scibert | nobiblio | 0.274 ± 0.000 | 0.003 ± 0.000 | 0.945 ± 0.000 | 0.572 ± 0.000 | 0.348 ± 0.000 | 0.250 ± 0.000 |
| mlp | specter | biblio | 0.218 ± 0.000 | -0.056 ± 0.000 | 0.949 ± 0.000 | 0.492 ± 0.000 | 0.329 ± 0.000 | 0.250 ± 0.000 |
| mlp | specter | nobiblio | 0.208 ± 0.000 | -0.090 ± 0.000 | 0.976 ± 0.000 | 0.527 ± 0.000 | 0.262 ± 0.000 | 0.250 ± 0.000 |
| sage | none | biblio | 0.407 ± 0.000 | 0.116 ± 0.000 | 0.863 ± 0.000 | 0.561 ± 0.000 | 0.142 ± 0.000 | 0.083 ± 0.000 |
| sage | none | nobiblio | 0.403 ± 0.000 | 0.173 ± 0.000 | 0.827 ± 0.000 | 0.666 ± 0.000 | 0.311 ± 0.000 | 0.250 ± 0.000 |
| sage | scibert | biblio | 0.284 ± 0.000 | -0.019 ± 0.000 | 0.931 ± 0.000 | 0.612 ± 0.000 | 0.356 ± 0.000 | 0.250 ± 0.000 |
| sage | scibert | nobiblio | 0.322 ± 0.000 | 0.096 ± 0.000 | 0.895 ± 0.000 | 0.628 ± 0.000 | 0.355 ± 0.000 | 0.250 ± 0.000 |
| sage | specter | biblio | 0.214 ± 0.000 | -0.053 ± 0.000 | 0.961 ± 0.000 | 0.472 ± 0.000 | 0.326 ± 0.000 | 0.250 ± 0.000 |
| sage | specter | nobiblio | 0.182 ± 0.000 | -0.146 ± 0.000 | 0.990 ± 0.000 | 0.513 ± 0.000 | 0.270 ± 0.000 | 0.250 ± 0.000 |

### Contributions (difference of mean scores)

| contrast | setting | bibliometric | Δ Spearman | Δ ROC-AUC |
|---|---|---|---|---|
| graph: sage − mlp | none | nobiblio | -0.041 | -0.064 |
| graph: gat − mlp | none | nobiblio | -0.025 | -0.035 |
| graph: sage − mlp | scibert | nobiblio | +0.048 | +0.056 |
| graph: gat − mlp | scibert | nobiblio | +0.007 | -0.009 |
| graph: sage − mlp | specter | nobiblio | -0.025 | -0.013 |
| graph: gat − mlp | specter | nobiblio | +0.009 | -0.052 |
| text: scibert − none | mlp | nobiblio | -0.171 | -0.159 |
| text: specter − none | mlp | nobiblio | -0.237 | -0.204 |
| text: scibert − none | sage | nobiblio | -0.082 | -0.038 |
| text: specter − none | sage | nobiblio | -0.221 | -0.153 |
| text: scibert − none | gat | nobiblio | -0.139 | -0.133 |
| text: specter − none | gat | nobiblio | -0.204 | -0.221 |
| graph: sage − mlp | none | biblio | -0.028 | -0.051 |
| graph: gat − mlp | none | biblio | +0.014 | -0.031 |
| graph: sage − mlp | scibert | biblio | -0.004 | +0.030 |
| graph: gat − mlp | scibert | biblio | +0.042 | -0.040 |
| graph: sage − mlp | specter | biblio | -0.004 | -0.019 |
| graph: gat − mlp | specter | biblio | -0.025 | +0.039 |
| text: scibert − none | mlp | biblio | -0.147 | -0.029 |
| text: specter − none | mlp | biblio | -0.217 | -0.120 |
| text: scibert − none | sage | biblio | -0.123 | +0.052 |
| text: specter − none | sage | biblio | -0.193 | -0.088 |
| text: scibert − none | gat | biblio | -0.119 | -0.038 |
| text: specter − none | gat | biblio | -0.256 | -0.050 |



## 7. Run time and environment

| exp | device | workers | runs | wall time (s) | mean run (s) | peak GPU MB / run |
|---|---|---|---|---|---|---|
| 1 | cuda | 2 | 12 | 9.3 | 0.4 | 117 |
| 2 | cuda | 2 | 22 | 9.5 | 0.3 | 102 |
| 3 | cuda | 2 | 14 | 10.8 | 0.6 | 452 |
| 4 | cuda | 2 | 3 | 18.4 | 6.1 | 48 |
| 5 | cuda | 2 | 36 | 11.9 | 0.3 | 110 |



Environment: python 3.12.7, platform Linux-5.15.0-198-generic-x86_64-with-glibc2.35, torch 2.14.1+cu130, torch_cuda 13.0, cuda_available True, torch_geometric 2.8.1, cpu_count 8, gpu NVIDIA RTX A4500, gpu_memory_gb 19.6, pyg_lib 0.9.0+pt214cu130, torch_scatter None, torch_sparse None, sentence_transformers 6.2.0, transformers 5.20.2, sklearn 1.9.1, pymupdf 2.0, numpy 2.5.4, pandas 3.1.0, scipy 1.18.1



## 8. Leakage control and residual leakage

Controls implemented in the code (asserted in `build_graph.py`, tested in `tests/`):
- citation edges go only from a reference to the citing paper, and edges whose reference is published after the
  citing paper are dropped; a paper never receives information from the papers that cite it;
- author, venue and reference statistics attached to a paper published in year y count only citations received
  before y (`total − Σ counts_by_year[t ≥ y]`); for years before 2012 (first covered year) the value as of 2011 is
  used, which is still earlier than every target (targets are published ≥ 2017);
- labels: citations in years y..y+k, papers kept only if y+k ≤ 2025; lab papers are never training labels;
- cold start: test papers (and every paper published from the test year on) are removed from the training graph;
- heterogeneous graph: author/venue/keyword nodes receive messages only from papers published before 2017;
  HAN metapath neighbours satisfy year(Q) < 2017 (T0 rule) or year(Q) < year(P) (temporal rule);
  the "HGT naive" row uses all reverse edges and is reported only to show how much leakage inflates scores.

Residual leakage (not removable with a 2026 snapshot):
- OpenAlex author disambiguation, topics and keywords are computed today, with information posterior to publication;
- SPECTER was trained on citation triplets of papers published up to about 2019–2020, so its embeddings may encode
  citation information of part of the 2017–2020 papers (SciBERT is trained on text only);
- the sample contains only works still indexed in 2026 (survivorship), with reference lists as currently recorded;
- the label window includes citations received in the publication year itself;
- reference-node selection uses the number of citers among sampled papers (a structural, label-free criterion);
- bf16 autocast is used for all models except HGT (pyg-lib `segment_matmul` does not accept bf16 inputs).


## 9. Exploratory (post hoc): text embeddings reduced to 64 PCA components

Decided after inspecting sub-topic 5 (the 768-dimensional text inputs appeared to overfit). PCA is fitted without labels on all paper nodes. Explained variance: specter 0.806, scibert 0.756. Main results above remain the reference.

### Sub-topic 1 with PCA-64 text

| k | model | Spearman (768-d) | Spearman (64-d) | R² (768-d) | R² (64-d) |
|---|---|---|---|---|---|
| 3 | MLP (no graph) | 0.218 ± 0.000 | 0.360 ± 0.000 | -0.056 ± 0.000 | 0.051 ± 0.000 |
| 3 | MLP + reference aggregates | 0.285 ± 0.000 | 0.371 ± 0.000 | 0.027 ± 0.000 | 0.083 ± 0.000 |
| 3 | GraphSAGE | 0.211 ± 0.000 | 0.386 ± 0.000 | -0.054 ± 0.000 | 0.124 ± 0.000 |
| 3 | GraphSAGE + edge features | 0.234 ± 0.000 | 0.364 ± 0.000 | -0.084 ± 0.000 | 0.116 ± 0.000 |
| 3 | GAT | 0.191 ± 0.000 | 0.394 ± 0.000 | -0.076 ± 0.000 | 0.092 ± 0.000 |
| 3 | GAT + edge features | 0.277 ± 0.000 | 0.368 ± 0.000 | 0.015 ± 0.000 | 0.084 ± 0.000 |
| 5 | MLP (no graph) | 0.165 ± 0.000 | 0.366 ± 0.000 | -0.341 ± 0.000 | 0.036 ± 0.000 |
| 5 | MLP + reference aggregates | 0.171 ± 0.000 | 0.303 ± 0.000 | -0.266 ± 0.000 | 0.091 ± 0.000 |
| 5 | GraphSAGE | 0.145 ± 0.000 | 0.344 ± 0.000 | -0.093 ± 0.000 | 0.071 ± 0.000 |
| 5 | GraphSAGE + edge features | 0.219 ± 0.000 | 0.332 ± 0.000 | -0.212 ± 0.000 | 0.039 ± 0.000 |
| 5 | GAT | 0.164 ± 0.000 | 0.414 ± 0.000 | -0.208 ± 0.000 | 0.135 ± 0.000 |
| 5 | GAT + edge features | 0.212 ± 0.000 | 0.375 ± 0.000 | -0.056 ± 0.000 | 0.089 ± 0.000 |

## 10. Implementation check on ogbn-arxiv (test accuracy %)

Reference values: Hu et al. 2020, Open Graph Benchmark, Table 3 (test accuracy %); our models keep their residual architecture (not a reproduction).

| model | ours (mean ± std) | seeds | published |
|---|---|---|---|
| mlp | 55.23 ± 0.14 | 3 | 55.50 ± 0.23 |
| sage | 69.98 ± 0.44 | 3 | 71.49 ± 0.27 |
| gat | 71.25 ± 0.20 | 3 | – |
