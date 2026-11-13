# Axis 1 – Citation graphs and GNNs: results summary (full run)

Generated 2026-10-25 17:33 by `summarize.py` from `data/` and `results/`.

## 1. Data

### 1.1 PDF matching (PyMuPDF + OpenAlex)

| corpus | distinct PDFs | matched | duplicate work | inherited match rejected | OpenAlex DOI printed in PDF | match method |
|---|---|---|---|---|---|---|
| lab | 37 | 35 | 0 | 0 | 28 | fbn_match: 32, doi_lookup: 3, unmatched: 2 |
| open | 1010 | 968 | 0 | 12 | 431 | manifest_openalex_id: 510, fbn_match: 448, unmatched: 42, doi_lookup: 9, title_search: 1 |


Lab folder: 40 files, 3 exact duplicates (sha256) removed.

### 1.2 OpenAlex collection

Calls = distinct API calls of the step (each sent once, then served from the disk cache). Credits are reconstructed from the remaining-credit values printed in `logs/collect_<profile>.log` (an upward jump means a new API key with a daily limit of 10,000 credits).

| step | calls | credits consumed | seconds (last run, from cache) |
|---|---|---|---|
| seeds | 65 | 370 | 15.2 |
| sample | 96 | 96 | 5.1 |
| refs | 7198 | 7197 | 119.1 |
| nodes | 940 | 947 | 16.1 |
| authors | 893 | 893 | 3.5 |
| venues | 550 | 548 | 3.4 |
| total | 9742 | 10051 |  |

| collected | count |
|---|---|
| sampled papers | 12000 |
| seed records | 1003 |
| unique references of sampled and seed papers (excluding themselves) | 359873 |
| reference records returned by OpenAlex | 352005 |
| author records | 44623 |
| venue records | 27469 |

### 1.3 Topics of the labeled sample

| topic | name | group | stratum | papers sampled |
|---|---|---|---|---|
| T10246 | Mobile Ad Hoc Networks | lab |  | 250 |
| T13373 | Data Mining and Machine Learning Applications | lab |  | 250 |
| T10102 | scientometrics and bibliometrics research | lab |  | 250 |
| T10270 | Blockchain Technology Applications and Security | lab |  | 250 |
| T12080 | Injection Molding Process and Properties | lab |  | 250 |
| T10050 | Multi-Criteria Decision Making | lab |  | 250 |
| T10761 | Vehicular Ad Hoc Networks (VANETs) | lab |  | 250 |
| T13904 | Artificial Intelligence Applications | lab |  | 250 |
| T14260 | Impact of AI and Big Data on Business and Society | lab |  | 250 |
| T11326 | Stock Market Forecasting Methods | lab |  | 250 |
| T11357 | Risk and Safety Analysis | lab |  | 250 |
| T10036 | Advanced Neural Network Applications | lab |  | 250 |
| T10215 | Semantic Web and Ontologies | lab |  | 250 |
| T10328 | Supply Chain and Inventory Management | lab |  | 250 |
| T10524 | Traffic control and management | lab |  | 250 |
| T10763 | Digital Transformation in Industry | lab |  | 250 |
| T10780 | Reliability and Maintenance Optimization | lab |  | 250 |
| T10876 | Fault Detection and Control Systems | lab |  | 250 |
| T11443 | Advanced Statistical Process Monitoring | lab |  | 250 |
| T11450 | Model-Driven Software Engineering Techniques | lab |  | 250 |
| T11498 | Security in Wireless Sensor Networks | lab |  | 250 |
| T11504 | Advanced Authentication Protocols Security | lab |  | 250 |
| T11734 | Decision Support System Applications | lab |  | 250 |
| T11814 | Advanced Manufacturing and Logistics Optimization | lab |  | 250 |
| T10845 | Advanced Causal Inference Techniques | open | econ | 250 |
| T10621 | Gene Regulatory Network Analysis | open | Biochemistry, Genetics and Molecular Biology | 250 |
| T10079 | Nanoparticles: synthesis and applications | open | Materials Science | 250 |
| T12336 | Insect Utilization and Effects | open | Agricultural and Biological Sciences | 250 |
| T10399 | Hydrocarbon exploration and reservoir analysis | open | Engineering | 250 |
| T10007 | Monetary Policy and Economic Impact | open | Economics, Econometrics and Finance | 250 |
| T10029 | Climate variability and models | open | Environmental Science | 250 |
| T10118 | SARS-CoV-2 and COVID-19 Research | open | Medicine | 250 |
| T14309 | Employee Performance and Management | open | Social Sciences | 250 |
| T10168 | COVID-19 and Mental Health | open | Psychology | 250 |
| T10021 | EFL/ESL Teaching and Learning | open | Arts and Humanities | 250 |
| T10136 | Statistical Methods and Inference | open | stat | 250 |
| T11714 | Multimodal Machine Learning Applications | open | cs | 250 |
| T10020 | Quantum Information and Cryptography | open | physics | 250 |
| T11270 | Complex Systems and Time Series Analysis | open | q-fin | 250 |
| T11458 | Advanced Wireless Communication Technologies | open | eess | 250 |
| T11152 | Stochastic processes and statistical mechanics | open | math | 250 |
| T10241 | Functional Brain Connectivity Studies | open | q-bio | 250 |
| T10282 | Financial Risk and Volatility Modeling | open | econ | 250 |
| T10066 | Gut microbiota and health | open | Biochemistry, Genetics and Molecular Biology | 250 |
| T10245 | Metamaterials and Metasurfaces Applications | open | Materials Science | 250 |
| T10004 | Soil Carbon and Nitrogen Dynamics | open | Agricultural and Biological Sciences | 250 |
| T10972 | Power Systems Fault Detection | open | Engineering | 250 |
| T10127 | Banking stability, regulation, efficiency | open | Economics, Econometrics and Finance | 250 |

### 1.4 Citation graph

| quantity | value |
|---|---|
| paper nodes | 59985 |
| nodes by origin | ref: 46982, sample: 12000, seed_open: 968, seed_lab: 35 |
| known works (with metadata) | 364863 |
| citation edges (reference -> citing) | 629575 |
| edges dropped (reference published after citing paper) | 8988 |
| share of nodes with an abstract | 0.663 |
| author records / venue records | 44623 / 27469 |
| sample by topic group | lab: 6000, open: 6000 |

### 1.5 Labels (citations in years y..y+k, complete windows only)

| k | labeled papers | by year | mean citations | median | share zero | top-10 % rate | in-graph refs mean | median | share with none |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 12336 | 2017: 1344, 2018: 1534, 2019: 1734, 2020: 2180, 2021: 2617, 2022: 2927 | 16.77 | 6.0 | 0.134 | 0.102 | 12.59 | 9.0 | 0.061 |
| 5 | 6792 | 2017: 1344, 2018: 1534, 2019: 1734, 2020: 2180 | 26.57 | 10.0 | 0.100 | 0.101 | 12.97 | 10.0 | 0.053 |

### 1.6 Heterogeneous graph and venue graph

| quantity | value |
|---|---|
| authors | 44623 |
| venues | 27469 |
| keywords | 52293 |
| edges_cites | 629575 |
| edges_fwd_author | 79722 |
| edges_rev_all_author | 79722 |
| edges_rev_safe_author | 19132 |
| edges_fwd_venue | 58151 |
| edges_rev_all_venue | 58151 |
| edges_rev_safe_venue | 34439 |
| edges_fwd_keyword | 402918 |
| edges_rev_all_keyword | 402918 |
| edges_rev_safe_keyword | 238328 |
| edges_mp_t0_author | 121326 |
| edges_mp_temporal_author | 100504 |
| edges_mp_t0_venue | 596677 |
| edges_mp_temporal_venue | 515003 |
| edges_mp_t0_keyword | 1117668 |
| edges_mp_temporal_keyword | 1015255 |
| venue graph venues | 3136 |
| venue graph edges | 96052 |
| venue graph links | 264901 |
| venue graph self_links | 22016 |



## 2. Sub-topic 1: future impact with GAT (random split 70/15/15, regression on log1p citations)

### k = 3

| model | Spearman | R² | MAE (log) | runs |
|---|---|---|---|---|
| MLP (no graph) | 0.684 ± 0.014 | 0.469 ± 0.022 | 0.735 ± 0.009 | 5 |
| MLP + reference aggregates | 0.693 ± 0.014 | 0.481 ± 0.019 | 0.726 ± 0.010 | 5 |
| GraphSAGE | 0.682 ± 0.014 | 0.464 ± 0.023 | 0.736 ± 0.014 | 5 |
| GraphSAGE + edge features | 0.682 ± 0.017 | 0.460 ± 0.031 | 0.741 ± 0.017 | 5 |
| GAT | 0.678 ± 0.014 | 0.453 ± 0.020 | 0.741 ± 0.011 | 5 |
| GAT + edge features | 0.676 ± 0.022 | 0.453 ± 0.031 | 0.740 ± 0.015 | 5 |

### k = 5

| model | Spearman | R² | MAE (log) | runs |
|---|---|---|---|---|
| MLP (no graph) | 0.682 ± 0.010 | 0.468 ± 0.020 | 0.793 ± 0.015 | 5 |
| MLP + reference aggregates | 0.695 ± 0.013 | 0.483 ± 0.025 | 0.782 ± 0.012 | 5 |
| GraphSAGE | 0.684 ± 0.010 | 0.467 ± 0.014 | 0.792 ± 0.015 | 5 |
| GraphSAGE + edge features | 0.679 ± 0.009 | 0.459 ± 0.019 | 0.796 ± 0.011 | 5 |
| GAT | 0.673 ± 0.016 | 0.446 ± 0.026 | 0.800 ± 0.015 | 5 |
| GAT + edge features | 0.677 ± 0.018 | 0.453 ± 0.026 | 0.797 ± 0.015 | 5 |

### Attention analysis (test papers with >= 3 in-graph references; self-loops excluded)

Per-paper = Spearman between attention and the reference attribute within each citing paper, averaged over papers; pooled = Spearman between degree-normalised attention and the attribute over all edges.

| model k | weights | per-paper ρ(att, ref citations) | share ρ>0 | pooled ρ(cit) | per-paper ρ(att, ref age) | share ρ>0 | pooled ρ(age) | papers |
|---|---|---|---|---|---|---|---|---|
| gat k3 | trained | -0.065 ± 0.095 | 0.421 ± 0.120 | -0.054 ± 0.084 | 0.016 ± 0.137 | 0.512 ± 0.161 | 0.019 ± 0.118 | 1516 ± 15 |
| gat k3 | untrained | -0.046 ± 0.045 | 0.435 ± 0.050 | -0.051 ± 0.028 | -0.015 ± 0.031 | 0.474 ± 0.038 | -0.036 ± 0.020 | 1516 ± 15 |
| gat k3 | trained_layer1 | -0.074 ± 0.050 | 0.402 ± 0.062 | -0.072 ± 0.047 | 0.035 ± 0.064 | 0.540 ± 0.081 | 0.033 ± 0.057 | 1516 ± 15 |
| gat k5 | trained | 0.083 ± 0.077 | 0.599 ± 0.089 | 0.095 ± 0.078 | 0.187 ± 0.068 | 0.701 ± 0.066 | 0.179 ± 0.067 | 843 ± 7 |
| gat k5 | untrained | -0.045 ± 0.048 | 0.435 ± 0.058 | -0.051 ± 0.030 | -0.012 ± 0.035 | 0.476 ± 0.044 | -0.042 ± 0.022 | 843 ± 7 |
| gat k5 | trained_layer1 | 0.041 ± 0.062 | 0.540 ± 0.079 | 0.033 ± 0.057 | 0.154 ± 0.078 | 0.667 ± 0.091 | 0.131 ± 0.074 | 843 ± 7 |
| gat_e k3 | trained | -0.355 ± 0.426 | 0.260 ± 0.331 | -0.280 ± 0.374 | -0.153 ± 0.385 | 0.381 ± 0.360 | -0.099 ± 0.340 | 1516 ± 15 |
| gat_e k3 | untrained | 0.193 ± 0.306 | 0.647 ± 0.226 | 0.158 ± 0.275 | 0.166 ± 0.246 | 0.633 ± 0.180 | 0.121 ± 0.220 | 1516 ± 15 |
| gat_e k3 | trained_layer1 | -0.610 ± 0.266 | 0.075 ± 0.120 | -0.507 ± 0.227 | -0.392 ± 0.256 | 0.173 ± 0.190 | -0.313 ± 0.221 | 1516 ± 15 |
| gat_e k5 | trained | -0.133 ± 0.347 | 0.396 ± 0.317 | -0.099 ± 0.304 | 0.001 ± 0.306 | 0.520 ± 0.305 | 0.027 ± 0.264 | 843 ± 7 |
| gat_e k5 | untrained | 0.197 ± 0.308 | 0.648 ± 0.219 | 0.162 ± 0.286 | 0.178 ± 0.262 | 0.639 ± 0.184 | 0.120 ± 0.241 | 843 ± 7 |
| gat_e k5 | trained_layer1 | -0.272 ± 0.323 | 0.276 ± 0.258 | -0.229 ± 0.275 | -0.102 ± 0.321 | 0.413 ± 0.311 | -0.080 ± 0.263 | 843 ± 7 |

- gat k3: mean self-loop attention on test papers = 0.201 ± 0.004

- gat k5: mean self-loop attention on test papers = 0.187 ± 0.006

- gat_e k3: mean self-loop attention on test papers = 0.188 ± 0.007

- gat_e k5: mean self-loop attention on test papers = 0.176 ± 0.006

### Examples (GAT, attention averaged over seeds; highest-impact labeled papers)

- **Machine Learning: Algorithms, Real-World Applications and Research Directions** (2021, SN Computer Science; 3031 citations, 85 references in graph, uniform attention = 0.012)

    - top: Manufacturing process data analysis pipelines: a requirements analysis and survey (2019; 25 citations before citing year; attention 0.037)

    - top: Towards the development of realistic botnet dataset in the Internet of Things for network forensic analytics: Bot-IoT dataset (2019; 124 citations before citing year; attention 0.030)

    - top: Achieving sustainable performance in a data-driven agriculture supply chain: A review for research and applications (2019; 97 citations before citing year; attention 0.028)

    - bottom: Data Mining: Practical Machine Learning Tools and Techniques (2011; 22218 citations before citing year; attention 0.007)

    - bottom: Top 10 algorithms in data mining (2007; 3701 citations before citing year; attention 0.007)

    - bottom: Data mining: concepts and techniques (2012; 24160 citations before citing year; attention 0.007)

- **Machine learning and deep learning** (2021, Electronic Markets; 1634 citations, 24 references in graph, uniform attention = 0.042)

    - top: A strategy to apply machine learning to small datasets in materials science (2018; 129 citations before citing year; attention 0.075)

    - top: Robust Physical-World Attacks on Deep Learning Visual Classification (2018; 471 citations before citing year; attention 0.069)

    - top: Deep Learning (2016; 6271 citations before citing year; attention 0.054)

    - bottom: Machine learning: a review of classification and combining techniques (2006; 645 citations before citing year; attention 0.022)

    - bottom: A Survey on Deep Learning (2018; 224 citations before citing year; attention 0.027)

    - bottom: Term-weighting approaches in automatic text retrieval (1988; 7523 citations before citing year; attention 0.029)

- **Characterization of the receptor-binding domain (RBD) of 2019 novel coronavirus: implication for development of RBD protein as a viral attachment inhibitor and vaccine** (2020, Cellular and Molecular Immunology; 1437 citations, 17 references in graph, uniform attention = 0.059)

    - top: Theoretical Basis, Experimental Design, and Computerized Simulation of Synergism and Antagonism in Drug Combination Studies (2006; 2708 citations before citing year; attention 0.081)

    - top: Isolation of a Novel Coronavirus from a Man with Pneumonia in Saudi Arabia (2012; 1562 citations before citing year; attention 0.071)

    - top: Epidemiology and cause of severe acute respiratory syndrome (SARS) in Guangdong, People's Republic of China, in February, 2003 (2003; 282 citations before citing year; attention 0.069)

    - bottom: Receptor usage and cell entry of bat coronavirus HKU4 provide insight into bat-to-human transmission of MERS coronavirus (2014; 111 citations before citing year; attention 0.042)

    - bottom: MERS-CoV spike protein: a key target for antivirals (2016; 40 citations before citing year; attention 0.046)

    - bottom: Structure of SARS Coronavirus Spike Receptor-Binding Domain Complexed with Receptor (2005; 329 citations before citing year; attention 0.048)

- **Investigating Mental Health of US College Students During the COVID-19 Pandemic: Cross-Sectional Survey Study** (2020, Journal of Medical Internet Research; 1011 citations, 11 references in graph, uniform attention = 0.091)

    - top: Iranian mental health during the COVID-19 epidemic (2020; 1 citations before citing year; attention 0.102)

    - top: Online mental health services in China during the COVID-19 outbreak (2020; 6 citations before citing year; attention 0.096)

    - top: Multidisciplinary research priorities for the COVID-19 pandemic: a call for action for mental health science (2020; 21 citations before citing year; attention 0.093)

    - bottom: The PHQ-9: A New Depression Diagnostic and Severity Measure (2002; 2237 citations before citing year; attention 0.072)

    - bottom: Affective temperament, attachment style, and the psychological impact of the COVID-19 outbreak: an early report on the Italian general population (2020; 0 citations before citing year; attention 0.074)

    - bottom: College life is stressful today – Emerging stressors and depressive symptoms in college students (2018; 11 citations before citing year; attention 0.075)

- **Knowledge Graphs** (2021, ACM Computing Surveys; 950 citations, 147 references in graph, uniform attention = 0.007)

    - top: Virtuoso, a Hybrid RDBMS/Graph Column Store. (2012; 85 citations before citing year; attention 0.014)

    - top: Nanopublication beyond the sciences: the PeriodO period gazetteer (2016; 18 citations before citing year; attention 0.014)

    - top: Canonical Forms for Isomorphic and Equivalent RDF Graphs (2017; 8 citations before citing year; attention 0.012)

    - bottom: deqa: Deep Web Extraction for Question Answering (2012; 38 citations before citing year; attention 0.004)

    - bottom: Rule Learning from Knowledge Graphs Guided by Embedding Models (2018; 43 citations before citing year; attention 0.004)

    - bottom: Web-scale information extraction in knowitall (2004; 696 citations before citing year; attention 0.004)



## 3. Sub-topic 2: inductive cold start with GraphSAGE (temporal split)

### k = 3: train < 2020, validation = 2020, test >= 2021 (training graph: 53113 nodes, 551360 edges; test papers absent)

| variant | Spearman | R² | MAE (log) | runs |
|---|---|---|---|---|
| mlp_authors | 0.597 ± 0.003 | 0.351 ± 0.007 | 0.815 ± 0.006 | 5 |
| mlp_refagg | 0.590 ± 0.003 | 0.309 ± 0.009 | 0.839 ± 0.005 | 5 |
| mlp_authors_refagg | 0.644 ± 0.001 | 0.386 ± 0.011 | 0.786 ± 0.007 | 5 |
| sage_refs | 0.538 ± 0.002 | 0.199 ± 0.016 | 0.891 ± 0.007 | 5 |
| sage_refs_authors | 0.596 ± 0.005 | 0.303 ± 0.017 | 0.838 ± 0.009 | 5 |
| sage_refs_authors_text | 0.575 ± 0.005 | 0.300 ± 0.006 | 0.849 ± 0.004 | 5 |
| sage_full | 0.638 ± 0.005 | 0.378 ± 0.005 | 0.797 ± 0.002 | 5 |
| sage_plain_full | 0.635 ± 0.006 | 0.381 ± 0.012 | 0.796 ± 0.008 | 5 |
| mlp_full | 0.647 ± 0.004 | 0.404 ± 0.007 | 0.780 ± 0.005 | 5 |

### k = 5: train < 2019, validation = 2019, test >= 2020 (training graph: 49673 nodes, 499564 edges; test papers absent)

| variant | Spearman | R² | MAE (log) | runs |
|---|---|---|---|---|
| mlp_authors | 0.610 ± 0.002 | 0.364 ± 0.004 | 0.895 ± 0.003 | 5 |
| mlp_refagg | 0.592 ± 0.002 | 0.340 ± 0.002 | 0.912 ± 0.002 | 5 |
| mlp_authors_refagg | 0.655 ± 0.002 | 0.417 ± 0.003 | 0.859 ± 0.003 | 5 |
| sage_refs | 0.559 ± 0.007 | 0.274 ± 0.010 | 0.943 ± 0.008 | 5 |
| sage_refs_authors | 0.603 ± 0.006 | 0.342 ± 0.008 | 0.910 ± 0.007 | 5 |
| sage_refs_authors_text | 0.590 ± 0.003 | 0.328 ± 0.007 | 0.922 ± 0.006 | 5 |
| sage_full | 0.651 ± 0.003 | 0.397 ± 0.003 | 0.869 ± 0.003 | 5 |
| sage_plain_full | 0.657 ± 0.005 | 0.413 ± 0.009 | 0.859 ± 0.007 | 5 |
| mlp_full | 0.669 ± 0.004 | 0.428 ± 0.009 | 0.848 ± 0.005 | 5 |

### Lab papers (deployment models trained on all labeled years; mean over seeds)

Not published (see the ethical considerations of the article).

## 4. Sub-topic 3: heterogeneous graph, top-10 % detection (random split)

### k = 3

| model | ROC-AUC | AP | precision@10% | runs |
|---|---|---|---|---|
| MLP (no graph) | 0.816 ± 0.022 | 0.329 ± 0.043 | 0.357 ± 0.047 | 5 |
| GraphSAGE | 0.815 ± 0.014 | 0.319 ± 0.033 | 0.352 ± 0.033 | 5 |
| GAT | 0.817 ± 0.015 | 0.326 ± 0.035 | 0.361 ± 0.046 | 5 |
| HAN (T0 rule) | 0.833 ± 0.016 | 0.367 ± 0.046 | 0.403 ± 0.039 | 5 |
| HAN (temporal rule) | 0.830 ± 0.013 | 0.359 ± 0.028 | 0.385 ± 0.020 | 5 |
| HGT (T0 rule) | 0.841 ± 0.019 | 0.383 ± 0.050 | 0.415 ± 0.034 | 5 |
| HGT naive (leaky, reference only) | 0.839 ± 0.020 | 0.385 ± 0.051 | 0.413 ± 0.039 | 5 |

### k = 5

| model | ROC-AUC | AP | precision@10% | runs |
|---|---|---|---|---|
| MLP (no graph) | 0.804 ± 0.028 | 0.321 ± 0.055 | 0.335 ± 0.050 | 5 |
| GraphSAGE | 0.813 ± 0.016 | 0.321 ± 0.052 | 0.353 ± 0.045 | 5 |
| GAT | 0.807 ± 0.030 | 0.318 ± 0.043 | 0.318 ± 0.033 | 5 |
| HAN (T0 rule) | 0.826 ± 0.026 | 0.355 ± 0.056 | 0.355 ± 0.060 | 5 |
| HAN (temporal rule) | 0.833 ± 0.019 | 0.375 ± 0.035 | 0.400 ± 0.037 | 5 |
| HGT (T0 rule) | 0.846 ± 0.018 | 0.406 ± 0.046 | 0.416 ± 0.061 | 5 |
| HGT naive (leaky, reference only) | 0.845 ± 0.018 | 0.405 ± 0.049 | 0.412 ± 0.064 | 5 |



## 5. Sub-topic 4: citation cartels and self-citation (venue graph, injected anomalies)

Venue graph: 3136 venues, 96052 weighted edges, 264901 citation links. Injection per draw: 5 cartels of 4 venues, 10 self-citing venues; intensity = share of the venue's outgoing citations added; 3 draws.

### intensity mild (0.1)

| detector | cartel AUC | cartel AP | self AUC | self AP | all AUC | all AP |
|---|---|---|---|---|---|---|
| dominant_content | 0.469 ± 0.083 | 0.007 ± 0.001 | 0.502 ± 0.073 | 0.005 ± 0.002 | 0.480 ± 0.049 | 0.010 ± 0.002 |
| dominant_full | 0.484 ± 0.065 | 0.007 ± 0.001 | 0.508 ± 0.090 | 0.005 ± 0.001 | 0.492 ± 0.061 | 0.010 ± 0.002 |
| gae_content | 0.391 ± 0.035 | 0.005 ± 0.000 | 0.457 ± 0.031 | 0.004 ± 0.001 | 0.413 ± 0.033 | 0.008 ± 0.001 |
| gae_full | 0.454 ± 0.049 | 0.006 ± 0.001 | 0.547 ± 0.080 | 0.005 ± 0.001 | 0.485 ± 0.058 | 0.010 ± 0.002 |
| heur_max | 0.737 ± 0.023 | 0.013 ± 0.001 | 0.849 ± 0.023 | 0.012 ± 0.003 | 0.774 ± 0.008 | 0.022 ± 0.001 |
| heur_reciprocity | 0.831 ± 0.018 | 0.020 ± 0.003 | 0.480 ± 0.057 | 0.005 ± 0.002 | 0.714 ± 0.010 | 0.021 ± 0.000 |
| heur_self_ratio | 0.546 ± 0.065 | 0.007 ± 0.001 | 0.910 ± 0.016 | 0.020 ± 0.006 | 0.668 ± 0.049 | 0.019 ± 0.005 |
| iforest_content | 0.465 ± 0.095 | 0.006 ± 0.001 | 0.481 ± 0.065 | 0.005 ± 0.001 | 0.470 ± 0.065 | 0.010 ± 0.002 |
| iforest_full | 0.509 ± 0.096 | 0.008 ± 0.002 | 0.523 ± 0.102 | 0.006 ± 0.004 | 0.513 ± 0.066 | 0.011 ± 0.003 |

### intensity medium (0.25)

| detector | cartel AUC | cartel AP | self AUC | self AP | all AUC | all AP |
|---|---|---|---|---|---|---|
| dominant_content | 0.481 ± 0.085 | 0.007 ± 0.002 | 0.502 ± 0.069 | 0.005 ± 0.002 | 0.488 ± 0.050 | 0.011 ± 0.002 |
| dominant_full | 0.542 ± 0.059 | 0.010 ± 0.003 | 0.628 ± 0.082 | 0.006 ± 0.002 | 0.571 ± 0.061 | 0.014 ± 0.003 |
| gae_content | 0.401 ± 0.032 | 0.005 ± 0.000 | 0.468 ± 0.038 | 0.004 ± 0.001 | 0.424 ± 0.034 | 0.008 ± 0.001 |
| gae_full | 0.457 ± 0.077 | 0.006 ± 0.001 | 0.519 ± 0.037 | 0.004 ± 0.000 | 0.478 ± 0.061 | 0.010 ± 0.002 |
| heur_max | 0.834 ± 0.013 | 0.020 ± 0.002 | 0.961 ± 0.006 | 0.041 ± 0.006 | 0.876 ± 0.009 | 0.044 ± 0.004 |
| heur_reciprocity | 0.901 ± 0.009 | 0.034 ± 0.004 | 0.480 ± 0.057 | 0.005 ± 0.002 | 0.761 ± 0.016 | 0.031 ± 0.002 |
| heur_self_ratio | 0.529 ± 0.062 | 0.007 ± 0.001 | 0.980 ± 0.003 | 0.077 ± 0.013 | 0.679 ± 0.042 | 0.037 ± 0.007 |
| iforest_content | 0.465 ± 0.095 | 0.006 ± 0.001 | 0.481 ± 0.065 | 0.005 ± 0.001 | 0.470 ± 0.065 | 0.010 ± 0.002 |
| iforest_full | 0.567 ± 0.086 | 0.009 ± 0.003 | 0.606 ± 0.084 | 0.007 ± 0.004 | 0.580 ± 0.059 | 0.014 ± 0.003 |

### intensity strong (0.5)

| detector | cartel AUC | cartel AP | self AUC | self AP | all AUC | all AP |
|---|---|---|---|---|---|---|
| dominant_content | 0.488 ± 0.085 | 0.007 ± 0.002 | 0.500 ± 0.069 | 0.005 ± 0.002 | 0.492 ± 0.051 | 0.011 ± 0.002 |
| dominant_full | 0.618 ± 0.051 | 0.015 ± 0.004 | 0.803 ± 0.034 | 0.012 ± 0.004 | 0.679 ± 0.044 | 0.022 ± 0.005 |
| gae_content | 0.407 ± 0.042 | 0.005 ± 0.000 | 0.473 ± 0.044 | 0.004 ± 0.001 | 0.429 ± 0.043 | 0.008 ± 0.001 |
| gae_full | 0.461 ± 0.075 | 0.006 ± 0.002 | 0.484 ± 0.034 | 0.004 ± 0.001 | 0.469 ± 0.058 | 0.009 ± 0.001 |
| heur_max | 0.889 ± 0.012 | 0.032 ± 0.002 | 0.986 ± 0.001 | 0.108 ± 0.010 | 0.921 ± 0.008 | 0.088 ± 0.006 |
| heur_reciprocity | 0.936 ± 0.009 | 0.056 ± 0.005 | 0.480 ± 0.057 | 0.005 ± 0.002 | 0.784 ± 0.016 | 0.047 ± 0.002 |
| heur_self_ratio | 0.505 ± 0.059 | 0.006 ± 0.001 | 0.994 ± 0.001 | 0.206 ± 0.014 | 0.668 ± 0.040 | 0.079 ± 0.005 |
| iforest_content | 0.465 ± 0.095 | 0.006 ± 0.001 | 0.481 ± 0.065 | 0.005 ± 0.001 | 0.470 ± 0.065 | 0.010 ± 0.002 |
| iforest_full | 0.636 ± 0.082 | 0.012 ± 0.003 | 0.695 ± 0.061 | 0.011 ± 0.008 | 0.656 ± 0.054 | 0.018 ± 0.006 |

### Real graph: Spearman correlation of each detector's score with venue size

| detector | ρ(score, log out-links) | ρ(score, log in-links) |
|---|---|---|
| heur_self_ratio | 0.223 | 0.507 |
| heur_reciprocity | 0.419 | 0.801 |
| heur_max | 0.224 | 0.711 |
| dominant_content | -0.376 | -0.656 |
| gae_content | -0.610 | -0.661 |
| iforest_content | -0.251 | -0.453 |
| dominant_full | -0.390 | -0.577 |
| gae_full | -0.408 | -0.413 |
| iforest_full | -0.134 | -0.271 |

### Real graph: venues ranked highest by DOMINANT (cases for inspection, not accusations)

Not published (see the ethical considerations of the article).

## 6. Sub-topic 5: text vs structure (k = 3, random split)

| model | text | bibliometric | Spearman | R² | MAE | ROC-AUC | AP | precision@10% |
|---|---|---|---|---|---|---|---|---|
| gat | none | biblio | 0.712 ± 0.014 | 0.512 ± 0.018 | 0.699 ± 0.012 | 0.860 ± 0.017 | 0.447 ± 0.066 | 0.452 ± 0.034 |
| gat | none | nobiblio | 0.593 ± 0.019 | 0.349 ± 0.023 | 0.811 ± 0.010 | 0.799 ± 0.023 | 0.354 ± 0.071 | 0.371 ± 0.060 |
| gat | scibert | biblio | 0.688 ± 0.009 | 0.470 ± 0.017 | 0.731 ± 0.009 | 0.834 ± 0.020 | 0.377 ± 0.057 | 0.394 ± 0.061 |
| gat | scibert | nobiblio | 0.586 ± 0.016 | 0.339 ± 0.019 | 0.817 ± 0.011 | 0.805 ± 0.016 | 0.327 ± 0.044 | 0.349 ± 0.046 |
| gat | specter | biblio | 0.679 ± 0.015 | 0.456 ± 0.019 | 0.738 ± 0.012 | 0.818 ± 0.016 | 0.328 ± 0.039 | 0.361 ± 0.042 |
| gat | specter | nobiblio | 0.558 ± 0.021 | 0.308 ± 0.024 | 0.843 ± 0.010 | 0.797 ± 0.022 | 0.299 ± 0.026 | 0.337 ± 0.031 |
| mlp | none | biblio | 0.664 ± 0.016 | 0.444 ± 0.024 | 0.744 ± 0.015 | 0.820 ± 0.013 | 0.357 ± 0.047 | 0.370 ± 0.029 |
| mlp | none | nobiblio | 0.508 ± 0.023 | 0.255 ± 0.025 | 0.869 ± 0.004 | 0.724 ± 0.027 | 0.260 ± 0.053 | 0.280 ± 0.037 |
| mlp | scibert | biblio | 0.691 ± 0.010 | 0.478 ± 0.013 | 0.727 ± 0.004 | 0.820 ± 0.025 | 0.354 ± 0.057 | 0.374 ± 0.063 |
| mlp | scibert | nobiblio | 0.591 ± 0.018 | 0.339 ± 0.025 | 0.817 ± 0.014 | 0.783 ± 0.021 | 0.289 ± 0.040 | 0.318 ± 0.046 |
| mlp | specter | biblio | 0.684 ± 0.014 | 0.469 ± 0.022 | 0.735 ± 0.009 | 0.816 ± 0.022 | 0.329 ± 0.043 | 0.357 ± 0.047 |
| mlp | specter | nobiblio | 0.564 ± 0.019 | 0.314 ± 0.028 | 0.841 ± 0.015 | 0.781 ± 0.023 | 0.274 ± 0.036 | 0.303 ± 0.039 |
| sage | none | biblio | 0.708 ± 0.014 | 0.504 ± 0.019 | 0.705 ± 0.012 | 0.856 ± 0.009 | 0.444 ± 0.041 | 0.428 ± 0.030 |
| sage | none | nobiblio | 0.600 ± 0.022 | 0.358 ± 0.031 | 0.803 ± 0.013 | 0.804 ± 0.018 | 0.354 ± 0.069 | 0.366 ± 0.067 |
| sage | scibert | biblio | 0.696 ± 0.010 | 0.484 ± 0.016 | 0.719 ± 0.009 | 0.831 ± 0.021 | 0.371 ± 0.054 | 0.385 ± 0.052 |
| sage | scibert | nobiblio | 0.593 ± 0.020 | 0.345 ± 0.027 | 0.812 ± 0.017 | 0.797 ± 0.024 | 0.301 ± 0.049 | 0.324 ± 0.053 |
| sage | specter | biblio | 0.683 ± 0.015 | 0.466 ± 0.024 | 0.735 ± 0.016 | 0.816 ± 0.014 | 0.320 ± 0.033 | 0.355 ± 0.034 |
| sage | specter | nobiblio | 0.561 ± 0.023 | 0.303 ± 0.032 | 0.844 ± 0.016 | 0.789 ± 0.023 | 0.289 ± 0.035 | 0.328 ± 0.037 |

### Contributions (difference of mean scores)

| contrast | setting | bibliometric | Δ Spearman | Δ ROC-AUC |
|---|---|---|---|---|
| graph: sage − mlp | none | nobiblio | +0.093 | +0.080 |
| graph: gat − mlp | none | nobiblio | +0.085 | +0.075 |
| graph: sage − mlp | scibert | nobiblio | +0.002 | +0.014 |
| graph: gat − mlp | scibert | nobiblio | -0.005 | +0.021 |
| graph: sage − mlp | specter | nobiblio | -0.003 | +0.009 |
| graph: gat − mlp | specter | nobiblio | -0.007 | +0.016 |
| text: scibert − none | mlp | nobiblio | +0.083 | +0.059 |
| text: specter − none | mlp | nobiblio | +0.057 | +0.057 |
| text: scibert − none | sage | nobiblio | -0.008 | -0.007 |
| text: specter − none | sage | nobiblio | -0.040 | -0.015 |
| text: scibert − none | gat | nobiblio | -0.007 | +0.006 |
| text: specter − none | gat | nobiblio | -0.035 | -0.002 |
| graph: sage − mlp | none | biblio | +0.044 | +0.036 |
| graph: gat − mlp | none | biblio | +0.048 | +0.040 |
| graph: sage − mlp | scibert | biblio | +0.005 | +0.011 |
| graph: gat − mlp | scibert | biblio | -0.003 | +0.014 |
| graph: sage − mlp | specter | biblio | -0.001 | +0.000 |
| graph: gat − mlp | specter | biblio | -0.004 | +0.002 |
| text: scibert − none | mlp | biblio | +0.027 | -0.001 |
| text: specter − none | mlp | biblio | +0.020 | -0.004 |
| text: scibert − none | sage | biblio | -0.013 | -0.026 |
| text: specter − none | sage | biblio | -0.025 | -0.040 |
| text: scibert − none | gat | biblio | -0.024 | -0.027 |
| text: specter − none | gat | biblio | -0.033 | -0.042 |



## 7. Run time and environment

| exp | device | workers | runs | wall time (s) | mean run (s) | peak GPU MB / run |
|---|---|---|---|---|---|---|
| 1 | cuda | 3 | 60 | 306.2 | 12.7 | 3041 |
| 2 | cuda | 3 | 110 | 272.7 | 7.2 | 2290 |
| 3 | cuda | 1 | 70 | 510.9 | 7.3 | 10353 |
| 4 | cuda | 3 | 9 | 59.5 | 11.0 | 904 |
| 5 | cuda | 3 | 180 | 731.7 | 12.0 | 2700 |



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

Decided after inspecting sub-topic 5 (the 768-dimensional text inputs appeared to overfit). PCA is fitted without labels on all paper nodes. Explained variance: specter 0.804, scibert 0.735. Main results above remain the reference.

### Sub-topic 1 with PCA-64 text

| k | model | Spearman (768-d) | Spearman (64-d) | R² (768-d) | R² (64-d) |
|---|---|---|---|---|---|
| 3 | MLP (no graph) | 0.684 ± 0.014 | 0.704 ± 0.013 | 0.469 ± 0.022 | 0.502 ± 0.019 |
| 3 | MLP + reference aggregates | 0.693 ± 0.014 | 0.717 ± 0.015 | 0.481 ± 0.019 | 0.518 ± 0.021 |
| 3 | GraphSAGE | 0.682 ± 0.014 | 0.708 ± 0.013 | 0.464 ± 0.023 | 0.504 ± 0.020 |
| 3 | GraphSAGE + edge features | 0.682 ± 0.017 | 0.710 ± 0.016 | 0.460 ± 0.031 | 0.506 ± 0.023 |
| 3 | GAT | 0.678 ± 0.014 | 0.712 ± 0.014 | 0.453 ± 0.020 | 0.508 ± 0.023 |
| 3 | GAT + edge features | 0.676 ± 0.022 | 0.711 ± 0.016 | 0.453 ± 0.031 | 0.505 ± 0.025 |
| 5 | MLP (no graph) | 0.682 ± 0.010 | 0.718 ± 0.013 | 0.468 ± 0.020 | 0.516 ± 0.026 |
| 5 | MLP + reference aggregates | 0.695 ± 0.013 | 0.726 ± 0.011 | 0.483 ± 0.025 | 0.529 ± 0.022 |
| 5 | GraphSAGE | 0.684 ± 0.010 | 0.718 ± 0.008 | 0.467 ± 0.014 | 0.517 ± 0.021 |
| 5 | GraphSAGE + edge features | 0.679 ± 0.009 | 0.721 ± 0.011 | 0.459 ± 0.019 | 0.521 ± 0.023 |
| 5 | GAT | 0.673 ± 0.016 | 0.720 ± 0.012 | 0.446 ± 0.026 | 0.520 ± 0.023 |
| 5 | GAT + edge features | 0.677 ± 0.018 | 0.721 ± 0.007 | 0.453 ± 0.026 | 0.519 ± 0.021 |

### Sub-topic 3 with PCA-64 text

| k | model | ROC-AUC (768-d) | ROC-AUC (64-d) | AP (768-d) | AP (64-d) |
|---|---|---|---|---|---|
| 3 | MLP (no graph) | 0.816 ± 0.022 | 0.861 ± 0.018 | 0.329 ± 0.043 | 0.432 ± 0.047 |
| 3 | GraphSAGE | 0.815 ± 0.014 | 0.858 ± 0.017 | 0.319 ± 0.033 | 0.417 ± 0.047 |
| 3 | GAT | 0.817 ± 0.015 | 0.851 ± 0.021 | 0.326 ± 0.035 | 0.405 ± 0.051 |
| 3 | HAN (T0 rule) | 0.833 ± 0.016 | 0.863 ± 0.013 | 0.367 ± 0.046 | 0.428 ± 0.041 |
| 3 | HAN (temporal rule) | 0.830 ± 0.013 | 0.860 ± 0.014 | 0.359 ± 0.028 | 0.424 ± 0.034 |
| 3 | HGT (T0 rule) | 0.841 ± 0.019 | 0.862 ± 0.016 | 0.383 ± 0.050 | 0.443 ± 0.064 |
| 3 | HGT naive (leaky, reference only) | 0.839 ± 0.020 | 0.860 ± 0.017 | 0.385 ± 0.051 | 0.436 ± 0.062 |
| 5 | MLP (no graph) | 0.804 ± 0.028 | 0.851 ± 0.021 | 0.321 ± 0.055 | 0.403 ± 0.098 |
| 5 | GraphSAGE | 0.813 ± 0.016 | 0.851 ± 0.020 | 0.321 ± 0.052 | 0.399 ± 0.080 |
| 5 | GAT | 0.807 ± 0.030 | 0.826 ± 0.019 | 0.318 ± 0.043 | 0.338 ± 0.073 |
| 5 | HAN (T0 rule) | 0.826 ± 0.026 | 0.857 ± 0.021 | 0.355 ± 0.056 | 0.414 ± 0.070 |
| 5 | HAN (temporal rule) | 0.833 ± 0.019 | 0.850 ± 0.024 | 0.375 ± 0.035 | 0.411 ± 0.070 |
| 5 | HGT (T0 rule) | 0.846 ± 0.018 | 0.853 ± 0.020 | 0.406 ± 0.046 | 0.427 ± 0.068 |
| 5 | HGT naive (leaky, reference only) | 0.845 ± 0.018 | 0.853 ± 0.020 | 0.405 ± 0.049 | 0.427 ± 0.067 |

### Sub-topic 5 with PCA-64 text (runs without text are identical to the main grid)

| model | text | bibliometric | Spearman | R² | ROC-AUC | AP |
|---|---|---|---|---|---|---|
| gat | scibert | biblio | 0.718 ± 0.012 | 0.517 ± 0.018 | 0.854 ± 0.023 | 0.433 ± 0.070 |
| gat | scibert | nobiblio | 0.615 ± 0.014 | 0.379 ± 0.017 | 0.803 ± 0.016 | 0.333 ± 0.045 |
| gat | specter | biblio | 0.711 ± 0.014 | 0.508 ± 0.022 | 0.851 ± 0.021 | 0.405 ± 0.051 |
| gat | specter | nobiblio | 0.597 ± 0.018 | 0.356 ± 0.021 | 0.798 ± 0.020 | 0.298 ± 0.041 |
| mlp | scibert | biblio | 0.705 ± 0.013 | 0.499 ± 0.018 | 0.854 ± 0.023 | 0.438 ± 0.054 |
| mlp | scibert | nobiblio | 0.604 ± 0.017 | 0.360 ± 0.025 | 0.805 ± 0.024 | 0.334 ± 0.064 |
| mlp | specter | biblio | 0.704 ± 0.013 | 0.502 ± 0.019 | 0.861 ± 0.018 | 0.432 ± 0.047 |
| mlp | specter | nobiblio | 0.591 ± 0.015 | 0.349 ± 0.025 | 0.802 ± 0.022 | 0.322 ± 0.056 |
| sage | scibert | biblio | 0.713 ± 0.010 | 0.509 ± 0.015 | 0.857 ± 0.019 | 0.435 ± 0.060 |
| sage | scibert | nobiblio | 0.615 ± 0.016 | 0.376 ± 0.024 | 0.811 ± 0.021 | 0.335 ± 0.059 |
| sage | specter | biblio | 0.708 ± 0.013 | 0.504 ± 0.020 | 0.858 ± 0.017 | 0.417 ± 0.048 |
| sage | specter | nobiblio | 0.597 ± 0.019 | 0.354 ± 0.025 | 0.802 ± 0.020 | 0.305 ± 0.051 |

### Graph contribution with PCA-64 text (difference of mean scores)

| contrast | text | bibliometric | Δ Spearman | Δ ROC-AUC |
|---|---|---|---|---|
| graph: sage − mlp | scibert | nobiblio | +0.012 | +0.006 |
| graph: gat − mlp | scibert | nobiblio | +0.012 | -0.001 |
| graph: sage − mlp | specter | nobiblio | +0.006 | -0.000 |
| graph: gat − mlp | specter | nobiblio | +0.006 | -0.005 |
| graph: sage − mlp | scibert | biblio | +0.008 | +0.002 |
| graph: gat − mlp | scibert | biblio | +0.013 | -0.001 |
| graph: sage − mlp | specter | biblio | +0.004 | -0.003 |
| graph: gat − mlp | specter | biblio | +0.007 | -0.010 |

## 10. Implementation check on ogbn-arxiv (test accuracy %)

Reference values: Hu et al. 2020, Open Graph Benchmark, arXiv:2005.00687v7, Table 6 (test accuracy %); our models keep their residual architecture (not a reproduction).

| model | ours (mean ± std) | seeds | published |
|---|---|---|---|
| mlp | 55.23 ± 0.14 | 3 | 55.50 ± 0.23 |
| sage | 69.98 ± 0.44 | 3 | 71.49 ± 0.27 |
| gat | 71.25 ± 0.20 | 3 | – |
