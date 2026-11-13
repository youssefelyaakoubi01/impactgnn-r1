"""Job grids of the revision. Each grid is a list of jobs for ``experiments_rev1.py``; an *arm* is one configuration
(model, features, text, graph, hyperparameters) and is run on every data partition and initialisation seed of its
protocol.

Protocols and repetitions
- ``mature``  (primary; k = 3, prediction date 2022, cutoff-restricted graph): 5 validation partitions x 3 seeds.
- ``kfold``   (retrospective, transductive): 2 repetitions x 6 folds x 2 seeds.
- ``retro``   (temporal split of the submitted article): the split is fixed, 5 seeds.

Hyperparameters: ``tuned`` reads ``results/rev1/hpo/best.json`` (one configuration per model family, task and
protocol family, selected on validation data by ``hpo.py``); ``default`` is the configuration of the submitted
article.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from config import ROOT, Config

FEATS = ["base", "author", "venue"]
PRIMARY_TEXT = "scibert"
TEXTS = ["none", "tfidf", "scibert", "specter"]
MATURE = dict(protocol="mature", T=2022, k=3, graph="cut2022sym")
KFOLD = dict(protocol="kfold", graph="submitted")
HPO_FILE = ROOT / "results" / "rev1" / "hpo" / "best.json"
N_TRIALS = 24

SPACE = {"lr": [3e-4, 1e-3, 3e-3, 1e-2], "hidden": [64, 128, 256], "dropout": [0.1, 0.3, 0.5],
         "weight_decay": [0.0, 1e-4, 1e-3, 1e-2], "n_layers": [1, 2, 3], "heads": [2, 4, 8],
         "text_dim": [None, 64, 128]}
HPO_MODELS = {"mlp": "mlp", "sage": "sage", "gat": "gat", "han": "han_temporal", "hgt": "hgt_t0"}


def family(model: str) -> str:
    return model.split("_")[0]


def proto_family(protocol: str) -> str:
    return "mature" if protocol in ("mature", "retro", "deploy") else "kfold"


def tuned(model: str, task: str, protocol: str) -> dict:
    best = json.loads(HPO_FILE.read_text())
    return best[f"{family(model)}|{task}|{proto_family(protocol)}"]


def arm(name: str, model: str, base: dict, task: str, hp: str = "tuned", groups=None, text: str = PRIMARY_TEXT,
        **kw) -> dict:
    """One configuration. With tuned hyperparameters the text dimension selected for the model family is used for
    the SciBERT / SPECTER embeddings (``text_dim``), unless the arm sets it explicitly."""
    job = dict(base, arm=name, model=model, task=task, groups=list(groups or FEATS), text=text, biblio="exact",
               hp_mode=hp)
    job.update(kw)
    if hp == "tuned":
        t = tuned(model, task, job["protocol"])
        job["hp"] = t["hp"]
        if model == "hgt_roll":   # the snapshot graph with three HGT layers does not fit in 20 GB of GPU memory
            job["hp"] = dict(t["hp"], n_layers=min(t["hp"]["n_layers"], 2))
        if "text_dim" not in kw and text in ("scibert", "specter"):
            job["text_dim"] = t["text_dim"]
    return job


def expand(arms: list[dict]) -> list[dict]:
    """Repeat every arm over the partitions and seeds of its protocol."""
    jobs = []
    for a in arms:
        p = a["protocol"]
        if p == "mature":
            jobs += [dict(a, split_seed=s, init_seed=i) for s in range(5) for i in range(3)]
        elif p in ("kfold", "topicfold"):
            jobs += [dict(a, rep=r, fold=f, init_seed=i) for r in range(2) for f in range(6) for i in range(2)]
        elif p == "group":
            jobs += [dict(a, split_seed=s, init_seed=i) for s in range(5) for i in range(3)]
        elif p in ("retro", "deploy"):
            if a.get("train_subsample"):
                jobs += [dict(a, split_seed=s, init_seed=i) for s in range(5) for i in range(3)]
            else:
                jobs += [dict(a, init_seed=i) for i in range(5)]
        else:
            raise ValueError(p)
    return jobs


# ====================================================================== hyperparameter search
def hpo_trials(fam: str, task: str, pf: str) -> list[dict]:
    """``N_TRIALS`` configurations per model family: the submitted configuration, the same with 64 text components,
    then random draws from ``SPACE`` (same budget for every family)."""
    rng = np.random.default_rng(int(hashlib.sha1(f"{fam}|{task}|{pf}".encode()).hexdigest()[:8], 16))
    default = dict(lr=3e-3, hidden=128, dropout=0.3, weight_decay=1e-4, n_layers=2, heads=4)
    trials = [dict(hp=dict(default), text_dim=None), dict(hp=dict(default), text_dim=64)]
    seen = {json.dumps(t, sort_keys=True) for t in trials}
    while len(trials) < N_TRIALS:
        hp = {k: SPACE[k][rng.integers(len(SPACE[k]))] for k in ("lr", "hidden", "dropout", "weight_decay", "n_layers")}
        hp["heads"] = SPACE["heads"][rng.integers(3)] if fam in ("gat", "han", "hgt") else 4
        if fam == "hgt":
            hp["hidden"] = min(hp["hidden"], 128)        # memory of the heterogeneous graph on a 20 GB GPU
        t = dict(hp=hp, text_dim=SPACE["text_dim"][rng.integers(3)])
        key = json.dumps(t, sort_keys=True)
        if key not in seen:
            seen.add(key)
            trials.append(t)
    return trials


def grid_hpo(cfg: Config) -> list[dict]:
    jobs = []
    for pf, base in (("mature", MATURE), ("kfold", dict(KFOLD, k=3))):
        for task in ("reg", "cls"):
            for fam, model in HPO_MODELS.items():
                if task == "reg" and fam in ("han", "hgt"):   # the heterogeneous models are used for classification
                    continue
                for n, t in enumerate(hpo_trials(fam, task, pf)):
                    a = dict(base, arm=f"hpo|{fam}|{task}|{pf}|{n}", model=model, task=task, groups=FEATS,
                             text=PRIMARY_TEXT, biblio="exact", hp=t["hp"], text_dim=t["text_dim"], trial=n,
                             hp_mode="search")
                    if pf == "mature":
                        jobs += [dict(a, split_seed=s, init_seed=0) for s in range(3)]
                    else:
                        jobs += [dict(a, rep=0, fold=f, init_seed=0) for f in range(3)]
    return jobs


# ====================================================================== primary protocol (mature labels)
def arms_models(base: dict, hp: str = "tuned", tasks=("reg", "cls")) -> list[dict]:
    out = []
    for task in tasks:
        out += [arm("mlp", "mlp", base, task, hp),
                arm("mlp+refagg", "mlp", base, task, hp, groups=FEATS + ["refagg"]),
                arm("mlp+refagg_in", "mlp", base, task, hp, groups=FEATS + ["refagg_in"]),
                arm("sage", "sage", base, task, hp), arm("sage_e", "sage_e", base, task, hp),
                arm("sage+refagg", "sage", base, task, hp, groups=FEATS + ["refagg"]),
                arm("gat", "gat", base, task, hp), arm("gat_e", "gat_e", base, task, hp),
                arm("gat+refagg", "gat", base, task, hp, groups=FEATS + ["refagg"])]
        if task == "cls":
            out += [arm("han_temporal", "han_temporal", base, task, hp),
                    arm("hgt_t0", "hgt_t0", base, task, hp), arm("hgt_roll", "hgt_roll", base, task, hp),
                    arm("hgt_naive", "hgt_naive", base, task, hp)]
    return out


def arms_text(base: dict, hp: str = "tuned", tasks=("reg", "cls")) -> list[dict]:
    """Text x model x bibliometric features (RQ5)."""
    out = []
    for task in tasks:
        for m in ("mlp", "sage", "gat"):
            for text in TEXTS:
                for bib in (True, False):
                    out.append(arm(f"{m}|{text}|{'biblio' if bib else 'nobiblio'}", m, base, task, hp,
                                   groups=FEATS if bib else ["base"], text=text))
    return out


COLDSTART = [  # (variant, model, groups, text?, hide the paper's own text)
    ("mlp_authors", "mlp", ["base", "author"], False, False),
    ("mlp_refagg", "mlp", ["base", "refagg"], False, False),
    ("mlp_authors_refagg", "mlp", ["base", "author", "refagg"], False, False),
    ("sage_refs", "sage_e", ["base"], True, True),
    ("sage_refs_authors", "sage_e", ["base", "author"], True, True),
    ("sage_refs_authors_text", "sage_e", ["base", "author"], True, False),
    ("sage_full", "sage_e", FEATS, True, False),
    ("sage_plain_full", "sage", FEATS, True, False),
    ("mlp_full", "mlp", FEATS + ["refagg"], True, False),
]


def arms_coldstart(base: dict, hp: str = "tuned") -> list[dict]:
    return [arm(f"cs|{v}", m, base, "reg", hp, groups=gr, text=PRIMARY_TEXT if txt else "none", mask_own_text=mask)
            for v, m, gr, txt, mask in COLDSTART]


def grid_mature_main(cfg):
    return expand(arms_models(MATURE))


def grid_mature_text(cfg):
    return expand(arms_text(MATURE))


def grid_mature_coldstart(cfg):
    return expand(arms_coldstart(MATURE))


def grid_mature_default(cfg):
    """Submitted hyperparameters on the primary protocol: do the rankings change with tuning?"""
    arms = [a for a in arms_models(MATURE, hp="default")
            if a["arm"] in ("mlp", "mlp+refagg", "sage", "gat", "han_temporal", "hgt_t0", "hgt_roll")]
    return expand(arms)


def grid_mature_graphs(cfg):
    """Reference selection: submitted graph, strict cutoff, leave-self-out and symmetric cutoff variants."""
    arms = []
    for graph in ("submitted", "cut2022", "cut2022lso", "cut2022sym"):
        b = dict(MATURE, graph=graph)
        arms += [arm(f"mlp+refagg_in@{graph}", "mlp", b, "reg", groups=FEATS + ["refagg_in"]),
                 arm(f"sage@{graph}", "sage", b, "reg"), arm(f"gat@{graph}", "gat", b, "reg"),
                 arm(f"sage@{graph}", "sage", b, "cls"), arm(f"hgt_roll@{graph}", "hgt_roll", b, "cls")]
    return expand(arms)


def grid_mature_biblio(cfg):
    """Exact, strict and hindsight (submitted) author / venue statistics."""
    arms = []
    for bib in ("exact", "strict", "hind"):
        for task in ("reg", "cls"):
            for m in ("mlp", "sage"):
                for text in ("none", PRIMARY_TEXT):
                    arms.append(arm(f"{m}|{text}|{bib}", m, MATURE, task, text=text, biblio=bib))
    return expand(arms)


def grid_mature_audit(cfg):
    """Sensitivity to metadata whose state at the prediction date is uncertain (author disambiguation, keywords,
    venue records, abstract availability)."""
    arms = []
    for task in ("reg", "cls"):
        arms += [arm("mlp|no_author", "mlp", MATURE, task, groups=["base", "venue"]),
                 arm("mlp|no_venue", "mlp", MATURE, task, groups=["base", "author"]),
                 arm("mlp|base_only", "mlp", MATURE, task, groups=["base"]),
                 arm("sage|no_author", "sage", MATURE, task, groups=["base", "venue"])]
    arms += [arm("hgt_roll|no_keyword", "hgt_roll", MATURE, "cls", drop_rel=["keyword"]),   # HGT: classification only
             arm("hgt_roll|no_author", "hgt_roll", MATURE, "cls", drop_rel=["author"], groups=["base", "venue"]),
             arm("hgt_roll|papers_venues", "hgt_roll", MATURE, "cls", drop_rel=["author", "keyword"],
                 groups=["base", "venue"])]
    return expand(arms)


def grid_mature_origin2021(cfg):
    """Second prediction date (T = 2021: one training cohort, two test cohorts)."""
    b = dict(MATURE, T=2021, graph="cut2021sym")
    arms = [arm(f"{m}@T2021", m, b, task) for task in ("reg", "cls") for m in ("mlp", "sage", "gat")]
    arms += [arm("hgt_roll@T2021", "hgt_roll", b, "cls")]            # HGT: classification only
    arms += [arm("mlp+refagg@T2021", "mlp", b, "reg", groups=FEATS + ["refagg"])]
    return expand(arms)


def grid_mature_pca(cfg):
    """Text dimension as a controlled factor on the primary protocol (PCA fitted on the training graph)."""
    arms = []
    for text in ("scibert", "specter"):
        for dim in (None, 64):
            arms += [arm(f"{m}|{text}|d{dim or 768}", m, MATURE, task, text=text, text_dim=dim)
                     for task, ms in (("reg", ("mlp", "sage", "gat")), ("cls", ("mlp", "han_temporal", "hgt_roll")))
                     for m in ms]
    return expand(arms)


def grid_mature_field(cfg):
    """Top-10 % defined within publication year and field."""
    return expand([arm(f"{m}|field_year", m, MATURE, "cls", label="field_year")
                   for m in ("mlp", "sage", "gat", "han_temporal", "hgt_roll")])


# ====================================================================== retrospective protocols
def grid_retro(cfg):
    """Temporal split of the submitted article (labels not mature at the prediction date), with the corrected
    features, and the control with the training set reduced to the size of the mature protocol."""
    arms = []
    for k, T in ((3, 2021), (5, 2020)):
        b = dict(protocol="retro", T=T, k=k, graph="submitted")
        arms += [a for a in arms_models(b, tasks=("reg",)) if a["arm"] in ("mlp", "mlp+refagg", "sage", "sage_e", "gat")]
        arms += arms_coldstart(b)
        arms += [arm(m, m, b, "cls") for m in ("mlp", "sage", "hgt_roll")]
    b = dict(protocol="retro", T=2021, k=3, graph="submitted", train_subsample=2446)
    arms += [arm(f"{m}|sub2446", m, b, "reg") for m in ("mlp", "sage")]
    arms += [arm("mlp+refagg|sub2446", "mlp", b, "reg", groups=FEATS + ["refagg"])]
    return expand(arms)


def grid_kfold_main(cfg):
    arms = []
    for k in (3, 5):
        b = dict(KFOLD, k=k)
        arms += [a for a in arms_models(b, tasks=("reg",)) if not a["arm"].startswith(("han", "hgt"))]
        arms += [a for a in arms_models(b, tasks=("cls",)) if a["arm"] in ("mlp", "sage", "gat", "han_temporal",
                                                                             "hgt_t0", "hgt_roll", "hgt_naive")]
        arms += [arm("han_t0", "han_t0", b, "cls")]
    return expand(arms)


def grid_kfold_attention(cfg):
    """GAT attention against the prior citations and the age of the references (RQ1), k = 3 and 5."""
    arms = []
    for k in (3, 5):
        b = dict(KFOLD, k=k)
        arms += [arm(f"att|{m}", m, b, "reg", attention=True) for m in ("gat", "gat_e")]
    return expand(arms)


def grid_kfold_default(cfg):
    b = dict(KFOLD, k=3)
    arms = [a for a in arms_models(b, hp="default", tasks=("reg",)) if a["arm"] in ("mlp", "mlp+refagg", "sage", "gat")]
    arms += [a for a in arms_models(b, hp="default", tasks=("cls",))
             if a["arm"] in ("mlp", "sage", "gat", "han_temporal", "hgt_t0", "hgt_roll")]
    return expand(arms)


def grid_kfold_text(cfg):
    return expand(arms_text(dict(KFOLD, k=3)))


def grid_kfold_pca(cfg):
    """The exploratory comparison of the submitted article redone with PCA fitted on the training data: submitted
    hyperparameters, 768 dimensions against 64 components, plus the transductive fit (all nodes) as a control."""
    b = dict(KFOLD, k=3)
    arms = []
    for text in ("specter", "scibert"):
        for dim in (None, 64):
            arms += [arm(f"{m}|{text}|d{dim or 768}", m, b, task, hp="default", text=text, text_dim=dim)
                     for task, ms in (("reg", ("mlp", "sage", "gat")),
                                      ("cls", ("mlp", "han_temporal", "hgt_t0", "hgt_roll"))) for m in ms]
    arms += [arm(f"{m}|specter|d64|fit_all", m, b, "cls", hp="default", text="specter", text_dim=64, fit="all")
             for m in ("mlp", "hgt_t0")]
    return expand(arms)


def grid_kfold_biblio(cfg):
    b = dict(KFOLD, k=3)
    arms = []
    for bib in ("exact", "strict", "hind"):
        for task in ("reg", "cls"):
            for m in ("mlp", "sage"):
                for text in ("none", PRIMARY_TEXT):
                    arms.append(arm(f"{m}|{text}|{bib}", m, b, task, text=text, biblio=bib))
        arms.append(arm(f"hgt_t0|{PRIMARY_TEXT}|{bib}", "hgt_t0", b, "cls", biblio=bib))
    return expand(arms)


def grid_kfold_caps(cfg):
    """Sensitivity to the node cap and to the share of reference links kept."""
    arms = []
    for graph in ("cap30k", "cap45k", "cap60k", "uncapped"):
        b = dict(KFOLD, k=3, graph=graph)
        arms += [arm(f"mlp+refagg_in@{graph}", "mlp", b, "reg", groups=FEATS + ["refagg_in"]),
                 arm(f"sage@{graph}", "sage", b, "reg"), arm(f"gat@{graph}", "gat", b, "reg")]
    b = dict(KFOLD, k=3)
    for drop in (0.25, 0.5, 0.75, 1.0):
        arms += [arm(f"sage|drop{drop}", "sage", b, "reg", edge_drop=drop),
                 arm(f"gat|drop{drop}", "gat", b, "reg", edge_drop=drop)]
    return expand(arms)


def grid_kfold_firsthop(cfg):
    """Feature parity: GNNs on the graph in which every reference with metadata is a node (leaves carry the title
    only), against the MLP with all-reference aggregates of ``kfold_main``."""
    b = dict(KFOLD, k=3, graph="uncapped", graph_dir="rev1_firsthop")
    return expand([arm(f"{m}@firsthop", m, b, "reg") for m in ("sage", "sage_e", "gat")])


def grid_mature_firsthop(cfg):
    """The same graph in the primary protocol. Every reference of a paper is a node, so no selection by the number
    of citing papers is involved; the protocol keeps only the papers published before T in the training graph."""
    b = dict(MATURE, graph="uncapped", graph_dir="rev1_firsthop")
    return expand([arm(f"{m}@firsthop", m, b, "reg") for m in ("sage", "sage_e", "gat")])


def grid_kfold_field(cfg):
    b = dict(KFOLD, k=3)
    return expand([arm(f"{m}|field_year", m, b, "cls", label="field_year")
                   for m in ("mlp", "sage", "gat", "han_temporal", "hgt_t0", "hgt_roll")])


def grid_topicfold(cfg):
    """Cross-topic tests: K-fold over the 48 topics, and laboratory-related against open-corpus topics."""
    b = dict(protocol="topicfold", graph="submitted", k=3)
    arms = [arm(f"{m}|topicfold", m, b, task) for task, ms in (("reg", ("mlp", "sage", "gat")),
                                                               ("cls", ("mlp", "sage", "hgt_roll"))) for m in ms]
    arms += [arm("mlp+refagg|topicfold", "mlp", b, "reg", groups=FEATS + ["refagg"])]
    for grp in ("lab", "open"):
        g = dict(protocol="group", graph="submitted", k=3, test_group=grp)
        arms += [arm(f"{m}|test_{grp}", m, g, "reg") for m in ("mlp", "sage", "gat")]
    return expand(arms)


def grid_deploy(cfg):
    """Laboratory case study (illustrative): models trained on all labelled years, laboratory papers held out."""
    arms = []
    for k in (3, 5):
        b = dict(protocol="deploy", graph="submitted", k=k)
        arms += [arm("deploy|sage_full", "sage_e", b, "reg"),
                 arm("deploy|mlp_full", "mlp", b, "reg", groups=FEATS + ["refagg"])]
    return expand(arms)


def grid_frozen(cfg):
    """Arms of the confirmatory holdout (``holdout.py``): the primary protocol, with the model, scalers and
    projections of every run saved. Tuned arms answer the principal comparisons; the SPECTER arms with the
    submitted hyperparameters repeat the exploratory 768-dimension against 64-component comparison."""
    arms = [arm("mlp", "mlp", MATURE, "reg"), arm("mlp+refagg", "mlp", MATURE, "reg", groups=FEATS + ["refagg"]),
            arm("sage", "sage", MATURE, "reg"), arm("gat", "gat", MATURE, "reg"),
            arm("mlp", "mlp", MATURE, "cls"), arm("sage", "sage", MATURE, "cls"),
            arm("hgt_roll", "hgt_roll", MATURE, "cls")]
    for dim in (None, 64):
        arms += [arm(f"{m}|specter|d{dim or 768}", m, MATURE, "cls", hp="default", text="specter", text_dim=dim)
                 for m in ("mlp", "hgt_roll")]
    return [dict(j, save_ckpt=True) for j in expand(arms)]


GRIDS = {
    "hpo": grid_hpo, "frozen": grid_frozen,
    "mature_main": grid_mature_main, "mature_text": grid_mature_text, "mature_coldstart": grid_mature_coldstart,
    "mature_default": grid_mature_default, "mature_graphs": grid_mature_graphs, "mature_biblio": grid_mature_biblio,
    "mature_audit": grid_mature_audit, "mature_origin2021": grid_mature_origin2021, "mature_pca": grid_mature_pca,
    "mature_field": grid_mature_field,
    "retro": grid_retro, "kfold_main": grid_kfold_main, "kfold_default": grid_kfold_default,
    "kfold_text": grid_kfold_text, "kfold_pca": grid_kfold_pca, "kfold_biblio": grid_kfold_biblio,
    "kfold_caps": grid_kfold_caps, "kfold_attention": grid_kfold_attention, "kfold_firsthop": grid_kfold_firsthop, "mature_firsthop": grid_mature_firsthop, "kfold_field": grid_kfold_field, "topicfold": grid_topicfold,
    "deploy": grid_deploy,
}
