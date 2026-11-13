"""Experiment runner of the revision (rev1). The submitted runner (``experiments.py``) is unchanged.

``python experiments_rev1.py --grid <name> [--workers N] [--limit N]``   (grids are defined in ``grids_rev1.py``)

What changes with respect to the submitted runner:

- the data split (``split_seed`` / ``rep`` / ``fold``) and the weight initialisation (``init_seed``) are separate;
- standardisation, PCA and TF-IDF are fitted on the nodes that are neither validation nor test targets and that
  exist in the training graph;
- every run stores its test and validation predictions per paper (``results/rev1/preds/<id>.npz``), so that paired
  comparisons, per-topic and per-field results are computed from the same runs;
- hyperparameters are part of the job (``hp``); mixed precision is off for every model;
- the graph is a variant of the augmented graph (``graph_variants.py``) and the protocol is one of ``protocols.py``;
- bibliometric features come in three variants: ``exact`` (statistics dated before the publication year),
  ``strict`` (no author or venue citation counts at all) and ``hind`` (the features of the submitted article, which
  include citations received after publication and are kept only to measure their effect);
- the text of the keyword nodes follows the text setting of the job; ``hgt_roll`` uses year-specific entity
  snapshots that aggregate only earlier papers.

Results are appended to ``results/rev1/runs/<grid>.jsonl``; a job that is already there is not run again.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_graph import scatter
from config import CACHE_DIR, ROOT, Config, add_common_args, get_config, resolve_device, setup_logging
from experiments import attention_arrays, attention_stats, cls_metrics, predict, reg_metrics, train_model
from graph_variants import edge_mask, node_mask
from models import GAT, HAN, HGT, MLP, SAGE
from protocols import make_split

logger = logging.getLogger("experiments_rev1")

RES = ROOT / "results" / "rev1"
DEFAULT_HP = dict(hidden=128, heads=4, dropout=0.3, lr=3e-3, weight_decay=1e-4, n_layers=2, epochs=400, patience=40)
TRANSDUCTIVE = ("kfold", "topicfold", "group", "legacy")
SNAP_CAP = 50       # papers sampled per entity snapshot for the reverse (paper -> snapshot) edges
_G: dict = {}


def job_id(job: dict) -> str:
    return hashlib.sha1(json.dumps(job, sort_keys=True, default=str).encode()).hexdigest()[:14]


# ====================================================================== data
def load(cfg: Config, device, sub: str = "rev1") -> dict:
    key = (cfg.profile, str(device), sub)
    if _G.get("key") == key:
        return _G
    d = cfg.processed_dir / sub
    g = torch.load(d / "graph.pt", weights_only=False)
    het = torch.load(d / "hetero.pt", weights_only=False) if (d / "hetero.pt").exists() else {}
    to = lambda x: {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}  # noqa: E731
    _G.clear()
    _G.update(key=key, g=to(g), het=to(het), cache={})
    return _G


def cached(G: dict, key, fn):
    c = G["cache"]
    if key not in c:
        c[key] = fn()
    return c[key]


def t_idx(a: np.ndarray, device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a), device=device, dtype=torch.long)


def t_mask(a: np.ndarray, device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a), device=device, dtype=torch.bool)


def standardise(x: torch.Tensor, fit: torch.Tensor):
    mu, sd = x[fit].mean(0), x[fit].std(0).clamp_min(1e-6)
    return (x - mu) / sd, mu, sd


def fit_mask(vn: torch.Tensor, train_nodes: torch.Tensor, tr: torch.Tensor, va: torch.Tensor, te: torch.Tensor,
             mode: str | None = None) -> torch.Tensor:
    """Rows on which the scaler, the PCA and the TF-IDF model are fitted. Default: the nodes of the training graph
    that are neither validation nor test targets. ``train``: the labelled training papers only (control).
    ``all``: every node of the graph, as in the submitted code (kept to reproduce it)."""
    if mode == "all":
        return vn.clone()
    if mode == "train":
        fit = torch.zeros_like(vn)
        fit[tr] = True
        return fit
    fit = vn & train_nodes
    fit[va] = False
    fit[te] = False
    return fit


# ====================================================================== features
def pca_fit(x: torch.Tensor, fit: torch.Tensor, dim: int):
    mean = x[fit].mean(0)
    xc = x[fit] - mean
    evals, evecs = torch.linalg.eigh(xc.T @ xc / (len(xc) - 1))
    comp = evecs[:, -dim:].flip(1)
    return mean, comp, float(evals[-dim:].sum() / evals.sum())


def tfidf_features(G: dict, fit: torch.Tensor, dim: int) -> torch.Tensor:
    """TF-IDF of title + abstract reduced by truncated SVD; vocabulary, weights and SVD fitted on ``fit`` rows."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    g = G["g"]
    rows = fit.nonzero().squeeze(1).cpu().numpy()
    key = hashlib.sha1(rows.tobytes()).hexdigest()[:16] + f"_{dim}_{len(g['ids'])}"
    path = CACHE_DIR / "tfidf" / f"{key}.npy"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        docs = [d.replace("[SEP]", " ") for d in g["docs"]]
        vec = TfidfVectorizer(sublinear_tf=True, min_df=3, max_df=0.5, stop_words="english", max_features=50000,
                              dtype=np.float32).fit([docs[i] for i in rows])
        X = vec.transform(docs)
        svd = TruncatedSVD(dim, random_state=0).fit(X[rows])
        np.save(path, svd.transform(X).astype(np.float32))
    return torch.as_tensor(np.load(path), device=g["year"].device)


def text_matrix(G: dict, job: dict, fit: torch.Tensor, frozen: dict | None = None):
    """Text block of the paper features, and the projection used (for checkpoints). ``frozen`` applies a saved
    projection instead of fitting one."""
    g, text, dim = G["g"], job["text"], job.get("text_dim")
    if text == "none":
        return None, None
    if text == "tfidf":
        assert frozen is None, "TF-IDF arms are not part of the frozen evaluation"
        return tfidf_features(G, fit, dim or 128), None
    x = g[f"x_text_{text}"].float()
    if not dim:
        return x, None
    if frozen is not None:
        mean, comp, expl = frozen["mean"].to(x.device), frozen["comp"].to(x.device), frozen["explained"]
    else:
        mean, comp, expl = pca_fit(x, fit, dim)
    return (x - mean) @ comp, {"mean": mean.cpu(), "comp": comp.cpu(), "explained": expl}


def refagg_ingraph(g: dict, emask: torch.Tensor) -> torch.Tensor:
    """The seven reference aggregates of ``x_refagg``, restricted to the references that are nodes of the graph."""
    n = g["year"].shape[0]
    dst = g["edge_index"][1, emask]
    lc, age = g["edge_attr"][emask, 0].float(), g["edge_attr"][emask, 1].float()
    n_in = scatter(torch.ones_like(lc), dst, n, "sum")
    mean_lc = scatter(lc, dst, n, "mean")
    std_lc = (scatter(lc ** 2, dst, n, "mean") - mean_lc ** 2).clamp_min(0).sqrt()
    return torch.stack([mean_lc, scatter(lc, dst, n, "amax"), std_lc, scatter(age, dst, n, "mean"),
                        scatter(age, dst, n, "amin"), torch.log1p(n_in),
                        (n_in / g["n_refs"].clamp_min(1)).clamp(max=1)], 1)


def biblio_blocks(g: dict, biblio: str) -> dict[str, torch.Tensor]:
    if biblio == "hind":
        return {"author": g["x_biblio_hind"][:, 0:8], "venue": g["x_biblio_hind"][:, 8:12]}
    if biblio == "strict":
        return {"author": g["x_author_exact"][:, 4:8], "venue": g["x_venue_exact"][:, [1, 3]]}
    if biblio == "exact":
        return {"author": g["x_author_exact"], "venue": g["x_venue_exact"]}
    raise ValueError(biblio)


def paper_matrix(G: dict, job: dict, fit: torch.Tensor, emask: torch.Tensor, frozen: dict | None = None):
    g = G["g"]
    bib = biblio_blocks(g, job.get("biblio", "exact"))
    year_feature = job.get("year_feature", job["protocol"] in TRANSDUCTIVE)
    parts = []
    for grp in job["groups"]:
        if grp == "base":
            parts.append(g["x_base"] if year_feature else g["x_base"][:, 1:])
        elif grp in bib:
            parts.append(bib[grp])
        elif grp == "refagg":
            parts.append(g["x_refagg"])
        elif grp == "refagg_in":
            parts.append(refagg_ingraph(g, emask))
        else:
            raise ValueError(grp)
    xt, proj = text_matrix(G, job, fit, frozen["pca"] if frozen else None)
    d_text = 0
    if xt is not None:
        parts.append(xt)
        d_text = xt.shape[1]
    if frozen:
        mu, sd = frozen["mu"].to(xt.device if xt is not None else parts[0].device), frozen["sd"]
        sd = sd.to(mu.device)
        x = (torch.cat(parts, 1).float() - mu) / sd
    else:
        x, mu, sd = standardise(torch.cat(parts, 1).float(), fit)
    if job.get("mask_own_text") and d_text:
        target = torch.as_tensor([o != "ref" for o in g["origin"]], device=x.device)
        x[target, -d_text:] = 0
    return x, {"mu": mu.cpu(), "sd": sd.cpu(), "pca": proj, "d_text": d_text}


def cls_labels(G: dict, k: int, label: str, cfg: Config) -> torch.Tensor:
    """Top-10 % label: within the publication-year cohort (``year``, as submitted) or within the cohort of the same
    year and field (``field_year``; cells with fewer than 50 labelled papers are pooled per year)."""
    g = G["g"]
    if label == "year":
        return g[f"top{k}"].float()

    def build():
        m = g[f"m{k}"].bool().cpu().numpy()
        c, years = g[f"c{k}"].cpu().numpy(), g["year"].cpu().numpy()
        field = np.asarray(g["field"])
        cell = pd.Series([f"{y}|{f}" for y, f in zip(years, field)])
        size = cell[m].value_counts()
        small = cell.map(size).fillna(0).to_numpy() < 50
        cell = np.where(small, [f"{y}|other" for y in years], cell.to_numpy())
        top = np.zeros(len(c), dtype=np.float32)
        for u in np.unique(cell[m]):
            mm = m & (cell == u)
            top[mm] = (c[mm] >= np.quantile(c[mm], 1 - cfg.top_share)) & (c[mm] > 0)
        return torch.as_tensor(top, device=g["year"].device)

    return cached(G, ("cls_label", k, label), build)


# ====================================================================== heterogeneous inputs
def kw_text(G: dict, job: dict, frozen: dict | None = None):
    """Text of the keyword nodes and its projection (fitted on the keyword embeddings, or ``frozen``)."""
    het, text, dim = G["het"], job["text"], job.get("text_dim")
    if text not in ("specter", "scibert"):
        return None, None
    x = het[f"x_kw_text_{text}"].float()
    if not dim:
        return x, None

    def build():
        if frozen is not None:
            mean, comp = frozen["mean"].to(x.device), frozen["comp"].to(x.device)
        else:
            mean, comp, _ = pca_fit(x, torch.ones(len(x), dtype=torch.bool, device=x.device), dim)
        return (x - mean) @ comp, {"mean": mean.cpu(), "comp": comp.cpu()}

    return cached(G, ("kw_pca", text, dim, frozen is not None), build)


def entity_features(G: dict, job: dict, ent: dict | None = None, frozen_kw: dict | None = None):
    """Raw features of the author, venue and keyword nodes. ``ent`` maps a type to (entity index, column of the
    yearly tables) for snapshot nodes; without it the nodes are the entities as of T0 (first table column)."""
    het, biblio = G["het"], job.get("biblio", "exact")
    lo = int(het["snap_years"][0])

    def pick(name, n):
        if ent is None:
            return torch.arange(n, device=het["author_listed"].device), torch.zeros(n, dtype=torch.long,
                                                                                     device=het["author_listed"].device)
        return ent[name]

    ai, aj = pick("author", het["author_cit_tab"].shape[0])
    vi, vj = pick("venue", het["venue_cit_tab"].shape[0])
    ki, kj = pick("keyword", het["keyword_tab"].shape[0])
    listed = het["author_listed"][ai].float()
    works = torch.log1p(het["author_works_tab"][ai, aj])
    active = ((lo + aj).float() - het["author_first_year"][ai]).clamp(0, 60) / 10.0
    v_works = torch.log1p(het["venue_works_tab"][vi, vj])
    if biblio == "hind":
        assert ent is None, "the hindsight entity features exist only for the T0 rule"
        xa, xv = het["x_author"].float(), het["x_venue"].float()
    elif biblio == "strict":
        xa = torch.stack([works, active], 1)
        xv = torch.cat([v_works[:, None], het["venue_type"][vi]], 1)
    else:
        cit = torch.log1p(het["author_cit_tab"][ai, aj]) * listed
        xa = torch.stack([cit, works, active, listed], 1)
        vc, vn = het["venue_cit_tab"][vi, vj], het["venue_cnt_tab"][vi, vj]
        xv = torch.cat([torch.log1p(vc)[:, None], v_works[:, None], torch.log1p(vc / vn.clamp_min(1))[:, None],
                        het["venue_type"][vi]], 1)
    kc = torch.log1p(het["keyword_tab"][ki, kj])[:, None]
    kt, kw_proj = kw_text(G, job, frozen_kw)
    xk = kc if kt is None else torch.cat([kt[ki], kc], 1)
    return {"author": xa, "venue": xv, "keyword": xk}, kw_proj


def snapshots(G: dict) -> dict:
    """Year-specific entity nodes: snapshot (e, s) sends to the papers of entity e published in year s (clipped to
    the snapshot years) and receives only from the papers of e published before s."""
    def build():
        g, het = G["g"], G["het"]
        dev = g["year"].device
        years = g["year"].cpu().numpy()
        lo, hi = int(het["snap_years"][0]), int(het["snap_years"][-1])
        out = {}
        for name in ("author", "venue", "keyword"):
            e = het[f"rev_all_{name}"].cpu().numpy()
            p, ent = e[0], e[1]
            s = np.clip(years[p], lo, hi)
            uniq, inv = np.unique(ent.astype(np.int64) * 10000 + s, return_inverse=True)
            s_ent, s_year = uniq // 10000, uniq % 10000
            dp = pd.DataFrame({"ent": ent, "paper": p, "year": years[p]})
            ds = pd.DataFrame({"ent": s_ent, "sid": np.arange(len(uniq)), "s": s_year})
            m = dp.merge(ds, on="ent")
            m = m[m.year < m.s].sample(frac=1.0, random_state=0)
            m = m[m.groupby("sid").cumcount() < SNAP_CAP]
            out[name] = {"ent": t_idx(s_ent, dev), "col": t_idx(s_year - lo, dev), "year": t_idx(s_year, dev),
                         "fwd": t_idx(np.stack([inv, p]), dev),
                         "rev": t_idx(np.stack([m.paper.to_numpy(), m.sid.to_numpy()]), dev)}
        return out

    return cached(G, "snapshots", build)


REL = {"author": ("writes", "rev_writes"), "venue": ("publishes", "rev_publishes"), "keyword": ("tags", "rev_tags")}


def hetero_setup(G: dict, job: dict, x: torch.Tensor, cites: torch.Tensor, T: int | None, frozen: dict | None = None):
    """Node features, the edge dictionary restricted to a set of papers (function of a node mask), the metadata and
    the scalers of the entity features (``frozen`` applies saved scalers instead of fitting them)."""
    g, het, name = G["g"], G["het"], job["model"]
    if name.startswith("han"):
        rule = name.split("_", 1)[1]
        rels = {("paper", "pap", "paper"): het[f"mp_{rule}_author"], ("paper", "pvp", "paper"): het[f"mp_{rule}_venue"],
                ("paper", "pkp", "paper"): het[f"mp_{rule}_keyword"]}

        def eid(nm, em):
            d = {("paper", "cites", "paper"): g["edge_index"][:, em]}
            d.update({k: e[:, nm[e[0]] & nm[e[1]]] for k, e in rels.items()})
            return d

        return {"paper": x}, eid, (["paper"], [("paper", "cites", "paper")] + list(rels)), None

    if name == "hgt_roll":
        sn = snapshots(G)
        raw, kw_proj = entity_features(G, job, {k: (v["ent"], v["col"]) for k, v in sn.items()},
                                       frozen["kw_pca"] if frozen else None)
        fits = {k: (sn[k]["year"] < T) if T else torch.ones_like(sn[k]["year"], dtype=torch.bool) for k in sn}
        fwd = {k: sn[k]["fwd"] for k in sn}
        rev = {k: sn[k]["rev"] for k in sn}
    else:
        raw, kw_proj = entity_features(G, job, None, frozen["kw_pca"] if frozen else None)
        fits = {k: torch.ones(len(v), dtype=torch.bool, device=v.device) for k, v in raw.items()}
        kind = "rev_all" if name == "hgt_naive" else "rev_safe"
        fwd = {k: het[f"fwd_{k}"] for k in REL}
        rev = {k: het[f"{kind}_{k}"] for k in REL}
    xd, ent_scaler = {"paper": x}, {"kw_pca": kw_proj}
    for k, v in raw.items():
        if frozen:
            mu, sd = (t.to(v.device) for t in frozen[k])
            xd[k] = (v.float() - mu) / sd
        else:
            xd[k], mu, sd = standardise(v.float(), fits[k])
        ent_scaler[k] = (mu.cpu(), sd.cpu())

    rel = {k: v for k, v in REL.items() if k not in (job.get("drop_rel") or [])}   # audit: remove an entity type
    xd = {k: v for k, v in xd.items() if k == "paper" or k in rel}

    def eid(nm, em):
        d = {("paper", "cites", "paper"): g["edge_index"][:, em]}
        for k, (f, r) in rel.items():
            d[(k, f, "paper")] = fwd[k][:, nm[fwd[k][1]]]
            d[("paper", r, k)] = rev[k][:, nm[rev[k][0]]]
        return d

    etypes = [("paper", "cites", "paper")] + [t for k, (f, r) in rel.items() for t in ((k, f, "paper"), ("paper", r, k))]
    return xd, eid, (list(xd), etypes), ent_scaler


# ====================================================================== models
def build_model(name: str, in_dim: int, hp: dict, metadata=None, in_dims=None):
    base, kw = name.split("_")[0], dict(dropout=hp["dropout"], n_layers=hp["n_layers"])
    if base == "mlp":
        return MLP(in_dim, hp["hidden"], **kw)
    if base == "gat":
        return GAT(in_dim, hp["hidden"], heads=hp["heads"], edge_dim=2 if name == "gat_e" else None, **kw)
    if base == "sage":
        return SAGE(in_dim, hp["hidden"], edge_dim=2 if name == "sage_e" else None, **kw)
    if base == "han":
        return HAN(in_dim, hp["hidden"], metadata, heads=hp["heads"], **kw)
    if base == "hgt":
        return HGT(in_dims, hp["hidden"], metadata, heads=hp["heads"], **kw)
    raise ValueError(name)


# ====================================================================== one run
def run_node_job(job: dict, cfg: Config, G: dict, device) -> dict:
    g = G["g"]
    k, task, name = job["k"], job["task"], job["model"]
    hp = {**DEFAULT_HP, **(job.get("hp") or {})}
    seed = job.get("init_seed", job.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    N = g["year"].shape[0]
    split = make_split(g, job, cfg)
    tr, va, te = (t_idx(split[s], device) for s in ("tr", "va", "te"))
    train_nodes = t_mask(split["train_nodes"], device)

    variant = job.get("graph", "submitted")
    vn = cached(G, ("nodes", variant), lambda: node_mask(g, variant))
    em = cached(G, ("edges", variant), lambda: edge_mask(g, variant))
    if job.get("edge_drop"):                       # ablation: remove a share of the edges into target papers
        gen = torch.Generator(device="cpu").manual_seed(777)
        is_target = torch.as_tensor([o != "ref" for o in g["origin"]], device=device)[g["edge_index"][1]]
        drop = (torch.rand(len(em), generator=gen) < job["edge_drop"]).to(device) & is_target
        em = em & ~drop
    em_tr = em & train_nodes[g["edge_index"][0]] & train_nodes[g["edge_index"][1]]

    fit = fit_mask(vn, train_nodes, tr, va, te, job.get("fit"))
    x, scaler = paper_matrix(G, job, fit, em)
    y_all = (g[f"y{k}"] if task == "reg" else cls_labels(G, k, job.get("label", "year"), cfg)).float()
    amp = bool(job.get("amp")) and device.type == "cuda" and not name.startswith("hgt")
    ea = g["edge_attr"]

    ent_scaler = None
    if name.startswith(("han", "hgt")):
        xd, eid, meta, ent_scaler = hetero_setup(G, job, x, em, job.get("T"))
        model = build_model(name, x.shape[1], hp, metadata=meta, in_dims={t: v.shape[1] for t, v in xd.items()})
        e_tr, e_inf = eid(vn & train_nodes, em_tr), eid(vn, em)
        fwd_tr, fwd_inf = (lambda m: m(xd, e_tr)), (lambda m: m(xd, e_inf))
    else:
        model = build_model(name, x.shape[1], hp)
        ei_tr, ea_tr = g["edge_index"][:, em_tr], ea[em_tr]
        ei_inf, ea_inf = g["edge_index"][:, em], ea[em]
        fwd_tr, fwd_inf = (lambda m: m(x, ei_tr, ea_tr)), (lambda m: m(x, ei_inf, ea_inf))
    model = model.to(device)
    att = {}
    if job.get("attention"):                       # GAT only: attention of the untrained network as a reference
        att["untrained"] = attention_arrays(model, x, ei_inf, ea_inf, device)
    cfg_hp = replace(cfg, lr=hp["lr"], weight_decay=hp["weight_decay"], epochs=hp["epochs"], patience=hp["patience"])
    epochs = train_model(model, fwd_tr, y_all, tr, va, task, cfg_hp, device, amp)
    p = predict(model, fwd_inf, device, amp)
    if job.get("attention"):
        att["trained"] = attention_arrays(model, x, ei_inf, ea_inf, device)

    metric = reg_metrics if task == "reg" else (lambda a, b: cls_metrics(a, b, cfg.top_share))
    res = {"epochs": epochs, "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
           "in_dim": int(x.shape[1]), "train_graph": {"nodes": int((vn & train_nodes).sum()), "edges": int(em_tr.sum())},
           "graph": {"nodes": int(vn.sum()), "edges": int(em.sum())},
           "pca_explained": (scaler["pca"] or {}).get("explained")}
    ids = np.asarray(g["ids"])
    out = {}
    for part, idx in (("te", te), ("va", va)):
        i = idx.cpu().numpy()
        out[f"{part}_ids"], out[f"{part}_p"] = ids[i], p[idx].cpu().numpy().astype(np.float32)
        out[f"{part}_y"] = y_all[idx].cpu().numpy().astype(np.float32)
    if job["protocol"] != "deploy":
        res["metrics"] = metric(out["te_y"], out["te_p"])
        res["val_metrics"] = metric(out["va_y"], out["va_p"])
    if att:   # per-paper correlation of the last-layer attention with the reference's prior citations and age
        ei_np, ea_np, targets = ei_inf.cpu().numpy(), ea_inf.cpu().numpy(), te.cpu().numpy()
        res["attention"] = {kind: attention_stats(a["a2"], ei_np, ea_np, targets) for kind, a in att.items()}
        res["attention"]["trained_layer1"] = attention_stats(att["trained"]["a1"], ei_np, ea_np, targets)
        res["attention"]["self_attention_test_mean"] = float(att["trained"]["self"][targets].mean())
    jid = job_id(job)
    (RES / "preds").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RES / "preds" / f"{jid}.npz", **out)
    if job.get("save_ckpt"):
        (RES / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save({"state": {a: b.cpu() for a, b in model.state_dict().items()}, "scaler": scaler,
                    "ent_scaler": ent_scaler, "hp": hp, "job": job, "in_dim": int(x.shape[1])},
                   RES / "checkpoints" / f"{jid}.pt")
    return res


def predict_frozen(ckpt: dict, G: dict, device) -> torch.Tensor:
    """Predictions of a saved model on every node of the graphs in ``G``: the model weights, the feature scalers
    and the text projections are those of the checkpoint; nothing is fitted."""
    g, job, hp = G["g"], ckpt["job"], ckpt["hp"]
    name, variant = job["model"], job.get("graph", "submitted")
    vn, em = node_mask(g, variant), edge_mask(g, variant)
    x, _ = paper_matrix(G, job, vn, em, frozen=ckpt["scaler"])
    assert x.shape[1] == ckpt["in_dim"], "feature layout differs from the checkpoint"
    if name.startswith(("han", "hgt")):
        xd, eid, meta, _ = hetero_setup(G, job, x, em, job.get("T"), frozen=ckpt["ent_scaler"])
        model = build_model(name, x.shape[1], hp, metadata=meta, in_dims={t: v.shape[1] for t, v in xd.items()})
        e_inf = eid(vn, em)
        fwd = lambda m: m(xd, e_inf)  # noqa: E731
    else:
        model = build_model(name, x.shape[1], hp)
        ei, ea = g["edge_index"][:, em], g["edge_attr"][em]
        fwd = lambda m: m(x, ei, ea)  # noqa: E731
    model.load_state_dict(ckpt["state"])
    model = model.to(device).eval()
    return predict(model, fwd, device, False)


def run_job(job: dict) -> dict:
    cfg = get_config(job.get("profile", "full"), device=job.get("device"))
    device = resolve_device(cfg)
    G = load(cfg, device, sub=job.get("graph_dir", "rev1"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # two processes share the GPU (light and heterogeneous models): the models that need several GB take turns
    heavy = device.type == "cuda" and job["model"].split("_")[0] in ("gat", "han", "hgt")
    lock = open(RES / ".gpu.lock", "w") if heavy else None
    if lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        t = time.time()
        for attempt in range(40):      # memory may still be held by something else: wait and try again
            try:
                res = run_node_job(job, cfg, G, device)
                break
            except torch.OutOfMemoryError:
                if attempt == 39:
                    raise
                torch.cuda.empty_cache()
                time.sleep(30)
                t = time.time()
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if lock:
            lock.close()
    res.update(job=job, id=job_id(job), time_s=round(time.time() - t, 2),
               peak_gpu_mb=round(torch.cuda.max_memory_allocated(device) / 2**20, 1) if device.type == "cuda" else None)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


# ====================================================================== grids
def runs_path(grid: str) -> Path:
    p = RES / "runs" / f"{grid}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_runs(grid: str) -> list[dict]:
    """Records of a grid, one per run. A job that two processes ran at the same time has two lines; the last one
    is kept, because it is the run whose per-paper predictions are stored (``preds/<id>.npz`` is overwritten)."""
    p = runs_path(grid)
    if not p.exists():
        return []
    last: dict[str, dict] = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            last[r["id"]] = r
    return list(last.values())


def run_grid(grid: str, jobs: list[dict], workers: int = 1, limit: int | None = None) -> list[dict]:
    done = {r["id"] for r in read_runs(grid)}
    todo = [j for j in jobs if job_id(j) not in done]
    if limit:
        todo = todo[:limit]
    logger.info("=== grid %s: %d jobs, %d done, %d to run with %d worker(s)", grid, len(jobs), len(jobs) - len(todo),
                len(todo), workers)
    t0 = time.time()
    with open(runs_path(grid), "a") as f:
        def record(i, r):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                logger.info("  %s: %d/%d runs, %.0f s", grid, i + 1, len(todo), time.time() - t0)

        if workers <= 1:
            for i, j in enumerate(todo):
                if job_id(j) in {r["id"] for r in read_runs(grid)}:   # written meanwhile by another process
                    continue
                record(i, run_job(j))
        else:
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
                for i, r in enumerate(pool.map(run_job, todo, chunksize=1)):
                    record(i, r)
    return read_runs(grid)


def main() -> None:
    import grids_rev1

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--grid", required=True, help=f"comma list of {sorted(grids_rev1.GRIDS)}")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--models", default=None,
                    help="comma list of model families (mlp,sage,gat,han,hgt): run only these, so that two "
                         "processes can share a grid")
    args = ap.parse_args()
    cfg = get_config(args.profile, device=args.device)
    setup_logging("experiments_rev1", cfg)
    for name in args.grid.split(","):
        jobs = [dict(j, profile=cfg.profile) for j in grids_rev1.GRIDS[name](cfg)]
        if args.models:
            jobs = [j for j in jobs if j["model"].split("_")[0] in args.models.split(",")]
        run_grid(name, jobs, args.workers, args.limit)


if __name__ == "__main__":
    main()
