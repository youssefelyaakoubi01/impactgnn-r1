"""Experiments for the five sub-topics of Axis 1.

``python experiments.py --exp all|1,3 --profile small|full [--jobs N] [--device cuda|cpu]``

exp1  future impact (log citations at k = 3, 5 years) with GAT; attention analysis
exp2  inductive cold start with GraphSAGE (temporal split, test papers absent from training graph);
      predictions for the lab papers
exp3  heterogeneous graph (HAN, HGT) vs homogeneous GNNs for top-10 % detection
exp4  citation cartels / self-citation: injected anomalies on the venue citation graph
exp5  text vs structure ablation (SciBERT / SPECTER x MLP / GraphSAGE / GAT x bibliometric features)

Each experiment writes ``results/expN.json`` (per-run values and mean +- std over seeds). Runs are
executed by ``--jobs`` worker processes sharing the GPU; each worker loads the graphs once.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from torch_geometric.utils import subgraph

from config import Config, add_common_args, get_config, resolve_device, setup_logging, write_env_json
from models import GAT, HAN, HGT, MLP, SAGE, fit_detector

logger = logging.getLogger("experiments")

AUTHOR_COLS, VENUE_COLS = slice(0, 8), slice(8, 12)
_G: dict = {}  # per-process cache of the graphs on the device


# ====================================================================== data access
def load_graphs(cfg: Config, device) -> dict:
    key = (cfg.profile, str(device))
    if _G.get("key") == key:
        return _G
    g = torch.load(cfg.processed_dir / "graph.pt", weights_only=False)
    het = torch.load(cfg.processed_dir / "hetero.pt", weights_only=False)
    vg = torch.load(cfg.processed_dir / "venue_graph.pt", weights_only=False)
    to = lambda d: {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}
    _G.clear()
    _G.update(key=key, g=to(g), het=to(het), vg=vg)
    return _G


def text_features(g: dict, text: str, dim: int | None = None) -> torch.Tensor:
    """Text embeddings, optionally reduced to ``dim`` principal components (unsupervised PCA fit on all
    paper nodes, no labels involved; cached per process)."""
    x = g[f"x_text_{text}"].float()
    if not dim:
        return x
    key = f"x_text_{text}_pca{dim}"
    if key not in g:
        xc = x - x.mean(0)
        evals, evecs = torch.linalg.eigh(xc.T @ xc / (len(xc) - 1))  # ascending eigenvalues
        g[key] = xc @ evecs[:, -dim:].flip(1)
        g[f"{key}_explained"] = float(evals[-dim:].sum() / evals.sum())
    return g[key]


def feature_matrix(g: dict, groups: list[str], text: str, text_dim: int | None = None) -> torch.Tensor:
    parts = []
    if "base" in groups:
        parts.append(g["x_base"])
    if "author" in groups:
        parts.append(g["x_biblio"][:, AUTHOR_COLS])
    if "venue" in groups:
        parts.append(g["x_biblio"][:, VENUE_COLS])
    if "refagg" in groups:
        parts.append(g["x_refagg"])
    if text != "none":
        parts.append(text_features(g, text, text_dim))
    return torch.cat(parts, 1).float()


def standardise(x: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
    ref = x if rows is None else x[rows]
    return (x - ref.mean(0)) / ref.std(0).clamp_min(1e-6)


def random_split(g: dict, k: int, seed: int, frac) -> tuple[torch.Tensor, ...]:
    """Stratified (by publication year) random split of the labeled papers."""
    m, years = g[f"m{k}"].bool().cpu().numpy(), g["year"].cpu().numpy()
    rng = np.random.default_rng(seed)
    parts = ([], [], [])
    for y in np.unique(years[m]):
        idx = np.flatnonzero(m & (years == y))
        rng.shuffle(idx)
        a, b = round(frac[0] * len(idx)), round((frac[0] + frac[1]) * len(idx))
        for p, sl in zip(parts, (idx[:a], idx[a:b], idx[b:])):
            p.append(sl)
    dev = g["year"].device
    return tuple(torch.as_tensor(np.concatenate(p), device=dev) for p in parts)


def target_mask(g: dict) -> torch.Tensor:
    """Papers that are prediction targets (sample, open seeds, lab), as opposed to reference nodes."""
    return torch.as_tensor([o != "ref" for o in g["origin"]], device=g["year"].device)


# ====================================================================== metrics
def reg_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"spearman": float(spearmanr(y, p).correlation), "r2": float(r2_score(y, p)),
            "mae": float(mean_absolute_error(y, p))}


def cls_metrics(y: np.ndarray, s: np.ndarray, share: float = 0.10) -> dict:
    n_top = max(1, int(math.ceil(share * len(s))))
    top = np.argsort(-s, kind="stable")[:n_top]
    return {"roc_auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s)),
            "p_at_10": float(y[top].mean())}


# ====================================================================== models and training
def make_model(name: str, in_dim: int, cfg: Config, metadata=None, in_dims=None):
    base = name.split("_")[0]
    if base == "mlp":
        return MLP(in_dim, cfg.hidden, dropout=cfg.dropout)
    if base == "gat":
        return GAT(in_dim, cfg.hidden, heads=cfg.heads, dropout=cfg.dropout, edge_dim=2 if name == "gat_e" else None)
    if base == "sage":
        return SAGE(in_dim, cfg.hidden, dropout=cfg.dropout, edge_dim=2 if name == "sage_e" else None)
    if base == "han":
        return HAN(in_dim, cfg.hidden, metadata, heads=cfg.heads, dropout=cfg.dropout)
    if base == "hgt":
        return HGT(in_dims, cfg.hidden, metadata, heads=cfg.heads, dropout=cfg.dropout)
    raise ValueError(name)


def use_amp(name: str, cfg: Config, device) -> bool:
    """bf16 autocast on CUDA, except HGT: pyg-lib's segment_matmul (used by HGTConv) rejects bf16 inputs."""
    return cfg.amp and device.type == "cuda" and not name.startswith("hgt")


def train_model(model, fwd, y, idx_tr, idx_va, task: str, cfg: Config, device, amp: bool) -> int:
    """Full-batch training on the device with early stopping on the validation loss; returns epochs run."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if task == "cls":
        pos = y[idx_tr].mean().clamp(1e-3, 1 - 1e-3)
        pw = (1 - pos) / pos
        loss_fn = lambda o, t: F.binary_cross_entropy_with_logits(o, t, pos_weight=pw)
    else:
        loss_fn = F.mse_loss
    best, best_state, bad, ep = math.inf, None, 0, 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = fwd(model)
        loss = loss_fn(out.float().squeeze(-1)[idx_tr], y[idx_tr])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        model.eval()
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = fwd(model)
        vl = loss_fn(out.float().squeeze(-1)[idx_va], y[idx_va]).item()
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return ep


def predict(model, fwd, device, amp: bool) -> torch.Tensor:
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        return fwd(model).float().squeeze(-1)


# ====================================================================== node-level jobs (exp 1, 2, 3, 5)
def hetero_inputs(G: dict, model: str, x_paper: torch.Tensor):
    g, het = G["g"], G["het"]
    if model.startswith("han"):
        rule = model.split("_", 1)[1]  # t0 | temporal
        eid = {("paper", "cites", "paper"): g["edge_index"],
               ("paper", "pap", "paper"): het[f"mp_{rule}_author"],
               ("paper", "pvp", "paper"): het[f"mp_{rule}_venue"],
               ("paper", "pkp", "paper"): het[f"mp_{rule}_keyword"]}
        return {"paper": x_paper}, eid, (["paper"], list(eid))
    rev = "rev_all" if model == "hgt_naive" else "rev_safe"
    xd = {"paper": x_paper, "author": standardise(het["x_author"].float()),
          "venue": standardise(het["x_venue"].float()), "keyword": standardise(het["x_keyword"].float())}
    eid = {("paper", "cites", "paper"): g["edge_index"],
           ("author", "writes", "paper"): het["fwd_author"], ("venue", "publishes", "paper"): het["fwd_venue"],
           ("keyword", "tags", "paper"): het["fwd_keyword"],
           ("paper", "rev_writes", "author"): het[f"{rev}_author"],
           ("paper", "rev_publishes", "venue"): het[f"{rev}_venue"],
           ("paper", "rev_tags", "keyword"): het[f"{rev}_keyword"]}
    return xd, eid, (list(xd), list(eid))


def node_job(job: dict, cfg: Config, G: dict, device) -> dict:
    g = G["g"]
    k, task, name, seed = job["k"], job["task"], job["model"], job["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    y_all = (g[f"y{k}"] if task == "reg" else g[f"top{k}"]).float()
    N = g["year"].shape[0]
    x_raw = feature_matrix(g, job["groups"], job["text"], job.get("text_dim"))
    d_text = text_features(g, job["text"], job.get("text_dim")).shape[1] if job["text"] != "none" else 0
    ei, ea = g["edge_index"], g["edge_attr"]
    amp = use_amp(name, cfg, device)
    res: dict = {"amp": amp}

    if job["split"] == "random":
        tr, va, te = random_split(g, k, seed, cfg.split)
        x = standardise(x_raw)
        if job.get("mask_own_text") and job["text"] != "none":
            x[target_mask(g), -d_text:] = 0
        if name.startswith(("han", "hgt")):
            xd, eid, meta = hetero_inputs(G, name, x)
            model = make_model(name, x.shape[1], cfg, metadata=meta,
                               in_dims={t: v.shape[1] for t, v in xd.items()}).to(device)
            fwd = lambda m: m(xd, eid)
        else:
            model = make_model(name, x.shape[1], cfg).to(device)
            fwd = lambda m: m(x, ei, ea)
        if job.get("attention"):
            res["att_untrained"] = attention_arrays(model, x, ei, ea, device)
        res["epochs"] = train_model(model, fwd, y_all, tr, va, task, cfg, device, amp)
        p = predict(model, fwd, device, amp)
        if job.get("attention"):
            res["att_trained"] = attention_arrays(model, x, ei, ea, device)
            res["test_idx"] = te.cpu().numpy()
        eval_idx = te
    else:
        # ---- inductive temporal split: the training graph contains only papers published before T
        mask_lab = g[f"m{k}"].bool()
        years = g["year"]
        if job["split"] == "temporal":
            T = cfg.coldstart_test_start[k]
            keep = years < T
            tr, va = mask_lab & (years < T - 1), mask_lab & (years == T - 1)
            te = mask_lab & (years >= T)
        else:  # "deploy": all labeled years for training, last labeled year for validation; lab papers held out
            ymax = int(years[mask_lab].max())
            keep = (years <= ymax) & torch.as_tensor([o != "seed_lab" for o in g["origin"]], device=device)
            tr, va = mask_lab & (years < ymax), mask_lab & (years == ymax)
            te = torch.as_tensor([o == "seed_lab" for o in g["origin"]], device=device)
        assert not (te & keep).any() or job["split"] != "temporal", "test node inside the training graph"
        sub_ei, sub_ea = subgraph(keep, ei, ea, relabel_nodes=True, num_nodes=N)
        new_id = torch.cumsum(keep.long(), 0) - 1
        x_tr = standardise(x_raw[keep])
        mu, sd = x_raw[keep].mean(0), x_raw[keep].std(0).clamp_min(1e-6)
        x_full = (x_raw - mu) / sd
        if job.get("mask_own_text") and job["text"] != "none":
            d = d_text
            tm = target_mask(g)
            x_full[tm, -d:] = 0
            x_tr[tm[keep], -d:] = 0
        model = make_model(name, x_raw.shape[1], cfg).to(device)
        idx_tr, idx_va = new_id[tr & keep], new_id[va & keep]
        res["epochs"] = train_model(model, lambda m: m(x_tr, sub_ei, sub_ea), y_all[keep], idx_tr, idx_va,
                                    task, cfg, device, amp)
        p = predict(model, lambda m: m(x_full, ei, ea), device, amp)
        eval_idx = te.nonzero().squeeze(1)
        res["train_graph_nodes"] = int(keep.sum())
        res["train_graph_edges"] = int(sub_ei.shape[1])
        if job["split"] == "deploy":
            res["lab_idx"] = eval_idx.cpu().numpy()
            res["lab_pred"] = p[eval_idx].cpu().numpy()
            return res

    yt, pt = y_all[eval_idx].cpu().numpy(), p[eval_idx].cpu().numpy()
    res["metrics"] = reg_metrics(yt, pt) if task == "reg" else cls_metrics(yt, pt, cfg.top_share)
    res["n_train"], res["n_test"] = int(len(tr) if tr.dtype != torch.bool else tr.sum()), int(len(eval_idx))
    return res


def attention_arrays(model, x, ei, ea, device) -> dict:
    """Last- and first-layer attention (mean over heads) of the original edges, and self-loop attention."""
    model.eval()
    with torch.no_grad():
        _, (ei2, a1, a2) = model(x.float(), ei, ea, return_attention=True)
    E = ei.shape[1]
    assert torch.equal(ei2[:, :E], ei), "GATConv changed the edge order"
    self_dst = ei2[1, E:]
    self_att = torch.zeros(x.shape[0], device=x.device)
    self_att[self_dst] = a2[E:].float().mean(1)
    return {"a2": a2[:E].float().mean(1).cpu().numpy().astype(np.float32),
            "a1": a1[:E].float().mean(1).cpu().numpy().astype(np.float32),
            "self": self_att.cpu().numpy().astype(np.float32)}


# ====================================================================== anomaly jobs (exp 4)
def venue_dense(vg: dict) -> np.ndarray:
    n = len(vg["venue_ids"])
    W = np.zeros((n, n), dtype=np.float64)
    np.add.at(W, (vg["src"].numpy(), vg["dst"].numpy()), vg["w"].numpy())
    return W


def venue_structure(W: np.ndarray, out_total: np.ndarray) -> dict[str, np.ndarray]:
    """Structural indicators per venue; ``out_total`` includes citations to venues outside the graph."""
    n = W.shape[0]
    off = W.copy()
    np.fill_diagonal(off, 0)
    out_in = off.sum(1)
    sym = off + off.T
    s = sym.sum(1)
    return {
        "log_out": np.log1p(out_total), "log_in": np.log1p(W.sum(0)),
        "self_ratio": np.diag(W) / np.maximum(out_total, 1),
        "reciprocity": np.minimum(off, off.T).sum(1) / np.maximum(out_in, 1),
        "top_partner_share": sym.max(1) / np.maximum(s, 1),
        "log_partners": np.log1p((sym > 0).sum(1)),
        "hhi": ((sym / np.maximum(s, 1)[:, None]) ** 2).sum(1),
    }


def inject(W: np.ndarray, out_total: np.ndarray, cfg: Config, f: float, rng: np.random.Generator):
    """Add cartel groups (heavy mutual citation) and self-citing venues; returns W', new out totals, labels."""
    n = W.shape[0]
    W2, out2 = W.copy(), out_total.astype(np.float64).copy()
    labels = np.zeros(n, dtype=np.int64)
    perm = rng.permutation(n)
    m = cfg.cartel_size
    groups = [perm[i * m:(i + 1) * m] for i in range(cfg.n_cartels)]
    selfers = perm[cfg.n_cartels * m: cfg.n_cartels * m + cfg.n_selfcit]
    for grp in groups:
        for i in grp:
            add = max(1, round(f * out_total[i] / (m - 1)))
            for j in grp:
                if i != j:
                    W2[i, j] += add
                    out2[i] += add
        labels[grp] = 1
    for i in selfers:
        add = max(1, round(f * out_total[i]))
        W2[i, i] += add
        out2[i] += add
    labels[selfers] = 2
    return W2, out2, labels, [g.tolist() for g in groups], selfers.tolist()


def adjacency_target(W: np.ndarray) -> np.ndarray:
    A = np.log1p(W + W.T - np.diag(np.diag(W)))
    return A / max(A.max(), 1e-9)


def anomaly_scores(W: np.ndarray, out_total: np.ndarray, content: np.ndarray, seed: int, device) -> dict:
    st = venue_structure(W, out_total)
    S = np.column_stack(list(st.values()))
    A = torch.as_tensor(adjacency_target(W), dtype=torch.float32, device=device)
    scores = {"heur_self_ratio": st["self_ratio"], "heur_reciprocity": st["reciprocity"]}
    r = lambda v: pd.Series(v).rank(pct=True).to_numpy()
    scores["heur_max"] = np.maximum(r(st["self_ratio"]), r(st["reciprocity"]))
    for fs, X in (("content", content), ("full", np.column_stack([content, S]))):
        Xs = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
        xt = torch.as_tensor(Xs, dtype=torch.float32, device=device)
        scores[f"dominant_{fs}"] = fit_detector("dominant", xt, A, seed).cpu().numpy()
        scores[f"gae_{fs}"] = fit_detector("gae", xt, A, seed).cpu().numpy()
        iso = IsolationForest(n_estimators=300, random_state=seed).fit(Xs)
        scores[f"iforest_{fs}"] = -iso.score_samples(Xs)
    return scores, st


def anomaly_job(job: dict, cfg: Config, G: dict, device) -> dict:
    vg = G["vg"]
    W, out_total = venue_dense(vg), vg["out_total"].numpy().astype(np.float64)
    content = np.column_stack([vg["content"].numpy(), np.log1p(vg["works_count"].numpy())])
    if job["intensity"] == "none":  # real graph: cases for inspection
        scores, st = anomaly_scores(W, out_total, content, job["seed"], device)
        return {"scores": {k: v.astype(np.float32) for k, v in scores.items()},
                "structure": {k: v.astype(np.float32) for k, v in st.items()}}
    rng = np.random.default_rng(10_000 + job["seed"])
    W2, out2, labels, groups, selfers = inject(W, out_total, cfg, cfg.intensities[job["intensity"]], rng)
    scores, _ = anomaly_scores(W2, out2, content, job["seed"], device)
    metrics = {}
    for det, s in scores.items():
        for target, lab in (("cartel", 1), ("self", 2), ("all", None)):
            keep = (labels == 0) | (labels == lab) if lab else np.ones_like(labels, bool)
            yt = (labels[keep] > 0).astype(int)
            metrics[f"{det}|{target}"] = {"roc_auc": float(roc_auc_score(yt, s[keep])),
                                          "ap": float(average_precision_score(yt, s[keep]))}
    return {"metrics": metrics, "n_venues": int(len(labels)), "n_cartel": int((labels == 1).sum()),
            "n_self": int((labels == 2).sum())}


# ====================================================================== job runner
def run_job(job: dict) -> dict:
    cfg = get_config(job["profile"], device=job.get("device"))
    device = resolve_device(cfg)
    G = load_graphs(cfg, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t = time.time()
    res = (anomaly_job if job["kind"] == "anomaly" else node_job)(job, cfg, G, device)
    res.update(job=job, time_s=round(time.time() - t, 2), device=str(device),
               peak_gpu_mb=round(torch.cuda.max_memory_allocated(device) / 2**20, 1) if device.type == "cuda" else None)
    if device.type == "cuda":
        torch.cuda.empty_cache()  # return cached blocks so that concurrent workers do not starve
    return res


def run_all(jobs: list[dict], n_workers: int) -> list[dict]:
    if n_workers <= 1:
        out = []
        for i, j in enumerate(jobs):
            out.append(run_job(j))
            if (i + 1) % 10 == 0:
                logger.info("  %d/%d runs done", i + 1, len(jobs))
        return out
    out = []
    with ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn")) as pool:
        for i, r in enumerate(pool.map(run_job, jobs, chunksize=1)):
            out.append(r)
            if (i + 1) % 10 == 0:
                logger.info("  %d/%d runs done", i + 1, len(jobs))
    return out


def agg(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def summarise_runs(runs: list[dict], key_fn) -> dict:
    groups = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if "metrics" not in r:
            continue
        for m, v in r["metrics"].items():
            groups[key_fn(r["job"])][m].append(v)
    return {k: {m: agg(v) for m, v in ms.items()} for k, ms in groups.items()}


def strip(r: dict) -> dict:
    """Keep the JSON-serialisable part of a run."""
    return {k: v for k, v in r.items() if k in ("job", "metrics", "epochs", "time_s", "device", "peak_gpu_mb", "amp",
                                                "n_train", "n_test", "train_graph_nodes", "train_graph_edges")}


# ====================================================================== experiment definitions
def jobs_exp1(cfg: Config) -> list[dict]:
    feats = ["base", "author", "venue"]
    models = [("mlp", feats), ("mlp_refagg", feats + ["refagg"]), ("sage", feats), ("sage_e", feats),
              ("gat", feats), ("gat_e", feats)]
    return [dict(kind="node", exp=1, model=m, groups=gr, text=cfg.default_text, task="reg", k=k, split="random",
                 seed=s, attention=m.startswith("gat"))
            for k in cfg.horizons for m, gr in models for s in cfg.seeds]


COLDSTART = [  # (variant, model, groups, text?, mask own text)
    ("mlp_authors", "mlp", ["base", "author"], False, False),
    ("mlp_refagg", "mlp", ["base", "refagg"], False, False),
    ("mlp_authors_refagg", "mlp", ["base", "author", "refagg"], False, False),
    ("sage_refs", "sage_e", ["base"], True, True),
    ("sage_refs_authors", "sage_e", ["base", "author"], True, True),
    ("sage_refs_authors_text", "sage_e", ["base", "author"], True, False),
    ("sage_full", "sage_e", ["base", "author", "venue"], True, False),
    ("sage_plain_full", "sage", ["base", "author", "venue"], True, False),
    ("mlp_full", "mlp", ["base", "author", "venue", "refagg"], True, False),
]
DEPLOY = ["sage_full", "mlp_full"]


def jobs_exp2(cfg: Config) -> list[dict]:
    jobs = []
    for k in cfg.horizons:
        for var, m, gr, txt, mask in COLDSTART:
            for s in cfg.seeds:
                base = dict(kind="node", exp=2, variant=var, model=m, groups=gr,
                            text=cfg.default_text if txt else "none", mask_own_text=mask, task="reg", k=k, seed=s)
                jobs.append(dict(base, split="temporal"))
                if var in DEPLOY:
                    jobs.append(dict(base, split="deploy"))
    return jobs


def jobs_exp3(cfg: Config) -> list[dict]:
    models = ["mlp", "sage", "gat", "han_t0", "han_temporal", "hgt_t0", "hgt_naive"]
    return [dict(kind="node", exp=3, model=m, groups=["base", "author", "venue"], text=cfg.default_text,
                 task="cls", k=k, split="random", seed=s)
            for k in cfg.horizons for m in models for s in cfg.seeds]


def jobs_exp4(cfg: Config) -> list[dict]:
    jobs = [dict(kind="anomaly", exp=4, intensity=lvl, seed=d)
            for lvl in cfg.intensities for d in range(cfg.injection_draws)]
    jobs += [dict(kind="anomaly", exp=4, intensity="none", seed=s) for s in cfg.seeds]
    return jobs


def jobs_exp5(cfg: Config) -> list[dict]:
    jobs = []
    for task in ("reg", "cls"):
        for m in ("mlp", "sage", "gat"):
            for text in ("none", "scibert", "specter"):
                for biblio in (False, True):
                    gr = ["base"] + (["author", "venue"] if biblio else [])
                    for s in cfg.seeds:
                        jobs.append(dict(kind="node", exp=5, model=m, groups=gr, text=text, biblio=biblio,
                                         task=task, k=3, split="random", seed=s))
    return jobs


# ====================================================================== post-processing
def attention_stats(alpha: np.ndarray, ei: np.ndarray, ea: np.ndarray, targets: np.ndarray, min_deg: int = 3) -> dict:
    """Correlation of attention with the reference's prior citations and age, per citing paper and pooled."""
    df = pd.DataFrame({"dst": ei[1], "a": alpha, "cit": ea[:, 0], "age": ea[:, 1]})
    df = df[np.isin(df.dst, targets)]
    deg = df.groupby("dst").a.transform("size")
    df = df[deg >= min_deg]
    if df.empty:
        return {"n_targets": 0}
    rk = df.groupby("dst")[["a", "cit", "age"]].rank()
    rk["dst"] = df.dst.values

    def per_paper(col):
        def corr(g):
            if g[col].nunique() < 2 or g.a.nunique() < 2:
                return np.nan
            return np.corrcoef(g.a, g[col])[0, 1]
        c = rk.groupby("dst").apply(corr, include_groups=False).dropna()
        return {"mean": float(c.mean()), "median": float(c.median()), "share_positive": float((c > 0).mean()),
                "n": int(len(c))}

    dn = df.a * df.groupby("dst").a.transform("size")  # degree-normalised attention (1 = uniform)
    return {"n_targets": int(df.dst.nunique()), "n_edges": int(len(df)),
            "per_paper_cit": per_paper("cit"), "per_paper_age": per_paper("age"),
            "pooled_cit": float(spearmanr(dn, df.cit).correlation),
            "pooled_age": float(spearmanr(dn, df.age).correlation)}


def post_exp1(cfg: Config, runs: list[dict], G: dict) -> dict:
    g = G["g"]
    ei, ea = g["edge_index"].cpu().numpy(), g["edge_attr"].cpu().numpy()
    out = {"summary": {f"k{k}": summarise_runs([r for r in runs if r["job"]["k"] == k], lambda j: j["model"])
                       for k in cfg.horizons}}
    att = {}
    for m in ("gat", "gat_e"):
        for k in cfg.horizons:
            rs = [r for r in runs if r["job"]["model"] == m and r["job"]["k"] == k]
            if not rs:
                continue
            per = {"trained": [], "untrained": [], "trained_layer1": []}
            for r in rs:
                per["trained"].append(attention_stats(r["att_trained"]["a2"], ei, ea, r["test_idx"]))
                per["untrained"].append(attention_stats(r["att_untrained"]["a2"], ei, ea, r["test_idx"]))
                per["trained_layer1"].append(attention_stats(r["att_trained"]["a1"], ei, ea, r["test_idx"]))
            flat = {}
            for kind, lst in per.items():
                lst = [s for s in lst if s.get("n_targets", 0) > 0]
                if not lst:
                    continue
                flat[kind] = {
                    "per_paper_cit_mean": agg([s["per_paper_cit"]["mean"] for s in lst]),
                    "per_paper_cit_share_pos": agg([s["per_paper_cit"]["share_positive"] for s in lst]),
                    "per_paper_age_mean": agg([s["per_paper_age"]["mean"] for s in lst]),
                    "per_paper_age_share_pos": agg([s["per_paper_age"]["share_positive"] for s in lst]),
                    "pooled_cit": agg([s["pooled_cit"] for s in lst]),
                    "pooled_age": agg([s["pooled_age"] for s in lst]),
                    "n_targets": agg([s["n_targets"] for s in lst]),
                }
            labeled = g[f"m{k}"].bool().cpu().numpy()
            self_share = [float(r["att_trained"]["self"][r["test_idx"]].mean()) for r in rs]
            flat["self_attention_test_mean"] = agg(self_share)
            att[f"{m}|k{k}"] = flat
            # concrete examples: attention averaged over seeds, 5 labeled papers with the highest impact
            if m == "gat" and k == cfg.horizons[0]:
                a_mean = np.mean([r["att_trained"]["a2"] for r in rs], 0)
                out["examples"] = attention_examples(g, ei, ea, a_mean, labeled, k)
    out["attention"] = att
    return out


def attention_examples(g: dict, ei, ea, a_mean, labeled, k: int, n: int = 5, min_deg: int = 5) -> list[dict]:
    deg = np.bincount(ei[1], minlength=len(g["ids"]))
    yk = g[f"c{k}"].cpu().numpy()
    cand = np.flatnonzero(labeled & (deg >= min_deg))
    cand = cand[np.argsort(-yk[cand])][:n]
    years = g["year"].cpu().numpy()
    ex = []
    for p in cand:
        e = np.flatnonzero(ei[1] == p)
        order = e[np.argsort(-a_mean[e])]
        fmt = lambda j: {"title": g["titles"][ei[0, j]], "year": int(years[ei[0, j]]),
                         "citations_before_citing_year": int(round(np.expm1(ea[j, 0]))),
                         "attention": float(a_mean[j]), "uniform": float(1 / len(e))}
        ex.append({"title": g["titles"][p], "year": int(years[p]), "venue": g["venue_name"][p],
                   f"citations_{k}y": int(yk[p]), "n_refs_in_graph": int(len(e)),
                   "top3": [fmt(j) for j in order[:3]], "bottom3": [fmt(j) for j in order[-3:][::-1]]})
    return ex


def post_exp2(cfg: Config, runs: list[dict], G: dict) -> dict:
    g = G["g"]
    ev = [r for r in runs if r["job"]["split"] == "temporal"]
    out = {"summary": {f"k{k}": summarise_runs([r for r in ev if r["job"]["k"] == k], lambda j: j["variant"])
                       for k in cfg.horizons},
           "split": {f"k{k}": {"test_from": cfg.coldstart_test_start[k], "val_year": cfg.coldstart_test_start[k] - 1}
                     for k in cfg.horizons},
           "train_graph": {f"k{k}": next(({"nodes": r["train_graph_nodes"], "edges": r["train_graph_edges"]}
                                          for r in ev if r["job"]["k"] == k), None) for k in cfg.horizons}}
    # ---- lab predictions (deployment models trained on all labeled years, averaged over seeds)
    rows = {}
    years = g["year"].cpu().numpy()
    for r in runs:
        if r["job"]["split"] != "deploy":
            continue
        k, var = r["job"]["k"], r["job"]["variant"]
        for i, p in zip(r["lab_idx"], r["lab_pred"]):
            rows.setdefault(int(i), {}).setdefault(f"{var}_k{k}", []).append(float(p))
    table = []
    for i, preds in rows.items():
        row = {"doc_id": g["doc_id"][i], "openalex_id": g["ids"][i], "title": g["titles"][i], "year": int(years[i]),
               "venue": g["venue_name"][i]}
        for key, v in sorted(preds.items()):
            k = int(key.rsplit("_k", 1)[1])
            mean_log = float(np.mean(v))
            lab = g[f"m{k}"].bool().cpu().numpy()
            row[f"{key}_pred_citations"] = round(float(np.expm1(mean_log)), 1)
            row[f"{key}_pred_log_std"] = round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 3)
            row[f"{key}_percentile"] = round(float((g[f"y{k}"].cpu().numpy()[lab] < mean_log).mean() * 100), 1)
        for k in cfg.horizons:
            hi = years[i] + k if cfg.label_inclusive else years[i] + k - 1
            complete = cfg.cby_first_year <= years[i] and hi <= cfg.last_complete_year
            row[f"actual_citations_k{k}"] = int(g[f"c{k}"][i]) if complete else None
        row["extrapolated_year"] = bool(years[i] > int(years[g[f"m{cfg.horizons[0]}"].bool().cpu().numpy()].max())
                                        or years[i] < cfg.t0)
        table.append(row)
    table.sort(key=lambda r: (-r["year"], r["title"]))
    pd.DataFrame(table).to_csv(cfg.results_dir / "exp2_lab_predictions.csv", index=False)
    out["lab_predictions"] = table
    return out


def post_exp3(cfg: Config, runs: list[dict], G: dict) -> dict:
    return {"summary": {f"k{k}": summarise_runs([r for r in runs if r["job"]["k"] == k], lambda j: j["model"])
                        for k in cfg.horizons}}


def post_exp4(cfg: Config, runs: list[dict], G: dict) -> dict:
    vg = G["vg"]
    inj = [r for r in runs if r["job"]["intensity"] != "none"]
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in inj:
        for key, mv in r["metrics"].items():
            det, target = key.split("|")
            for m, v in mv.items():
                summary[r["job"]["intensity"]][f"{det}|{target}"][m].append(v)
    out = {"summary": {lvl: {k: {m: agg(v) for m, v in d.items()} for k, d in dd.items()}
                       for lvl, dd in summary.items()},
           "graph": {"venues": len(vg["venue_ids"]), "edges": int(len(vg["w"])), "links": int(vg["w"].sum())},
           "injection": {"n_cartels": cfg.n_cartels, "cartel_size": cfg.cartel_size, "n_selfcit": cfg.n_selfcit,
                         "intensities": cfg.intensities, "draws": cfg.injection_draws}}
    # ---- real graph: venues ranked by DOMINANT (full features), averaged over seeds -> cases for inspection
    real = [r for r in runs if r["job"]["intensity"] == "none"]
    if real:
        s = np.mean([pd.Series(r["scores"]["dominant_full"]).rank(pct=True).to_numpy() for r in real], 0)
        st = real[0]["structure"]
        W = venue_dense(vg)
        off = W + W.T
        np.fill_diagonal(off, 0)
        top = np.argsort(-s)[:15]
        out["cases_for_inspection"] = [{
            "venue": vg["names"][i], "openalex_id": vg["venue_ids"][i], "dominant_rank_pct": float(s[i]),
            "self_ratio": float(st["self_ratio"][i]), "reciprocity": float(st["reciprocity"][i]),
            "out_links": int(vg["out_total"][i]), "in_links_graph": int(W[:, i].sum()),
            "top_partner": vg["names"][int(off[i].argmax())], "top_partner_share": float(st["top_partner_share"][i]),
        } for i in top]
        out["size_correlation"] = {
            det: {"log_out": float(spearmanr(np.mean([r["scores"][det] for r in real], 0), st["log_out"]).correlation),
                  "log_in": float(spearmanr(np.mean([r["scores"][det] for r in real], 0), st["log_in"]).correlation)}
            for det in real[0]["scores"]}
        out["real_graph_reference"] = {k: {"median": float(np.median(v)), "p90": float(np.quantile(v, 0.9))}
                                       for k, v in st.items() if k in ("self_ratio", "reciprocity", "top_partner_share")}
    return out


def post_exp5(cfg: Config, runs: list[dict], G: dict) -> dict:
    key = lambda j: f"{j['model']}|{j['text']}|{'biblio' if j['biblio'] else 'nobiblio'}"
    return {"summary": {t: summarise_runs([r for r in runs if r["job"]["task"] == t], key) for t in ("reg", "cls")}}


# HGT on the full heterogeneous graph peaks at ~10 GB (measured): one worker for exp3 on a 20 GB GPU
MAX_WORKERS = {3: 1}

EXPS = {1: (jobs_exp1, post_exp1), 2: (jobs_exp2, post_exp2), 3: (jobs_exp3, post_exp3),
        4: (jobs_exp4, post_exp4), 5: (jobs_exp5, post_exp5)}


def text_explained(G: dict, text: str, dim: int) -> float:
    text_features(G["g"], text, dim)
    return G["g"][f"x_text_{text}_pca{dim}_explained"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--exp", default="all", help="all or comma list, e.g. 1,3")
    ap.add_argument("--text-dim", type=int, default=None,
                    help="exploratory: reduce text embeddings to this many PCA components (outputs expN_pcaD.json)")
    args = ap.parse_args()
    cfg = get_config(args.profile, device=args.device, jobs=args.jobs)
    setup_logging("experiments", cfg)
    device = resolve_device(cfg)
    env = write_env_json(cfg)
    exps = sorted(EXPS) if args.exp == "all" else [int(e) for e in args.exp.split(",")]
    G = load_graphs(cfg, device)
    for e in exps:
        jobs_fn, post_fn = EXPS[e]
        jobs = [dict(j, profile=cfg.profile, device=args.device, text_dim=args.text_dim) for j in jobs_fn(cfg)]
        workers = min(cfg.jobs, MAX_WORKERS.get(e, cfg.jobs))
        logger.info("=== exp%d: %d runs on %s with %d worker(s)", e, len(jobs), env.get("gpu", device), workers)
        t = time.time()
        runs = run_all(jobs, workers)
        res = post_fn(cfg, runs, G)
        res.update(exp=e, profile=cfg.profile, device=str(device), gpu=env.get("gpu"), workers=workers,
                   wall_time_s=round(time.time() - t, 1), config=cfg.to_dict(),
                   runs=[strip(r) for r in runs if r["job"]["kind"] == "node" or "metrics" in r])
        suffix = f"_pca{args.text_dim}" if args.text_dim else ""
        if args.text_dim:
            res.update(text_dim=args.text_dim, exploratory=True,
                       pca_explained_variance={m: text_explained(G, m, args.text_dim) for m in ("specter", "scibert")})
        (cfg.results_dir / f"exp{e}{suffix}.json").write_text(json.dumps(res, indent=1, default=float))
        logger.info("=== exp%d done in %.0f s -> %s", e, time.time() - t, cfg.results_dir / f"exp{e}{suffix}.json")


if __name__ == "__main__":
    main()
