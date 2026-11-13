"""Embeddings, time-safe features, labels, homogeneous / heterogeneous graphs and the venue graph.

``python build_graph.py --profile small|full``

Leakage rules implemented here (see ``check_*`` functions, also unit-tested):
- citation edges go only from a reference to the citing paper, and only if year(ref) <= year(citing);
- every statistic attached to a paper published in year y is computed from citations received
  before y, with ``before(y) = total - sum(counts_by_year[t] for t >= y)``; years before the first
  covered year (2012) are clamped to it, i.e. they use information dated <= 2011, which is still
  earlier than every target (targets are published >= 2017);
- heterogeneous graph: author / venue / keyword nodes receive messages only from papers published
  before T0 = 2017 (``rev_safe``); the leaky variant (``rev_all``) is kept only to measure leakage;
- HAN metapath neighbours Q of a target P satisfy year(Q) < T0 (rule ``t0``) or year(Q) < year(P)
  (rule ``temporal``).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch

from config import CACHE_DIR, TEXT_MODELS, Config, add_common_args, get_config, resolve_device, setup_logging, write_env_json

logger = logging.getLogger("build_graph")

TYPES = ["article", "review", "preprint"]          # others -> "other"
VENUE_TYPES = ["journal", "conference", "repository", "book series"]
YEAR_HI = 2027                                     # suffix sums cover years cby_first .. YEAR_HI


# ====================================================================== time-safe counts
class YearlyCounts:
    """Totals and per-year counts of many entities; ``before(idx, y)`` = total minus counts in years >= y."""

    def __init__(self, totals, years_lists, counts_lists, first_year: int) -> None:
        self.first = first_year
        n, width = len(totals), YEAR_HI - first_year + 2
        per_year = np.zeros((n, width), dtype=np.int64)
        for i, (ys, cs) in enumerate(zip(years_lists, counts_lists)):
            for y, c in zip(ys, cs):
                if first_year <= y <= YEAR_HI:
                    per_year[i, y - first_year] += c
        # suffix[i, j] = sum of counts in years >= first + j
        self.suffix = np.cumsum(per_year[:, ::-1], axis=1)[:, ::-1]
        self.totals = np.asarray(totals, dtype=np.int64)
        self.per_year = per_year

    def before(self, idx, years) -> np.ndarray:
        idx = np.asarray(idx)
        j = np.clip(np.asarray(years) - self.first, 0, self.suffix.shape[1] - 1)
        return np.maximum(self.totals[idx] - self.suffix[idx, j], 0)

    def window(self, idx, y0, y1) -> np.ndarray:
        """Sum of counts in years y0..y1 (inclusive); requires y0 >= first covered year."""
        idx = np.asarray(idx)
        a = np.clip(np.asarray(y0) - self.first, 0, self.suffix.shape[1] - 1)
        b = np.clip(np.asarray(y1) + 1 - self.first, 0, self.suffix.shape[1] - 1)
        return self.suffix[idx, a] - self.suffix[idx, b]


def label_counts(cby_years, cby_counts, year: int, k: int, inclusive: bool = True) -> int:
    """Citations received in years year .. year+k (inclusive) or year .. year+k-1."""
    hi = year + k if inclusive else year + k - 1
    return int(sum(c for y, c in zip(cby_years, cby_counts) if year <= y <= hi))


def label_complete(year: int, k: int, cfg: Config) -> bool:
    hi = year + k if cfg.label_inclusive else year + k - 1
    return cfg.t0 <= year and year >= cfg.cby_first_year and hi <= cfg.last_complete_year


# ====================================================================== embeddings (GPU, fp16, cached)
def embed(ids: list[str], texts: list[str], model_key: str, kind: str, cfg: Config, device) -> np.ndarray:
    """Sentence embeddings cached by work id in ``data/cache/emb/<model>_<kind>.npz``."""
    from sentence_transformers import SentenceTransformer

    path = CACHE_DIR / "emb" / f"{model_key}_{kind}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    store: dict[str, np.ndarray] = {}
    if path.exists():
        z = np.load(path, allow_pickle=False)
        store = dict(zip(z["ids"].tolist(), z["vecs"]))
    missing = [(i, t) for i, t in zip(ids, texts) if i not in store]
    if missing:
        t0 = time.time()
        model = SentenceTransformer(TEXT_MODELS[model_key], device=str(device))
        model.max_seq_length = cfg.embed_max_len if kind == "doc" else 64
        if device.type == "cuda":
            model.half()
        vecs = model.encode([t for _, t in missing], batch_size=cfg.embed_batch, convert_to_numpy=True,
                            show_progress_bar=False, normalize_embeddings=False)
        for (i, _), v in zip(missing, vecs.astype(np.float16)):
            store[i] = v
        logger.info("Embedded %d texts with %s (%s) in %.0f s on %s", len(missing), model_key, kind,
                    time.time() - t0, device)
        all_ids = np.array(list(store.keys()))
        np.savez(path, ids=all_ids, vecs=np.stack([store[i] for i in all_ids]))
        del model
        torch.cuda.empty_cache()
    return np.stack([store[i] for i in ids]).astype(np.float32)


def doc_text(title: str, abstract: str | None) -> str:
    title = (title or "").strip()
    return f"{title} [SEP] {abstract.strip()}" if abstract else title


# ====================================================================== node table
def load_nodes(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paper nodes (seeds, sample, reference nodes) and the metadata of every known work."""
    raw = cfg.raw_dir
    seeds = pd.read_parquet(raw / "seeds.parquet")
    seeds = seeds[seeds.has_record & ~seeds.dup_work]
    sw = pd.read_parquet(raw / "seed_works.parquet")
    sample = pd.read_parquet(raw / "sample.parquet")
    refs = pd.read_parquet(raw / "refs.parquet")
    node_refs = set(json.loads((raw / "node_refs.json").read_text()))
    texts = pd.read_parquet(raw / "node_ref_texts.parquet").set_index("id")

    corpus = dict(zip(seeds.openalex_id, seeds.corpus))
    doc_id = dict(zip(seeds.openalex_id, seeds.doc_id))
    stratum = dict(zip(seeds.openalex_id, seeds.stratum))
    sw = sw[sw.id.isin(corpus)].assign(origin=lambda d: d.id.map(lambda i: f"seed_{corpus[i]}"))
    sample = sample.assign(origin="sample")
    rn = refs[refs.id.isin(node_refs)].copy()
    rn["abstract"] = rn.id.map(texts.abstract).fillna("")
    rn["keywords"] = rn.id.map(texts.keywords)
    rn["origin"] = "ref"
    prio = {"seed_lab": 0, "sample": 1, "seed_open": 2, "ref": 3}
    cols = ["id", "title", "abstract", "year", "type", "cited_by_count", "cby_years", "cby_counts", "refs",
            "n_refs", "venue_id", "venue_name", "author_ids", "keywords", "topic_id", "topic_name", "field",
            "origin"]
    nodes = pd.concat([sw.reindex(columns=cols + ["topic_group"]), sample.reindex(columns=cols + ["topic_group"]),
                       rn.reindex(columns=cols + ["topic_group"])], ignore_index=True)
    nodes["prio"] = nodes.origin.map(prio)
    in_sample = set(sample.id)
    nodes = nodes.sort_values(["prio", "id"]).drop_duplicates("id").drop(columns="prio")
    nodes["in_sample"] = nodes.id.isin(in_sample)
    nodes = nodes[nodes.year.notna()].reset_index(drop=True)
    nodes["year"] = nodes.year.astype(int)
    nodes["abstract"] = nodes.abstract.fillna("")
    nodes["keywords"] = nodes.keywords.apply(lambda k: list(k) if k is not None and not isinstance(k, float) else [])
    nodes["doc_id"] = nodes.id.map(doc_id).fillna("")
    nodes["stratum"] = nodes.id.map(stratum).fillna("")

    meta = pd.concat([nodes[["id", "year", "cited_by_count", "cby_years", "cby_counts", "venue_id", "title"]],
                      refs[["id", "year", "cited_by_count", "cby_years", "cby_counts", "venue_id", "title"]]],
                     ignore_index=True).drop_duplicates("id")
    meta = meta[meta.year.notna()].reset_index(drop=True)
    meta["year"] = meta.year.astype(int)
    return nodes, meta


# ====================================================================== features
def scatter(values: torch.Tensor, index: torch.Tensor, n: int, reduce: str) -> torch.Tensor:
    out = torch.zeros(n, device=values.device, dtype=values.dtype)
    return out.scatter_reduce(0, index, values, reduce=reduce, include_self=False)


def paper_features(nodes: pd.DataFrame, meta: pd.DataFrame, authors: pd.DataFrame, venues: pd.DataFrame,
                   cfg: Config, device) -> dict[str, np.ndarray]:
    """Base, bibliometric (authors + venue before y) and reference-aggregate features (GPU scatter)."""
    n = len(nodes)
    years = nodes.year.to_numpy()

    # ---- base
    typ = np.zeros((n, len(TYPES) + 1), dtype=np.float32)
    for i, t in enumerate(nodes.type):
        typ[i, TYPES.index(t) if t in TYPES else len(TYPES)] = 1
    base = np.column_stack([(years - cfg.t0) / 5.0, typ, np.log1p(nodes.n_refs.fillna(0).to_numpy()),
                            (nodes.abstract.str.len() > 0).to_numpy()]).astype(np.float32)

    # ---- authors before y (max / mean / first / last of log citations, works; team size)
    a_idx = {a: i for i, a in enumerate(authors.id)}
    ac = YearlyCounts(authors.cited_by_count, authors.cby_years, authors.cby_cited, cfg.cby_first_year)
    aw = YearlyCounts(authors.works_count, authors.cby_years, authors.cby_works, cfg.cby_first_year)
    rows, cols, pos, last = [], [], [], []
    for i, lst in enumerate(nodes.author_ids):
        lst = list(lst) if lst is not None else []
        for j, a in enumerate(lst):
            if a in a_idx:
                rows.append(i)
                cols.append(a_idx[a])
                pos.append(j)
                last.append(j == len(lst) - 1)
    rows, cols = np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)
    c = np.log1p(ac.before(cols, years[rows])).astype(np.float32) if len(rows) else np.zeros(0, np.float32)
    w = np.log1p(aw.before(cols, years[rows])).astype(np.float32) if len(rows) else np.zeros(0, np.float32)
    R = torch.as_tensor(rows, device=device)
    C, W = torch.as_tensor(c, device=device), torch.as_tensor(w, device=device)
    first_m = torch.as_tensor(np.array(pos) == 0, device=device)
    last_m = torch.as_tensor(np.array(last, dtype=bool), device=device)
    n_auth = np.array([len(x) if x is not None else 0 for x in nodes.author_ids], dtype=np.float32)
    n_rec = scatter(torch.ones_like(C), R, n, "sum")
    author_f = torch.stack([
        scatter(C, R, n, "amax"), scatter(C, R, n, "mean"),
        scatter(C[first_m], R[first_m], n, "amax"), scatter(C[last_m], R[last_m], n, "amax"),
        scatter(W, R, n, "amax"), scatter(W, R, n, "mean"),
        torch.as_tensor(np.log1p(n_auth), device=device),
        n_rec / torch.as_tensor(np.maximum(n_auth, 1), device=device),
    ], 1).cpu().numpy()

    # ---- venue before y
    v_idx = {v: i for i, v in enumerate(venues.id)}
    vc = YearlyCounts(venues.cited_by_count, venues.cby_years, venues.cby_cited, cfg.cby_first_year)
    vw = YearlyCounts(venues.works_count, venues.cby_years, venues.cby_works, cfg.cby_first_year)
    vi = nodes.venue_id.map(v_idx)
    has_v = vi.notna().to_numpy()
    venue_f = np.zeros((n, 4), dtype=np.float32)
    if has_v.any():
        ii = vi[has_v].astype(int).to_numpy()
        cb, wb = vc.before(ii, years[has_v]), vw.before(ii, years[has_v])
        venue_f[has_v] = np.column_stack([np.log1p(cb), np.log1p(wb), np.log1p(cb / np.maximum(wb, 1)),
                                          np.ones(len(ii))])
    biblio = np.concatenate([author_f, venue_f], 1).astype(np.float32)

    # ---- reference aggregates over all references with metadata (citations before y, age)
    m_idx = {m: i for i, m in enumerate(meta.id)}
    mc = YearlyCounts(meta.cited_by_count, meta.cby_years, meta.cby_counts, cfg.cby_first_year)
    myear = meta.year.to_numpy()
    rr, rm = [], []
    for i, lst in enumerate(nodes.refs):
        for r in (lst if lst is not None else []):
            j = m_idx.get(r)
            if j is not None and myear[j] <= years[i]:
                rr.append(i)
                rm.append(j)
    rr, rm = np.array(rr, dtype=np.int64), np.array(rm, dtype=np.int64)
    lc = np.log1p(mc.before(rm, years[rr])).astype(np.float32) if len(rr) else np.zeros(0, np.float32)
    age = np.log1p(np.maximum(years[rr] - myear[rm], 0)).astype(np.float32) if len(rr) else np.zeros(0, np.float32)
    R, LC, AG = (torch.as_tensor(x, device=device) for x in (rr, lc, age))
    n_meta = scatter(torch.ones_like(LC), R, n, "sum")
    mean_lc = scatter(LC, R, n, "mean")
    std_lc = (scatter(LC ** 2, R, n, "mean") - mean_lc ** 2).clamp_min(0).sqrt()
    n_refs = torch.as_tensor(np.maximum(nodes.n_refs.fillna(0).to_numpy(), 1), device=device, dtype=torch.float32)
    refagg = torch.stack([
        mean_lc, scatter(LC, R, n, "amax"), std_lc,
        scatter(AG, R, n, "mean"), scatter(AG, R, n, "amin"),
        torch.log1p(n_meta), (n_meta / n_refs).clamp(max=1),
    ], 1).cpu().numpy().astype(np.float32)
    return {"base": base, "biblio": biblio, "refagg": refagg}, mc, m_idx


# ====================================================================== homogeneous graph
def citation_edges(nodes: pd.DataFrame, mc: YearlyCounts, m_idx: dict, cfg: Config):
    """Edges reference -> citing among paper nodes, with (log citations of the ref before the citing year, log age)."""
    idx = {p: i for i, p in enumerate(nodes.id)}
    years = nodes.year.to_numpy()
    src, dst, dropped = [], [], 0
    for i, lst in enumerate(nodes.refs):
        for r in (lst if lst is not None else []):
            j = idx.get(r)
            if j is None or j == i:
                continue
            if years[j] > years[i]:
                dropped += 1
                continue
            src.append(j)
            dst.append(i)
    src, dst = np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)
    mi = np.array([m_idx[p] for p in nodes.id.to_numpy()[src]], dtype=np.int64)
    attr = np.column_stack([np.log1p(mc.before(mi, years[dst])), np.log1p(years[dst] - years[src])]).astype(np.float32)
    return np.stack([src, dst]), attr, dropped


def labels(nodes: pd.DataFrame, cfg: Config) -> dict[str, np.ndarray]:
    out = {}
    eligible = nodes.origin.isin(["sample", "seed_open"]).to_numpy()   # lab papers are the held-out case study
    for k in cfg.horizons:
        c = np.array([label_counts(ys, cs, y, k, cfg.label_inclusive)
                      for ys, cs, y in zip(nodes.cby_years, nodes.cby_counts, nodes.year)], dtype=np.float32)
        mask = eligible & np.array([label_complete(int(y), k, cfg) for y in nodes.year])
        top = np.zeros(len(nodes), dtype=np.float32)
        for y in np.unique(nodes.year[mask]):
            m = mask & (nodes.year.to_numpy() == y)
            thr = np.quantile(c[m], 1 - cfg.top_share)
            top[m] = (c[m] >= thr) & (c[m] > 0)
        out[f"c{k}"], out[f"y{k}"], out[f"m{k}"], out[f"top{k}"] = c, np.log1p(c), mask, top
    return out


# ====================================================================== heterogeneous graph
def metapath_edges(groups: list[np.ndarray], years: np.ndarray, rule: str, t0: int, k: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Paper->paper edges Q->P for papers sharing a group (author / venue / keyword).

    rule ``t0``: year(Q) < t0; rule ``temporal``: year(Q) < year(P). At most ``k`` sampled Q per
    target and group, then at most ``k`` per target overall.
    """
    src_l, dst_l = [], []
    for g in groups:
        if len(g) < 2:
            continue
        order = g[np.argsort(years[g], kind="stable")]
        ys = years[order]
        if rule == "t0":
            cnt = np.full(len(order), np.searchsorted(ys, t0, "left"))
        else:
            cnt = np.searchsorted(ys, ys, "left")  # members strictly older than each target
        has = cnt > 0
        if not has.any():
            continue
        tgt = np.repeat(order[has], np.minimum(cnt[has], k))
        lim = np.repeat(cnt[has], np.minimum(cnt[has], k))
        pick = order[(rng.random(len(lim)) * lim).astype(np.int64)]
        src_l.append(pick)
        dst_l.append(tgt)
    if not src_l:
        return np.zeros((2, 0), dtype=np.int64)
    e = np.unique(np.stack([np.concatenate(src_l), np.concatenate(dst_l)]), axis=1)
    e = e[:, e[0] != e[1]]
    # cap per target
    perm = rng.permutation(e.shape[1])
    e = e[:, perm]
    order = np.argsort(e[1], kind="stable")
    e = e[:, order]
    rank = np.arange(e.shape[1]) - np.searchsorted(e[1], e[1], "left")
    return e[:, rank < k]


def build_hetero(nodes: pd.DataFrame, feats: dict, homo_ei: np.ndarray, authors: pd.DataFrame,
                 venues: pd.DataFrame, cfg: Config, device) -> dict:
    years = nodes.year.to_numpy()
    n = len(nodes)
    rng = np.random.default_rng(cfg.sample_seed)
    a_idx = {a: i for i, a in enumerate(authors.id)}
    v_idx = {v: i for i, v in enumerate(venues.id)}
    kw_count = Counter(k for ks in nodes.keywords for k in set(ks))
    kws = sorted(k for k, c in kw_count.items() if c >= 2)
    k_idx = {k: i for i, k in enumerate(kws)}

    pa = [(i, a_idx[a]) for i, lst in enumerate(nodes.author_ids) for a in (lst if lst is not None else []) if a in a_idx]
    pv = [(i, v_idx[v]) for i, v in enumerate(nodes.venue_id) if v in v_idx]
    pk = [(i, k_idx[k]) for i, ks in enumerate(nodes.keywords) for k in set(ks) if k in k_idx]
    rel = {"author": np.array(pa, dtype=np.int64).reshape(-1, 2).T,
           "venue": np.array(pv, dtype=np.int64).reshape(-1, 2).T,
           "keyword": np.array(pk, dtype=np.int64).reshape(-1, 2).T}

    # ---- static node features of authors / venues / keywords, all dated before T0
    t0 = cfg.t0
    ac = YearlyCounts(authors.cited_by_count, authors.cby_years, authors.cby_cited, cfg.cby_first_year)
    aw = YearlyCounts(authors.works_count, authors.cby_years, authors.cby_works, cfg.cby_first_year)
    ai = np.arange(len(authors))
    first_year = np.array([min([y for y, w in zip(ys, ws) if w > 0] or [t0]) for ys, ws in
                           zip(authors.cby_years, authors.cby_works)])
    x_author = np.column_stack([np.log1p(ac.before(ai, t0)), np.log1p(aw.before(ai, t0)),
                                np.clip(t0 - first_year, 0, 60) / 10.0]).astype(np.float32)
    vc = YearlyCounts(venues.cited_by_count, venues.cby_years, venues.cby_cited, cfg.cby_first_year)
    vw = YearlyCounts(venues.works_count, venues.cby_years, venues.cby_works, cfg.cby_first_year)
    vi = np.arange(len(venues))
    vt = np.zeros((len(venues), len(VENUE_TYPES) + 1), dtype=np.float32)
    for i, t in enumerate(venues.type):
        vt[i, VENUE_TYPES.index(t) if t in VENUE_TYPES else len(VENUE_TYPES)] = 1
    cb, wb = vc.before(vi, t0), vw.before(vi, t0)
    x_venue = np.column_stack([np.log1p(cb), np.log1p(wb), np.log1p(cb / np.maximum(wb, 1)), vt]).astype(np.float32)
    pre = years < t0
    kw_pre = np.zeros(len(kws), dtype=np.float32)
    if rel["keyword"].size:
        np.add.at(kw_pre, rel["keyword"][1][pre[rel["keyword"][0]]], 1)
    x_kw_text = embed([f"kw::{k}" for k in kws], kws, cfg.default_text, "keyword", cfg, device) if kws else \
        np.zeros((0, 768), np.float32)
    x_keyword = np.concatenate([x_kw_text, np.log1p(kw_pre)[:, None]], 1).astype(np.float32)

    # ---- edges: forward (entity -> paper) always; reverse (paper -> entity) safe = papers before T0 only
    het = {"x_author": x_author, "x_venue": x_venue, "x_keyword": x_keyword, "keywords": kws,
           "author_ids": authors.id.tolist(), "venue_ids": venues.id.tolist(), "cites": homo_ei}
    for name, e in rel.items():
        het[f"fwd_{name}"] = np.stack([e[1], e[0]]) if e.size else np.zeros((2, 0), np.int64)  # entity -> paper
        rev = e  # paper -> entity
        het[f"rev_all_{name}"] = rev
        het[f"rev_safe_{name}"] = rev[:, pre[rev[0]]] if rev.size else rev

    # ---- HAN metapaths P-A-P, P-V-P, P-K-P under both leakage-free rules
    for name, e in rel.items():
        if not e.size:
            for rule in ("t0", "temporal"):
                het[f"mp_{rule}_{name}"] = np.zeros((2, 0), np.int64)
            continue
        order = np.argsort(e[1], kind="stable")
        ent, pap = e[1][order], e[0][order]
        cuts = np.flatnonzero(np.diff(ent)) + 1
        groups = np.split(pap, cuts)
        for rule in ("t0", "temporal"):
            het[f"mp_{rule}_{name}"] = metapath_edges(groups, years, rule, t0, cfg.metapath_k, rng)
    return het


# ====================================================================== venue citation graph (sub-topic 4)
def build_venue_graph(nodes: pd.DataFrame, meta: pd.DataFrame, venues: pd.DataFrame, cfg: Config, device) -> dict:
    """Weighted directed graph citing venue -> cited venue from all reference links of sample/seed papers."""
    vmeta = dict(zip(meta.id, meta.venue_id))
    citing = nodes[nodes.origin.isin(["sample", "seed_open", "seed_lab"])]
    pairs = Counter()
    for v, refs in zip(citing.venue_id, citing.refs):
        if not v:
            continue
        for r in set(refs if refs is not None else []):
            rv = vmeta.get(r)
            if rv:
                pairs[(v, rv)] += 1
    out_tot, in_tot = Counter(), Counter()
    for (a, b), w in pairs.items():
        out_tot[a] += w
        in_tot[b] += w
    known = set(venues.id)
    keep = sorted(v for v in out_tot if v in known and out_tot[v] >= cfg.venue_min_out
                  and out_tot[v] + in_tot[v] >= cfg.venue_min_links)
    vi = {v: i for i, v in enumerate(keep)}
    e = [(vi[a], vi[b], w) for (a, b), w in pairs.items() if a in vi and b in vi]
    src, dst, w = (np.array(x, dtype=np.int64) for x in zip(*e)) if e else (np.zeros(0, np.int64),) * 3
    # content features: mean title embedding of every known work published in the venue (PCA 32)
    vm = meta[meta.venue_id.isin(vi) & (meta.title.fillna("").str.len() > 0)]
    emb = embed([f"t::{i}" for i in vm.id], vm.title.tolist(), cfg.default_text, "title", cfg, device)
    sums = np.zeros((len(keep), emb.shape[1]), dtype=np.float64)
    np.add.at(sums, vm.venue_id.map(vi).to_numpy(), emb)
    cnt = np.bincount(vm.venue_id.map(vi).to_numpy(), minlength=len(keep))
    mean = sums / np.maximum(cnt, 1)[:, None]
    from sklearn.decomposition import PCA

    content = PCA(n_components=min(32, len(keep) - 1, mean.shape[1]), random_state=0).fit_transform(mean)
    vrec = venues.set_index("id")
    return {"venue_ids": keep, "names": [vrec.loc[v, "name"] for v in keep], "src": src, "dst": dst, "w": w,
            "out_total": np.array([out_tot[v] for v in keep]), "in_total": np.array([in_tot[v] for v in keep]),
            "n_titles": cnt, "content": content.astype(np.float32),
            "works_count": np.array([int(vrec.loc[v, "works_count"]) for v in keep])}


# ====================================================================== checks
def check_temporal_edges(ei: np.ndarray, years: np.ndarray) -> None:
    assert (years[ei[0]] <= years[ei[1]]).all(), "citation edge from a later paper"


def check_hetero(het: dict, years: np.ndarray, t0: int) -> None:
    for name in ("author", "venue", "keyword"):
        r = het[f"rev_safe_{name}"]
        assert (years[r[0]] < t0).all(), f"rev_safe_{name} carries a paper published >= T0"
        mp = het[f"mp_t0_{name}"]
        assert (years[mp[0]] < t0).all(), f"mp_t0_{name} violates the T0 rule"
        mp = het[f"mp_temporal_{name}"]
        assert (years[mp[0]] < years[mp[1]]).all(), f"mp_temporal_{name} violates year(Q) < year(P)"


# ====================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = get_config(args.profile, device=args.device)
    setup_logging("build_graph", cfg)
    device = resolve_device(cfg)
    env = write_env_json(cfg)
    logger.info("Device %s (%s)", device, env.get("gpu", "cpu"))
    t_start = time.time()

    nodes, meta = load_nodes(cfg)
    authors = pd.read_parquet(cfg.raw_dir / "authors.parquet")
    venues = pd.read_parquet(cfg.raw_dir / "venues.parquet")
    logger.info("Nodes: %d (%s); known works: %d", len(nodes), nodes.origin.value_counts().to_dict(), len(meta))

    feats, mc, m_idx = paper_features(nodes, meta, authors, venues, cfg, device)
    ei, eattr, dropped = citation_edges(nodes, mc, m_idx, cfg)
    years = nodes.year.to_numpy()
    check_temporal_edges(ei, years)
    lab = labels(nodes, cfg)
    texts = [doc_text(t, a) for t, a in zip(nodes.title, nodes.abstract)]
    text_emb = {m: embed(nodes.id.tolist(), texts, m, "doc", cfg, device) for m in TEXT_MODELS}

    graph = {
        "ids": nodes.id.tolist(), "titles": nodes.title.tolist(), "origin": nodes.origin.tolist(),
        "doc_id": nodes.doc_id.tolist(), "venue_name": nodes.venue_name.fillna("").tolist(),
        "topic_group": nodes.topic_group.fillna("").tolist(), "field": nodes.field.fillna("").tolist(),
        "year": torch.as_tensor(years), "edge_index": torch.as_tensor(ei), "edge_attr": torch.as_tensor(eattr),
        **{f"x_{k}": torch.as_tensor(v) for k, v in feats.items()},
        **{f"x_text_{m}": torch.as_tensor(v) for m, v in text_emb.items()},
        **{k: torch.as_tensor(v) for k, v in lab.items()},
    }
    torch.save(graph, cfg.processed_dir / "graph.pt")

    het = build_hetero(nodes, feats, ei, authors, venues, cfg, device)
    check_hetero(het, years, cfg.t0)
    torch.save({k: (torch.as_tensor(v) if isinstance(v, np.ndarray) else v) for k, v in het.items()},
               cfg.processed_dir / "hetero.pt")

    vg = build_venue_graph(nodes, meta, venues, cfg, device)
    torch.save({k: (torch.as_tensor(v) if isinstance(v, np.ndarray) else v) for k, v in vg.items()},
               cfg.processed_dir / "venue_graph.pt")

    # ---- statistics for summary.md
    deg = np.bincount(ei[1], minlength=len(nodes))
    stats = {
        "nodes": int(len(nodes)), "nodes_by_origin": nodes.origin.value_counts().to_dict(),
        "known_works": int(len(meta)), "edges": int(ei.shape[1]), "edges_dropped_ref_later": int(dropped),
        "abstract_share": float((nodes.abstract.str.len() > 0).mean()),
        "author_records": int(len(authors)), "venue_records": int(len(venues)),
        "years_nodes": {str(k): int(v) for k, v in nodes.year.value_counts().sort_index().items()},
        "sample_by_group": nodes[nodes.in_sample].topic_group.value_counts().to_dict(),
        "fields_sample": nodes[nodes.in_sample].field.value_counts().head(15).to_dict(),
    }
    for k in cfg.horizons:
        m = lab[f"m{k}"].astype(bool)
        stats[f"labels_k{k}"] = {
            "n": int(m.sum()), "by_year": {str(y): int(c) for y, c in zip(*np.unique(years[m], return_counts=True))},
            "citations_mean": float(lab[f"c{k}"][m].mean()) if m.any() else None,
            "citations_median": float(np.median(lab[f"c{k}"][m])) if m.any() else None,
            "zero_share": float((lab[f"c{k}"][m] == 0).mean()) if m.any() else None,
            "top_positive_rate": float(lab[f"top{k}"][m].mean()) if m.any() else None,
            "in_degree_mean": float(deg[m].mean()) if m.any() else None,
            "in_degree_median": float(np.median(deg[m])) if m.any() else None,
            "in_degree_zero_share": float((deg[m] == 0).mean()) if m.any() else None,
        }
    stats["hetero"] = {
        "authors": int(len(het["author_ids"])), "venues": int(len(het["venue_ids"])),
        "keywords": int(len(het["keywords"])),
        **{f"edges_{k}": int(v.shape[1]) for k, v in het.items() if isinstance(v, np.ndarray) and v.ndim == 2
           and v.shape[0] == 2},
    }
    stats["venue_graph"] = {"venues": len(vg["venue_ids"]), "edges": int(len(vg["w"])),
                            "links": int(vg["w"].sum()) if len(vg["w"]) else 0,
                            "self_links": int(vg["w"][vg["src"] == vg["dst"]].sum()) if len(vg["w"]) else 0}
    stats["build_seconds"] = round(time.time() - t_start, 1)
    stats["device"] = str(device)
    (cfg.processed_dir / "stats.json").write_text(json.dumps(stats, indent=1, default=int))
    logger.info("Graph: %d nodes, %d edges (%d dropped); labels k3=%d k5=%d; built in %.0f s",
                len(nodes), ei.shape[1], dropped, stats["labels_k3"]["n"], stats["labels_k5"]["n"],
                stats["build_seconds"])


if __name__ == "__main__":
    main()
