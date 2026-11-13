"""Revision graph (rev1): augmented node set, time-safe author and venue statistics, selection metadata.

``python build_rev1.py --profile full``  ->  ``data/processed/<profile>/rev1/{graph.pt, hetero.pt, stats.json}``

The outputs of ``build_graph.py`` (submitted version) are not touched. Differences:

- **Node set.** The submitted nodes come first, in the same order, followed by the shared-reference candidates that
  the 60,000-node cap had excluded. Every reference node carries the number of sample / seed papers that cite it
  and published before each cutoff year, so that the submitted graph, the uncapped graph, other caps and the graphs
  restricted to a prediction date are node masks of one graph (``graph_variants.py``).
- **Author statistics.** ``counts_by_year.cited_by_count`` of an OpenAlex author is keyed by the publication year of
  the cited works, so the submitted "citations before y" (kept here as the ``hind`` variant) include citations
  received after y. The ``exact`` variant sums, over the author's works published before y, the citations each
  work received before y (works-level yearly counts are keyed by citing year).
- **Venue statistics.** Same defect; the exact value cannot be obtained for venues, so the ``exact`` variant uses
  the citations received before y by the venue's works that are in the local corpus (all known works).
- **Heterogeneous graph.** Yearly tables of the entity statistics are stored so that entity snapshots can be built
  for any year (rolling representation), together with SciBERT keyword embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time

import numpy as np
import pandas as pd
import torch

from build_graph import (VENUE_TYPES, YEAR_HI, YearlyCounts, build_hetero, check_hetero, check_temporal_edges,
                         citation_edges, doc_text, embed, label_complete, labels, load_nodes, paper_features,
                         scatter)
from config import TEXT_MODELS, Config, add_common_args, get_config, resolve_device, setup_logging

logger = logging.getLogger("build_rev1")

GRID0, GRID1 = 1950, YEAR_HI + 1          # the "before year y" tables cover y = GRID0 .. GRID1
SNAP_YEARS = list(range(2017, 2024))      # years for which entity snapshots can be built
CUTOFFS = list(range(2017, 2025))         # prediction dates T for which the citer counts are stored


def rev1_dir(cfg: Config):
    d = cfg.processed_dir / "rev1"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ====================================================================== nodes
def load_nodes_aug(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Submitted nodes (same order) followed by the reference candidates cut by the cap; metadata of known works;
    the candidate table."""
    raw = cfg.raw_dir
    nodes, meta = load_nodes(cfg)
    cand = pd.read_parquet(raw / "rev1_ref_candidates.parquet")
    extra = set(cand.loc[~cand.in_submitted, "id"]) - set(nodes.id)
    refs = pd.read_parquet(raw / "refs.parquet")
    texts = pd.read_parquet(raw / "rev1_cut_ref_texts.parquet").drop_duplicates("id").set_index("id")
    rn = refs[refs.id.isin(extra) & refs.year.notna()].copy()
    rn["abstract"] = rn.id.map(texts.abstract).fillna("")
    rn["keywords"] = rn.id.map(texts.keywords).apply(
        lambda k: list(k) if k is not None and not isinstance(k, float) else [])
    rn["origin"], rn["in_sample"], rn["doc_id"], rn["stratum"] = "ref", False, "", ""
    rn["year"] = rn.year.astype(int)
    rn = rn.reindex(columns=nodes.columns).sort_values("id")
    nodes["submitted"] = True
    rn["submitted"] = False
    out = pd.concat([nodes, rn], ignore_index=True)
    assert out.id.is_unique
    return out, meta, cand


def clip_years(years_lists) -> list:
    """Years beyond the covered range (database artefacts such as 10000) are moved to the last covered year, so
    that the subtraction rule removes their counts instead of treating them as past citations."""
    return [[min(int(y), YEAR_HI) for y in ys] for ys in years_lists]


# ====================================================================== time-safe entity statistics
def before_table(ent: np.ndarray, wyear: np.ndarray, widx: np.ndarray, yc: YearlyCounts, n_ent: int,
                 first_year: int) -> tuple[np.ndarray, np.ndarray]:
    """``cit[e, j]``: citations received before year ``GRID0 + j`` by the works of entity ``e`` published before
    that year; ``cnt[e, j]``: number of those works. One row of ``ent`` / ``wyear`` / ``widx`` per (entity, work)."""
    width = GRID1 - GRID0 + 1
    cit, cnt = np.zeros((n_ent, width)), np.zeros((n_ent, width))
    for j in range(width):
        y = GRID0 + j
        m = wyear < y
        if m.any():
            cit[:, j] = np.bincount(ent[m], weights=yc.before(widx[m], max(y, first_year)), minlength=n_ent)
            cnt[:, j] = np.bincount(ent[m], minlength=n_ent)
    return cit, cnt


def author_exact(cfg: Config, authors: pd.DataFrame, dirs=("author_works",)) -> tuple[np.ndarray, np.ndarray, dict]:
    """Citations received before each year by the works of every author (rows aligned with ``authors``), and the
    mask of the authors whose works were listed."""
    shards = sorted(p for d in dirs for p in (cfg.raw_dir / d).glob("part-*.parquet"))
    a_idx = {a: i for i, a in enumerate(authors.id)}
    listed = np.zeros(len(authors), dtype=bool)
    info = {"shards": len(shards), "unattributed": 0, "pages": 0}
    frames = []
    for p in shards:
        meta = json.loads(p.with_suffix(".json").read_text())
        info["unattributed"] += meta["unattributed"]
        info["pages"] += meta["pages"]
        listed[[a_idx[a] for a in meta["authors"] if a in a_idx]] = True
        frames.append(pd.read_parquet(p))
    if not frames:
        return np.zeros((len(authors), GRID1 - GRID0 + 1)), listed, info
    w = pd.concat(frames, ignore_index=True)
    w = w[w.year.notna()]
    pairs = w[["work_id", "authors"]].explode("authors").drop_duplicates()
    works = w.drop_duplicates("work_id").reset_index(drop=True)
    w_idx = {x: i for i, x in enumerate(works.work_id)}
    yc = YearlyCounts(works.cited_by_count, clip_years(works.cby_years), works.cby_counts, cfg.cby_first_year)
    pairs = pairs[pairs.authors.isin(a_idx)]
    ent = pairs.authors.map(a_idx).to_numpy(np.int64)
    widx = pairs.work_id.map(w_idx).to_numpy(np.int64)
    wyear = works.year.to_numpy(np.int64)[widx]
    cit, _ = before_table(ent, wyear, widx, yc, len(authors), cfg.cby_first_year)
    info.update(works=int(len(works)), pairs=int(len(pairs)), authors_listed=int(listed.sum()),
                authors=int(len(authors)))
    return cit, listed, info


def venue_corpus(cfg: Config, venues: pd.DataFrame, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Citations received before each year by the known works of every venue published before that year, and the
    number of those works (local corpus: sample, seeds and all their references)."""
    v_idx = {v: i for i, v in enumerate(venues.id)}
    m = meta[meta.venue_id.isin(v_idx)].reset_index(drop=True)
    yc = YearlyCounts(m.cited_by_count, clip_years(m.cby_years), m.cby_counts, cfg.cby_first_year)
    ent = m.venue_id.map(v_idx).to_numpy(np.int64)
    return before_table(ent, m.year.to_numpy(np.int64), np.arange(len(m)), yc, len(venues), cfg.cby_first_year)


def col(years) -> np.ndarray:
    return np.clip(np.asarray(years) - GRID0, 0, GRID1 - GRID0)


def paper_features_exact(nodes: pd.DataFrame, authors: pd.DataFrame, venues: pd.DataFrame, a_cit: np.ndarray,
                         a_listed: np.ndarray, v_cit: np.ndarray, v_cnt: np.ndarray, cfg: Config,
                         device) -> tuple[np.ndarray, np.ndarray]:
    """Author block (9 columns) and venue block (4 columns) with statistics dated before the publication year.

    Author columns: max, mean, first-author and last-author log citations before y (authors whose works were
    listed); max and mean log works before y; log team size; share of authors with a record; share of authors with
    listed works. Venue columns: log corpus citations before y; log works before y; log corpus citations per corpus
    work; record flag.
    """
    n, years = len(nodes), nodes.year.to_numpy()
    a_idx = {a: i for i, a in enumerate(authors.id)}
    aw = YearlyCounts(authors.works_count, clip_years(authors.cby_years), authors.cby_works, cfg.cby_first_year)
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
    pos, last = np.array(pos), np.array(last, dtype=bool)
    ex = a_listed[cols]
    c = np.log1p(a_cit[cols, col(years[rows])]).astype(np.float32)
    w = np.log1p(aw.before(cols, years[rows])).astype(np.float32)
    t = lambda x: torch.as_tensor(x, device=device)  # noqa: E731
    R, C, W = t(rows), t(c), t(w)
    E, first_m, last_m = t(ex), t(ex & (pos == 0)), t(ex & last)
    n_auth = np.array([len(x) if x is not None else 0 for x in nodes.author_ids], dtype=np.float32)
    denom = t(np.maximum(n_auth, 1))
    ones = torch.ones_like(W)
    author_f = torch.stack([
        scatter(C[E], R[E], n, "amax"), scatter(C[E], R[E], n, "mean"),
        scatter(C[first_m], R[first_m], n, "amax"), scatter(C[last_m], R[last_m], n, "amax"),
        scatter(W, R, n, "amax"), scatter(W, R, n, "mean"),
        t(np.log1p(n_auth)), scatter(ones, R, n, "sum") / denom, scatter(ones[E], R[E], n, "sum") / denom,
    ], 1).cpu().numpy().astype(np.float32)

    v_idx = {v: i for i, v in enumerate(venues.id)}
    vw = YearlyCounts(venues.works_count, clip_years(venues.cby_years), venues.cby_works, cfg.cby_first_year)
    vi = nodes.venue_id.map(v_idx)
    has_v = vi.notna().to_numpy()
    venue_f = np.zeros((n, 4), dtype=np.float32)
    if has_v.any():
        ii = vi[has_v].astype(int).to_numpy()
        j = col(years[has_v])
        cb, cn, wb = v_cit[ii, j], v_cnt[ii, j], vw.before(ii, years[has_v])
        venue_f[has_v] = np.column_stack([np.log1p(cb), np.log1p(wb), np.log1p(cb / np.maximum(cn, 1)),
                                          np.ones(len(ii))])
    return author_f, venue_f


# ====================================================================== selection metadata of the reference nodes
def selection_meta(nodes: pd.DataFrame, cand: pd.DataFrame, cfg: Config) -> dict[str, np.ndarray]:
    """Per node: number of sample / seed citers published before each cutoff, total citers, laboratory-reference
    flag with the first year a laboratory paper cites it, and two ranks among the shared references (the submitted
    one, ties broken by identifier, and a fair one, ties broken by a hash)."""
    n = len(nodes)
    idx = {p: i for i, p in enumerate(nodes.id)}
    is_ref = (nodes.origin == "ref").to_numpy()
    before = np.zeros((n, len(CUTOFFS)), dtype=np.int32)
    n_citers = np.zeros(n, dtype=np.int32)
    lab_ref = np.zeros(n, dtype=bool)
    for r in cand.itertuples():
        i = idx.get(r.id)
        if i is None:
            continue
        ys = np.asarray(r.citer_years)
        n_citers[i] = len(ys)
        before[i] = [(ys < T).sum() for T in CUTOFFS]
        lab_ref[i] = r.lab_ref
    seeds = pd.read_parquet(cfg.raw_dir / "seeds.parquet")
    sw = pd.read_parquet(cfg.raw_dir / "seed_works.parquet")
    lab_ids = set(seeds.loc[seeds.corpus == "lab", "openalex_id"])
    lab_first = np.full(n, 9999, dtype=np.int32)
    for y, refs in zip(sw.loc[sw.id.isin(lab_ids), "year"], sw.loc[sw.id.isin(lab_ids), "refs"]):
        for r in refs:
            i = idx.get(r)
            if i is not None and pd.notna(y):
                lab_first[i] = min(lab_first[i], int(y))
    shared = np.flatnonzero(is_ref & ~lab_ref)
    ids = nodes.id.to_numpy()
    rank_sub = np.full(n, -1, dtype=np.int64)
    rank_fair = np.full(n, -1, dtype=np.int64)
    order = sorted(shared, key=lambda i: (-n_citers[i], ids[i]))
    rank_sub[order] = np.arange(len(order))
    h = {i: hashlib.sha1(f"{cfg.sample_seed}:{ids[i]}".encode()).hexdigest() for i in shared}
    order = sorted(shared, key=lambda i: (-n_citers[i], h[i]))
    rank_fair[order] = np.arange(len(order))
    return {"sel_citers_before": before, "sel_n_citers": n_citers, "sel_lab_ref": lab_ref,
            "sel_lab_first_year": lab_first, "sel_rank_submitted": rank_sub, "sel_rank_fair": rank_fair}


# ====================================================================== heterogeneous graph additions
def hetero_tables(het: dict, nodes: pd.DataFrame, authors: pd.DataFrame, venues: pd.DataFrame, a_cit, a_listed,
                  v_cit, v_cnt, cfg: Config, device) -> dict:
    """Yearly entity statistics (one column per snapshot year) and SciBERT keyword embeddings."""
    snap = np.array(SNAP_YEARS)
    aw = YearlyCounts(authors.works_count, clip_years(authors.cby_years), authors.cby_works, cfg.cby_first_year)
    vw = YearlyCounts(venues.works_count, clip_years(venues.cby_years), venues.cby_works, cfg.cby_first_year)
    ai, vi = np.arange(len(authors)), np.arange(len(venues))
    first_year = np.array([min([y for y, w in zip(ys, ws) if w > 0 and 1900 <= y <= YEAR_HI] or [cfg.t0])
                           for ys, ws in zip(authors.cby_years, authors.cby_works)])
    vt = np.zeros((len(venues), len(VENUE_TYPES) + 1), dtype=np.float32)
    for i, t in enumerate(venues.type):
        vt[i, VENUE_TYPES.index(t) if t in VENUE_TYPES else len(VENUE_TYPES)] = 1
    years = nodes.year.to_numpy()
    kw_e = np.asarray(het["rev_all_keyword"])  # paper -> keyword
    kw_tab = np.zeros((len(het["keywords"]), len(snap)), dtype=np.float32)
    for j, y in enumerate(snap):
        if kw_e.size:
            np.add.at(kw_tab[:, j], kw_e[1][years[kw_e[0]] < y], 1)
    kws = het["keywords"]
    out = {
        "snap_years": snap,
        "author_cit_tab": a_cit[:, col(snap)].astype(np.float32), "author_listed": a_listed,
        "author_works_tab": np.column_stack([aw.before(ai, y) for y in snap]).astype(np.float32),
        "author_first_year": first_year.astype(np.float32),
        "venue_cit_tab": v_cit[:, col(snap)].astype(np.float32), "venue_cnt_tab": v_cnt[:, col(snap)].astype(np.float32),
        "venue_works_tab": np.column_stack([vw.before(vi, y) for y in snap]).astype(np.float32),
        "venue_type": vt, "keyword_tab": kw_tab,
        "x_kw_text_specter": np.asarray(het["x_keyword"])[:, :-1].astype(np.float32),
        "x_kw_text_scibert": embed([f"kw::{k}" for k in kws], kws, "scibert", "keyword", cfg, device) if kws else
        np.zeros((0, 768), np.float32),
    }
    return out


# ====================================================================== confirmatory holdout
def add_fresh(cfg: Config, nodes: pd.DataFrame, meta: pd.DataFrame, cand: pd.DataFrame):
    """Append the fresh 2022 papers (origin ``fresh``) and the reference nodes they bring in under the rule of the
    primary protocol. The existing nodes keep their positions; the venue corpus statistics stay those of the
    original corpus."""
    from collect_rev1 import FRESH_YEAR, fresh_extra_refs

    raw = cfg.raw_dir
    fresh = pd.read_parquet(raw / "fresh_sample.parquet")
    assert not set(fresh.id) & set(nodes.id) and not set(fresh.id) & set(meta.id), "fresh paper already collected"
    assert (fresh.year == FRESH_YEAR).all()
    extra = fresh_extra_refs(cfg, fresh)
    refs = pd.read_parquet(raw / "refs.parquet")
    texts = pd.read_parquet(raw / "fresh_ref_texts.parquet").drop_duplicates("id").set_index("id")
    rn = refs[refs.id.isin(extra) & refs.year.notna() & ~refs.id.isin(set(nodes.id))].copy()
    rn["abstract"] = rn.id.map(texts.abstract).fillna("")
    rn["keywords"] = rn.id.map(texts.keywords)
    rn["origin"] = "ref"
    fr = fresh.copy()
    fr["origin"] = "fresh"
    add = pd.concat([fr.reindex(columns=nodes.columns), rn.reindex(columns=nodes.columns).sort_values("id")],
                    ignore_index=True)
    add["keywords"] = add.keywords.apply(lambda k: list(k) if k is not None and not isinstance(k, float) else [])
    add["abstract"] = add.abstract.fillna("")
    add["year"] = add.year.astype(int)
    add["in_sample"], add["submitted"], add["doc_id"], add["stratum"] = False, False, "", ""
    add.loc[add.origin == "fresh", "topic_group"] = fr.topic_group.to_numpy()
    out = pd.concat([nodes, add], ignore_index=True)
    assert out.id.is_unique
    cols = ["id", "year", "cited_by_count", "cby_years", "cby_counts", "venue_id", "title"]
    fresh_refs = pd.read_parquet(raw / "fresh_refs.parquet")
    meta2 = pd.concat([meta, fr[cols], fresh_refs[cols]], ignore_index=True).drop_duplicates("id")
    meta2 = meta2[meta2.year.notna()].reset_index(drop=True)
    meta2["year"] = meta2.year.astype(int)
    cand2 = pd.concat([cand, pd.DataFrame([dict(id=r, n_citers=len(ys), citer_years=sorted(ys), lab_ref=False,
                                                in_submitted=False) for r, ys in extra.items()])], ignore_index=True)
    return out, meta2, cand2, fresh


# ====================================================================== complete first-hop neighbourhood
def add_leaves(cfg: Config, nodes: pd.DataFrame) -> pd.DataFrame:
    """Append every known reference that is not yet a node as a leaf: title-only text, no reference list (so that
    only the first hop is added). Used for the feature-parity analysis: with the leaves, a GNN sees every reference
    with metadata of a paper, like the MLP with all-reference aggregates."""
    refs = pd.read_parquet(cfg.raw_dir / "refs.parquet")
    leaf = refs[~refs.id.isin(set(nodes.id)) & refs.year.notna()].copy()
    leaf["abstract"], leaf["origin"] = "", "ref"
    leaf["refs"] = [[] for _ in range(len(leaf))]
    leaf = leaf.reindex(columns=nodes.columns).sort_values("id")
    leaf["keywords"] = [[] for _ in range(len(leaf))]
    leaf["year"] = leaf.year.astype(int)
    leaf["in_sample"], leaf["submitted"], leaf["doc_id"], leaf["stratum"] = False, False, "", ""
    out = pd.concat([nodes.assign(leaf=False), leaf.assign(leaf=True)], ignore_index=True)
    assert out.id.is_unique
    return out


# ====================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--holdout", action="store_true",
                    help="build the graph that also contains the fresh 2022 sample (-> rev1_holdout/)")
    ap.add_argument("--firsthop", action="store_true",
                    help="build the graph with every known reference as a leaf node (-> rev1_firsthop/, no "
                         "heterogeneous graph)")
    args = ap.parse_args()
    cfg = get_config(args.profile, device=args.device)
    setup_logging("build_rev1", cfg)
    device = resolve_device(cfg)
    t_start = time.time()

    nodes, meta, cand = load_nodes_aug(cfg)
    meta_corpus = meta
    authors = pd.read_parquet(cfg.raw_dir / "authors.parquet")
    venues = pd.read_parquet(cfg.raw_dir / "venues.parquet")
    sample = pd.read_parquet(cfg.raw_dir / "sample.parquet", columns=["id", "sampled_topic", "subfield"])
    author_dirs = ("author_works",)
    if args.holdout:
        nodes, meta, cand, fresh = add_fresh(cfg, nodes, meta, cand)
        authors = pd.concat([authors, pd.read_parquet(cfg.raw_dir / "fresh_authors.parquet")],
                            ignore_index=True).drop_duplicates("id").reset_index(drop=True)
        venues = pd.concat([venues, pd.read_parquet(cfg.raw_dir / "fresh_venues.parquet")],
                           ignore_index=True).drop_duplicates("id").reset_index(drop=True)
        sample = pd.concat([sample, fresh[["id", "sampled_topic", "subfield"]]], ignore_index=True)
        author_dirs = ("author_works", "author_works_fresh")
    if args.firsthop:
        nodes = add_leaves(cfg, nodes)
    logger.info("Nodes: %d (%d submitted, %d added); known works: %d", len(nodes), int(nodes.submitted.sum()),
                int((~nodes.submitted).sum()), len(meta))

    feats, mc, m_idx = paper_features(nodes, meta, authors, venues, cfg, device)     # submitted ("hind") features
    a_cit, a_listed, a_info = author_exact(cfg, authors, author_dirs)
    v_cit, v_cnt = venue_corpus(cfg, venues, meta_corpus)
    author_f, venue_f = paper_features_exact(nodes, authors, venues, a_cit, a_listed, v_cit, v_cnt, cfg, device)
    logger.info("Author works: %s", a_info)

    if args.holdout:
        # The nodes of the training graph keep exactly the features they had when the models were trained. Without
        # this, a reference node whose author also wrote a fresh paper would gain an author record, and the
        # references of the fresh papers would add metadata to the aggregates of older papers.
        g0 = torch.load(rev1_dir(cfg) / "graph.pt", weights_only=False)
        n0 = len(g0["ids"])
        assert nodes.id.tolist()[:n0] == g0["ids"], "the training-graph nodes must come first, in the same order"
        feats["base"][:n0], feats["refagg"][:n0] = g0["x_base"].numpy(), g0["x_refagg"].numpy()
        feats["biblio"][:n0] = g0["x_biblio_hind"].numpy()
        author_f[:n0], venue_f[:n0] = g0["x_author_exact"].numpy(), g0["x_venue_exact"].numpy()
        del g0
    ei, eattr, dropped = citation_edges(nodes, mc, m_idx, cfg)
    years = nodes.year.to_numpy()
    check_temporal_edges(ei, years)
    lab = labels(nodes, cfg)
    if args.holdout:                       # labels of the fresh papers; top-10 % threshold of the original 2022 cohort
        fm = (nodes.origin == "fresh").to_numpy()
        for k in cfg.horizons:
            ok = fm & np.array([label_complete(int(y), k, cfg) for y in nodes.year])
            ref_m = lab[f"m{k}"].astype(bool) & (years == 2022)
            if ok.any() and ref_m.any():
                thr = np.quantile(lab[f"c{k}"][ref_m], 1 - cfg.top_share)
                lab[f"top{k}"][ok] = (lab[f"c{k}"][ok] >= thr) & (lab[f"c{k}"][ok] > 0)
            lab[f"m{k}_fresh"] = ok
    texts = [doc_text(t, a) for t, a in zip(nodes.title, nodes.abstract)]
    text_emb = {m: embed(nodes.id.tolist(), texts, m, "doc", cfg, device) for m in TEXT_MODELS}
    sel = selection_meta(nodes, cand, cfg)

    graph = {
        "ids": nodes.id.tolist(), "titles": nodes.title.tolist(), "origin": nodes.origin.tolist(),
        "doc_id": nodes.doc_id.tolist(), "venue_name": nodes.venue_name.fillna("").tolist(),
        "topic_group": nodes.topic_group.fillna("").tolist(), "field": nodes.field.fillna("").tolist(),
        "sampled_topic": nodes.id.map(dict(zip(sample.id, sample.sampled_topic))).fillna("").tolist(),
        "subfield": nodes.id.map(dict(zip(sample.id, sample.subfield))).fillna("").tolist(),
        "docs": texts, "submitted": torch.as_tensor(nodes.submitted.to_numpy()),
        "n_refs": torch.as_tensor(nodes.n_refs.fillna(0).to_numpy(np.float32)),
        "year": torch.as_tensor(years), "edge_index": torch.as_tensor(ei), "edge_attr": torch.as_tensor(eattr),
        "x_base": torch.as_tensor(feats["base"]), "x_refagg": torch.as_tensor(feats["refagg"]),
        "x_biblio_hind": torch.as_tensor(feats["biblio"]),
        "x_author_exact": torch.as_tensor(author_f), "x_venue_exact": torch.as_tensor(venue_f),
        **{f"x_text_{m}": torch.as_tensor(v) for m, v in text_emb.items()},
        **{k: torch.as_tensor(v) for k, v in lab.items()},
        **{k: torch.as_tensor(v) for k, v in sel.items()},
        "cutoffs": CUTOFFS, "author_works_complete": bool(a_info.get("authors_listed", 0) > 0),
    }
    out = rev1_dir(cfg)
    if args.holdout or args.firsthop:
        out = cfg.processed_dir / ("rev1_holdout" if args.holdout else "rev1_firsthop")
        out.mkdir(parents=True, exist_ok=True)
    torch.save(graph, out / "graph.pt")
    if args.firsthop:
        (out / "stats.json").write_text(json.dumps({"nodes": int(len(nodes)), "leaves": int(nodes.leaf.sum()),
                                                    "edges": int(ei.shape[1])}, indent=1))
        logger.info("first-hop graph: %d nodes (%d leaves), %d edges", len(nodes), int(nodes.leaf.sum()), ei.shape[1])
        return

    het = build_hetero(nodes, feats, ei, authors, venues, cfg, device)                 # submitted-style relations
    check_hetero(het, years, cfg.t0)
    het.update(hetero_tables(het, nodes, authors, venues, a_cit, a_listed, v_cit, v_cnt, cfg, device))
    torch.save({k: (torch.as_tensor(v) if isinstance(v, np.ndarray) else v) for k, v in het.items()},
               out / "hetero.pt")

    sub = nodes.submitted.to_numpy()
    stats = {
        "nodes": int(len(nodes)), "nodes_submitted": int(sub.sum()), "nodes_added": int((~sub).sum()),
        "edges": int(ei.shape[1]), "edges_submitted": int((sub[ei[0]] & sub[ei[1]]).sum()),
        "edges_dropped_ref_later": int(dropped), "author_works": a_info,
        "authors": int(len(authors)), "venues": int(len(venues)), "keywords": int(len(het["keywords"])),
        "labels": {f"k{k}": {str(y): int(c) for y, c in zip(*np.unique(years[lab[f'm{k}'].astype(bool)],
                                                                         return_counts=True))} for k in cfg.horizons},
        "build_seconds": round(time.time() - t_start, 1),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1, default=int))
    logger.info("rev1 graph: %d nodes, %d edges; built in %.0f s", len(nodes), ei.shape[1], stats["build_seconds"])


if __name__ == "__main__":
    main()
