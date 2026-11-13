"""Additional OpenAlex downloads of the revision (rev1). Only new files are written under ``data/raw/<profile>/``;
nothing produced by ``collect.py`` for the submitted version is overwritten.

Steps (``python collect_rev1.py --profile full --step cutrefs,issn,authorworks``):

``cutrefs``      abstract, keywords and topics of the shared-reference candidates that the 60,000-node cap
                 excluded (-> ``rev1_cut_ref_texts.parquet``), so that the uncapped and the cutoff-restricted
                 graphs can be built.
``issn``         ISSN-L of every venue of the venue graph (-> ``rev1_venue_issn.parquet``).
``authorworks``  works of every author of the sample and seed papers, with the citations of each work by citing
                 year (-> ``author_works/part-*.parquet``).
``fresh``        confirmatory holdout, to be run only after the models are frozen (see ``prereg/``): a new random
                 sample of 2022 articles of the same 48 topics, drawn with another seed and disjoint from every
                 paper already collected, with their references, authors (records and works) and venues
                 (-> ``fresh_*.parquet``, ``author_works_fresh/``).

Why ``authorworks``: ``counts_by_year.cited_by_count`` of an OpenAlex *author* (or source) is keyed by the
publication year of the cited works, not by the year of the citations. "Total minus the years from y on" therefore
gives the citations received up to today by works published before y, which is not known at publication time. The
yearly counts of a *work* are keyed by citing year, so the citations an author had received before year y are
recomputed here from the author's own works.

The listings are paged with a cursor and stored compacted, one parquet shard per batch of authors; a shard that
exists is never requested again, so the step resumes after the daily credit budget is exhausted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from collect import (AUTHOR_SELECT, FULL_SELECT, LIGHT_SELECT, SOURCE_SELECT, TEXT_SELECT, BudgetExhausted, OpenAlex,
                     chunks, compact_counts, compact_work, short_id, write_report)
from config import Config, add_common_args, get_config, setup_logging

logger = logging.getLogger("collect_rev1")

AUTHOR_WORKS_SELECT = "id,publication_year,cited_by_count,counts_by_year,authorships"
AUTHOR_BATCH = 50            # authors per request (OR filter)
MAX_AUTHOR_WORKS = 2000      # authors with more works are left out (merged identities, consortium lists)
SAMPLE_BEFORE = 2022         # last sampled year: no labelled paper needs works published from this year on


# ====================================================================== cut reference candidates
def shared_candidates(cfg: Config) -> pd.DataFrame:
    """Every reference cited by >= ``min_ref_citers`` sample or seed papers, with its number of citers, the year by
    which it had that many citers, and whether the submitted graph kept it."""
    seeds = pd.read_parquet(cfg.raw_dir / "seeds.parquet")
    sw = pd.read_parquet(cfg.raw_dir / "seed_works.parquet")
    sample = pd.read_parquet(cfg.raw_dir / "sample.parquet")
    known = set(pd.read_parquet(cfg.raw_dir / "refs.parquet", columns=["id"]).id)
    base = set(sample.id) | set(sw.id)
    lab_ids = set(seeds.loc[seeds.corpus == "lab", "openalex_id"])
    citing = pd.concat([sample[["id", "year", "refs"]], sw[["id", "year", "refs"]]]).drop_duplicates("id")
    citer_years: dict[str, list[int]] = {}
    for year, refs in zip(citing.year, citing.refs):
        for r in set(refs):
            citer_years.setdefault(r, []).append(int(year))
    lab_refs = {r for refs_ in sw.loc[sw.id.isin(lab_ids), "refs"] for r in refs_} & known
    kept = set(json.loads((cfg.raw_dir / "node_refs.json").read_text()))
    rows = []
    for r, ys in citer_years.items():
        if r in base or r not in known:
            continue
        if len(ys) < cfg.min_ref_citers and r not in lab_refs:
            continue
        ys = sorted(ys)
        rows.append(dict(id=r, n_citers=len(ys), citer_years=ys, lab_ref=r in lab_refs, in_submitted=r in kept))
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


def step_cutrefs(cfg: Config, oa: OpenAlex) -> pd.DataFrame:
    oa.step = "rev1_cutrefs"
    cand = shared_candidates(cfg)
    cand.to_parquet(cfg.raw_dir / "rev1_ref_candidates.parquet")
    todo = cand.loc[~cand.in_submitted, "id"].tolist()
    logger.info("Reference candidates: %d (%d in the submitted graph, %d cut by the cap)", len(cand),
                int(cand.in_submitted.sum()), len(todo))
    texts = pd.DataFrame([compact_work(w) for w in oa.by_ids("works", todo, TEXT_SELECT)])
    texts = texts[["id", "abstract", "keywords", "topic_ids", "topic_scores"]].drop_duplicates("id")
    texts.to_parquet(cfg.raw_dir / "rev1_cut_ref_texts.parquet")
    logger.info("Cut references: %d text records of %d requested", len(texts), len(todo))
    return texts


# ====================================================================== venue ISSN
def step_issn(cfg: Config, oa: OpenAlex) -> pd.DataFrame:
    import torch

    oa.step = "rev1_issn"
    vg = torch.load(cfg.processed_dir / "venue_graph.pt", weights_only=False)
    ids = sorted(vg["venue_ids"])
    recs = oa.by_ids("sources", ids, "id,display_name,issn_l,issn,type")
    df = pd.DataFrame([dict(id=short_id(r.get("id")), name=r.get("display_name") or "", issn_l=r.get("issn_l") or "",
                            issn=list(r.get("issn") or []), type=r.get("type") or "") for r in recs])
    df.drop_duplicates("id").to_parquet(cfg.raw_dir / "rev1_venue_issn.parquet")
    logger.info("Venue ISSN: %d records of %d venues", len(df), len(ids))
    return df


# ====================================================================== works of the authors
def author_groups(cfg: Config) -> dict[str, list[str]]:
    """Authors to list, by group: ``lab`` (laboratory papers, all years, for the case study) and ``base`` (sample and
    open-corpus seed papers; only works published before ``SAMPLE_BEFORE`` are needed)."""
    seeds = pd.read_parquet(cfg.raw_dir / "seeds.parquet")
    sw = pd.read_parquet(cfg.raw_dir / "seed_works.parquet")
    sample = pd.read_parquet(cfg.raw_dir / "sample.parquet")
    authors = pd.read_parquet(cfg.raw_dir / "authors.parquet", columns=["id", "works_count"])
    n_works = dict(zip(authors.id, authors.works_count))
    lab_ids = set(seeds.loc[seeds.corpus == "lab", "openalex_id"])
    lab = {a for lst in sw.loc[sw.id.isin(lab_ids), "author_ids"] for a in lst if a}
    base = {a for lst in pd.concat([sample.author_ids, sw.author_ids]) for a in lst if a} - lab
    skipped = sorted(a for a in base | lab if a not in n_works or n_works[a] > MAX_AUTHOR_WORKS)
    (cfg.raw_dir / "author_works_excluded.json").write_text(json.dumps(
        {"max_author_works": MAX_AUTHOR_WORKS, "no_record": [a for a in skipped if a not in n_works],
         "too_many_works": [a for a in skipped if a in n_works]}, indent=1))
    keep = lambda s: sorted(a for a in s if a in n_works and n_works[a] <= MAX_AUTHOR_WORKS)  # noqa: E731
    return {"lab": keep(lab), "base": keep(base)}


def list_author_works(oa: OpenAlex, batch: list[str], before: int | None) -> tuple[list[dict], dict]:
    """All cited works of the authors of ``batch`` (published before ``before`` when given), paged with a cursor."""
    filt = f"author.id:{'|'.join(batch)},cited_by_count:>0"
    if before:
        filt += f",publication_year:<{before}"
    wanted = set(batch)
    rows, cursor, pages, truncated_unmatched = [], "*", 0, 0
    while cursor:
        data = oa.get("/works", {"filter": filt, "select": AUTHOR_WORKS_SELECT, "per-page": 200, "cursor": cursor},
                      cache=False)
        res = data.get("results", [])
        pages += 1
        for w in res:
            ids = [short_id((a.get("author") or {}).get("id")) for a in (w.get("authorships") or [])]
            mine = sorted(wanted.intersection(ids))
            if not mine:  # author lists are truncated to 100 names in listings: the work cannot be attributed
                truncated_unmatched += 1
                continue
            cby = sorted((c["year"], c.get("cited_by_count", 0)) for c in (w.get("counts_by_year") or []))
            rows.append(dict(work_id=short_id(w.get("id")), year=w.get("publication_year"),
                             cited_by_count=int(w.get("cited_by_count") or 0),
                             cby_years=[int(y) for y, _ in cby], cby_counts=[int(c) for _, c in cby], authors=mine))
        cursor = (data.get("meta") or {}).get("next_cursor") if len(res) == 200 else None
    return rows, dict(pages=pages, works=len(rows), unattributed=truncated_unmatched)


def step_authorworks(cfg: Config, oa: OpenAlex, limit: int | None = None, groups: dict | None = None,
                     out_name: str = "author_works") -> None:
    oa.step = "rev1_authorworks"
    out_dir = cfg.raw_dir / out_name
    out_dir.mkdir(exist_ok=True)
    jobs = []
    for group, ids in (groups or author_groups(cfg)).items():
        before = None if group == "lab" else SAMPLE_BEFORE
        for batch in chunks(ids, AUTHOR_BATCH):
            key = hashlib.sha1(f"{group}|{before}|{'|'.join(batch)}".encode()).hexdigest()[:16]
            jobs.append((group, before, batch, out_dir / f"part-{group}-{key}.parquet"))
    todo = [j for j in jobs if not j[3].exists()]
    logger.info("Author works: %d batches, %d already done, %d to fetch", len(jobs), len(jobs) - len(todo), len(todo))
    if limit:
        todo = todo[:limit]

    def one(job) -> dict:
        group, before, batch, path = job
        rows, info = list_author_works(oa, batch, before)
        df = pd.DataFrame(rows, columns=["work_id", "year", "cited_by_count", "cby_years", "cby_counts", "authors"])
        path.with_suffix(".json").write_text(json.dumps(dict(info, group=group, before=before, authors=batch)))
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp)
        os.replace(tmp, path)  # the parquet shard is written last: its presence marks the batch as complete
        return info

    done, pages, t0 = 0, 0, time.time()
    with ThreadPoolExecutor(cfg.workers) as pool:
        for info in pool.map(one, todo):
            done += 1
            pages += info["pages"]
            if done % 20 == 0 or done == len(todo):
                logger.info("  author works: %d/%d batches, %d pages, %.0f s, %s credits left", done, len(todo), pages,
                            time.time() - t0, oa.remaining)


# ====================================================================== confirmatory holdout (fresh 2022 sample)
FRESH_YEAR, FRESH_PER_TOPIC, FRESH_DRAW, FRESH_SEED = 2022, 50, 200, 20261113


def fresh_extra_refs(cfg: Config, fresh: pd.DataFrame) -> dict[str, list[int]]:
    """References of the fresh papers that are not yet candidate nodes but are cited by at least one sample or seed
    paper published before ``FRESH_YEAR``: identifier -> publication years of those earlier citers."""
    raw = cfg.raw_dir
    sample = pd.read_parquet(raw / "sample.parquet", columns=["id", "year", "refs"])
    sw = pd.read_parquet(raw / "seed_works.parquet", columns=["id", "year", "refs"])
    base = set(sample.id) | set(sw.id)
    have = set(pd.read_parquet(raw / "rev1_ref_candidates.parquet", columns=["id"]).id) | base
    known = set(pd.read_parquet(raw / "refs.parquet", columns=["id"]).id)
    wanted = {r for lst in fresh.refs for r in lst} & known - have
    years: dict[str, list[int]] = {}
    for y, refs in zip(pd.concat([sample.year, sw.year]), pd.concat([sample.refs, sw.refs])):
        if pd.isna(y) or int(y) >= FRESH_YEAR:
            continue
        for r in set(refs) & wanted:
            years.setdefault(r, []).append(int(y))
    return years


def step_fresh(cfg: Config, oa: OpenAlex) -> None:
    oa.step = "rev1_fresh"
    raw = cfg.raw_dir
    topics = json.loads((raw / "topics.json").read_text())
    sample = pd.read_parquet(raw / "sample.parquet")
    sw = pd.read_parquet(raw / "seed_works.parquet")
    refs = pd.read_parquet(raw / "refs.parquet", columns=["id", "venue_id"])
    known = set(sample.id) | set(sw.id) | set(refs.id)       # no fresh paper may already be in the collection

    def one(t: dict) -> list[dict]:
        filt = (f"primary_topic.id:{t['topic_id']},publication_year:{FRESH_YEAR},type:article,language:en,"
                f"referenced_works_count:>4")
        data = oa.get("/works", {"filter": filt, "sample": FRESH_DRAW, "seed": FRESH_SEED, "per-page": 200,
                                 "page": 1, "select": FULL_SELECT})
        return data.get("results", [])

    rows, seen = [], set()
    with ThreadPoolExecutor(cfg.workers) as pool:
        for t, res in zip(topics, pool.map(one, topics)):
            n = 0
            for w in res:                                   # order returned by OpenAlex for the fixed seed
                rec = compact_work(w)
                if rec["id"] in known or rec["id"] in seen or n >= FRESH_PER_TOPIC:
                    continue
                seen.add(rec["id"])
                rec.update(sampled_topic=t["topic_id"], topic_group=t["group"])
                rows.append(rec)
                n += 1
            logger.info("  fresh sample, topic %s: %d of %d returned", t["topic_id"], n, len(res))
    fresh = pd.DataFrame(rows)
    fresh.to_parquet(raw / "fresh_sample.parquet")
    logger.info("Fresh sample: %d papers of %d topics", len(fresh), fresh.sampled_topic.nunique())

    ref_ids = sorted({r for lst in fresh.refs for r in lst} - known)
    logger.info("Fresh references: %d new (%d batches)", len(ref_ids), (len(ref_ids) + 49) // 50)
    fr = pd.DataFrame([compact_work(w) for w in oa.by_ids("works", ref_ids, LIGHT_SELECT)]).drop_duplicates("id")
    fr.to_parquet(raw / "fresh_refs.parquet")

    old_auth = set(pd.read_parquet(raw / "authors.parquet", columns=["id"]).id)
    a_ids = sorted({a for lst in fresh.author_ids for a in lst if a} - old_auth)
    fa = pd.DataFrame([compact_counts(r, "authors") for r in oa.by_ids("authors", a_ids, AUTHOR_SELECT)])
    fa.drop_duplicates("id").to_parquet(raw / "fresh_authors.parquet")
    old_ven = set(pd.read_parquet(raw / "venues.parquet", columns=["id"]).id)
    v_ids = sorted((set(fresh.venue_id) | set(fr.venue_id)) - old_ven - {""})
    fv = pd.DataFrame([compact_counts(r, "sources") for r in oa.by_ids("sources", v_ids, SOURCE_SELECT)])
    fv.drop_duplicates("id").to_parquet(raw / "fresh_venues.parquet")
    logger.info("Fresh authors: %d new records; fresh venues: %d new records", len(fa), len(fv))

    # reference nodes to add for the fresh papers: same rule as for the test papers of the primary protocol
    # (cited by the paper itself and by at least one sample or seed paper published before its year)
    extra = fresh_extra_refs(cfg, fresh)
    ft = pd.DataFrame([compact_work(w) for w in oa.by_ids("works", sorted(extra), TEXT_SELECT)])
    ft[["id", "abstract", "keywords", "topic_ids", "topic_scores"]].drop_duplicates("id").to_parquet(
        raw / "fresh_ref_texts.parquet")
    logger.info("Fresh reference nodes to add: %d (%d text records)", len(extra), len(ft))

    n_works = dict(zip(fa.id, fa.works_count)) if len(fa) else {}
    listable = sorted(a for a in a_ids if a in n_works and n_works[a] <= MAX_AUTHOR_WORKS)
    step_authorworks(cfg, oa, groups={"fresh": listable}, out_name="author_works_fresh")


STEPS = ["cutrefs", "issn", "authorworks"]          # "fresh" is run explicitly, after the models are frozen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--step", default="all", help=f"all or comma list of {STEPS}")
    ap.add_argument("--limit", type=int, default=None, help="authorworks: fetch at most this many batches (test)")
    args = ap.parse_args()
    cfg = get_config(args.profile)
    setup_logging("collect_rev1", cfg)
    steps = STEPS if args.step == "all" else args.step.split(",")
    oa = OpenAlex(cfg)
    t_steps: dict[str, float] = {}
    try:
        for s in steps:
            t = time.time()
            logger.info("=== step %s (profile %s)", s, cfg.profile)
            if s == "authorworks":
                step_authorworks(cfg, oa, args.limit)
            elif s == "fresh":
                step_fresh(cfg, oa)
            else:
                globals()[f"step_{s}"](cfg, oa)
            t_steps[f"rev1_{s}"] = time.time() - t
            logger.info("=== step %s done in %.0f s; credits left: %s", s, t_steps[f"rev1_{s}"], oa.remaining)
    except BudgetExhausted as exc:
        logger.error("Stopped: %s. Re-run the same command after the daily reset; finished batches are kept.", exc)
        raise SystemExit(2)
    finally:
        write_report(cfg, oa, t_steps)


if __name__ == "__main__":
    main()
