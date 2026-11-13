"""Graph variants of the revision, as node and edge masks of the augmented graph built by ``build_rev1.py``.

``submitted``     the graph of the submitted article (60,000-node cap, ties at the cap broken by identifier)
``uncapped``      every reference cited by at least two sample or seed papers
``cap<N>k``       cap of N thousand paper nodes; ties at the cap broken by a hash instead of the identifier
``cut<T>``        reference selection restricted to the information available at prediction date T: a shared
                  reference is a node if at least two sample or seed papers published before T cite it; a laboratory
                  reference is a node if a laboratory paper published before T cites it
``cut<T>lso``     same node set with a leave-self-out edge rule: the edge reference -> paper is kept only if the
                  reference would be selected without counting that paper among its citers. Training papers
                  (published before T) contribute to the selection of their own references, test papers cannot;
                  this rule removes that asymmetry by taking the self-count away from the training papers.
``cut<T>sym``     the other way of removing the asymmetry: the selection rule is applied when a paper enters the
                  graph, so a paper published from T on is linked to a reference cited by itself and by at least
                  one paper published before T. Only the reference list of the new paper and papers published
                  before T are used.

Sample and seed papers are nodes of every variant; papers published from T on are removed from the training graph
by the evaluation protocol, not here.
"""

from __future__ import annotations

import re

import torch


def _is_ref(g: dict) -> torch.Tensor:
    return torch.as_tensor([o == "ref" for o in g["origin"]], device=g["year"].device)


def node_mask(g: dict, variant: str, min_citers: int = 2) -> torch.Tensor:
    ref = _is_ref(g)
    if variant == "submitted":
        return g["submitted"].bool()
    if variant == "uncapped":
        return torch.ones_like(ref)
    m = re.fullmatch(r"cap(\d+)k", variant)
    if m:
        lab = ref & g["sel_lab_ref"].bool()
        budget = int(m.group(1)) * 1000 - int((~ref).sum()) - int(lab.sum())
        return ~ref | lab | (ref & (g["sel_rank_fair"] >= 0) & (g["sel_rank_fair"] < budget))
    m = re.fullmatch(r"cut(\d{4})(lso|sym)?", variant)
    if m:
        T = int(m.group(1))
        j = g["cutoffs"].index(T)
        lab_ok = g["sel_lab_ref"].bool() & (g["sel_lab_first_year"] < T)
        need = min_citers - 1 if m.group(2) == "sym" else min_citers
        return ~ref | lab_ok | (g["sel_citers_before"][:, j] >= need)
    raise ValueError(f"unknown graph variant: {variant}")


def edge_mask(g: dict, variant: str, min_citers: int = 2) -> torch.Tensor:
    """Edges of the variant: both endpoints are nodes of it and, for ``cut<T>lso``, the leave-self-out rule holds."""
    ei = g["edge_index"]
    nm = node_mask(g, variant, min_citers)
    keep = nm[ei[0]] & nm[ei[1]]
    m = re.fullmatch(r"cut(\d{4})(lso|sym)", variant)
    if m:
        T = int(m.group(1))
        j = g["cutoffs"].index(T)
        ref = _is_ref(g)
        src, dst = ei[0], ei[1]
        lab_ok = g["sel_lab_ref"].bool() & (g["sel_lab_first_year"] < T)
        if m.group(2) == "lso":   # a citing paper published before T is one of the counted citers: discount it
            need = min_citers + ((~ref[dst]) & (g["year"][dst] < T)).long()
        else:                     # a citing sample / seed paper published from T on counts for its own references
            need = min_citers - ((~ref[dst]) & (g["year"][dst] >= T)).long()
        ok = ~ref[src] | lab_ok[src] | (g["sel_citers_before"][src, j] >= need)
        keep = keep & ok
    return keep
