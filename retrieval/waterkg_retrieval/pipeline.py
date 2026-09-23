"""Hybrid retrieval: entity-graph channel fused with the dense (BGE) channel.

1. LLM1 splits the question into entity groups (OR inside a group, AND across groups).
2. LLM2 maps each entity word either to taxonomy categories (category channel: every paper
   mentioning any entity in the category) or to the evidence channel (full-text match on the
   entity surface forms recorded in the graph).
3. Graph candidates = papers that satisfy every group. They are ranked by the number of group
   members they hit (``graph_rank="hits"``, ties broken by dense score) or by dense score alone
   (``graph_rank="dense"``).
4. Final list = top ``graph_k`` graph papers, then filled up to ``k`` with the best dense
   papers not already selected.
"""
from __future__ import annotations

import time

from .dense import DenseIndex
from .store import GraphStore


class HybridRetriever:
    def __init__(self, store_path, vector_dir, device=None):
        self.store = GraphStore(store_path)
        self.dense = DenseIndex(vector_dir, device=device)

    def plan(self, question: str, model: str | None = None) -> dict:
        from .llm import plan_query

        return plan_query(question, self.store.taxonomy_text, model=model)

    def graph_candidates(self, plan: dict) -> tuple[dict[str, int], list[dict]]:
        """Return {paper_id: member hits} for papers hitting every group, plus a per-member trace."""
        per_group, member_sets, trace = [], [], []
        for g in plan["groups"]:
            gp = set()
            for m in g["members"]:
                out = plan["llm2"].get(f"{m['entity_type']}::{m['source_text']}") or {"layer": "evidence"}
                if out.get("layer") == "category":
                    ids, cats = set(), []
                    for match in out.get("matches") or []:
                        cid = self.store.resolve_category(match.get("entity_type") or m["entity_type"],
                                                          match.get("l3", ""), match.get("l2", ""))
                        if cid:
                            cats.append(cid)
                            ids |= self.store.category_papers(cid)
                    channel = f"category:{','.join(cats) or 'none'}"
                else:
                    ids = self.store.evidence_papers(m["source_text"])
                    channel = "evidence"
                gp |= ids
                member_sets.append(ids)
                trace.append({"group": g["label"], "member": m["source_text"], "channel": channel,
                              "papers": len(ids)})
            per_group.append(gp)
        hits = {}
        if per_group:
            for pid in set.intersection(*per_group):
                hits[pid] = sum(pid in s for s in member_sets)
        return hits, trace

    def search(self, question: str, k: int = 10, graph_k: int = 5, dense_k: int = 30,
               graph_rank: str = "hits", plan: dict | None = None, model: str | None = None) -> dict:
        t0 = time.perf_counter()
        plan = plan or self.plan(question, model=model)
        t_plan = time.perf_counter() - t0

        hits, trace = self.graph_candidates(plan)
        all_scores = self.dense.scores(self.dense.encode(question))
        dense_top = self.dense.top(all_scores, dense_k)

        def dscore(pid):
            return self.dense.score_of(all_scores, pid)

        if graph_rank == "hits":
            graph_ranked = sorted(hits, key=lambda p: (-hits[p], -dscore(p), p))
        elif graph_rank == "dense":
            graph_ranked = sorted(hits, key=lambda p: (-dscore(p), p))
        else:
            raise ValueError("graph_rank must be 'hits' or 'dense'")

        picked = [(p, "graph") for p in graph_ranked[:graph_k]]
        used = {p for p, _ in picked}
        for p, _ in dense_top:
            if len(picked) >= k:
                break
            if p not in used:
                picked.append((p, "dense"))
                used.add(p)

        meta = self.store.paper_meta([p for p, _ in picked])
        results = [{"rank": i + 1, "paper_id": p, "source": src, **meta.get(p, {}),
                    "graph_hits": hits.get(p), "dense_score": round(dscore(p), 4)}
                   for i, (p, src) in enumerate(picked)]
        return {"question": question, "results": results, "graph_candidates": len(hits),
                "graph_trace": trace, "plan": plan,
                "timings": {"plan_s": round(t_plan, 2), "total_s": round(time.perf_counter() - t0, 2)}}
