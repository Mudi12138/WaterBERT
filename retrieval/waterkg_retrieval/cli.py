"""Command line: build the store once, then search.

    python -m waterkg_retrieval build  --graph-dir WaterKG/graph --out waterkg.sqlite
    python -m waterkg_retrieval search "your question" --store waterkg.sqlite --vectors WaterKG/graph/vectors
"""
from __future__ import annotations

import argparse
import json


def main(argv=None):
    ap = argparse.ArgumentParser(prog="waterkg_retrieval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the SQLite graph store from the released parquet tables")
    b.add_argument("--graph-dir", required=True, help="directory that contains parquet/")
    b.add_argument("--out", required=True)

    s = sub.add_parser("search", help="hybrid graph + dense search")
    s.add_argument("question")
    s.add_argument("--store", required=True)
    s.add_argument("--vectors", required=True, help="directory with the paper embeddings")
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--graph-k", type=int, default=5)
    s.add_argument("--dense-k", type=int, default=30)
    s.add_argument("--graph-rank", choices=["hits", "dense"], default="hits")
    s.add_argument("--plan", help="JSON file with a precomputed plan (skips the LLM calls)")
    s.add_argument("--save-plan", help="write the LLM plan to this JSON file")
    s.add_argument("--model", help="LLM model name (default: $WATERKG_LLM_MODEL or gpt-5-mini)")
    s.add_argument("--device")
    s.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        from .store import build_store

        build_store(args.graph_dir, args.out)
        print(f"built {args.out}")
        return

    from .pipeline import HybridRetriever

    r = HybridRetriever(args.store, args.vectors, device=args.device)
    plan = json.load(open(args.plan, encoding="utf-8")) if args.plan else None
    res = r.search(args.question, k=args.k, graph_k=args.graph_k, dense_k=args.dense_k,
                   graph_rank=args.graph_rank, plan=plan, model=args.model)
    if args.save_plan:
        json.dump(res["plan"], open(args.save_plan, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    print(f"graph candidates: {res['graph_candidates']:,}")
    for t in res["graph_trace"]:
        print(f"  [{t['group']}] {t['member']!r} -> {t['channel']} ({t['papers']:,} papers)")
    for x in res["results"]:
        print(f"{x['rank']:>2}. {x['paper_id']} [{x['source']}] {x.get('doi') or '-'} "
              f"({x.get('year') or '?'}, {x.get('journal') or '?'}) dense={x['dense_score']}")


if __name__ == "__main__":
    main()
