# waterkg-retrieval

Hybrid literature retrieval over the WaterKG graph: an **entity-graph channel** fused with a
**dense channel** (BGE-large paper embeddings). It needs only the released graph tables and
paper embeddings — no paper text.

## How it works

1. **LLM1** splits the question into *entity groups*: members of one group are alternatives
   (OR); different groups must all be satisfied (AND).
2. **LLM2** (one call per entity word, in parallel) decides whether the word names a taxonomy
   category (e.g. *heavy metals* → L2 *Heavy metals*) or is a concrete entity.
3. **Graph channel.** Category words expand to every paper mentioning any entity in that
   category; concrete words are matched against the entity surface forms recorded in the graph
   (SQLite FTS5). Papers must hit every group; they are ranked by how many group members they
   hit (ties broken by dense similarity), or by dense similarity alone (`--graph-rank dense`).
4. **Dense channel.** The question is embedded with `BAAI/bge-large-en-v1.5` (retrieval
   instruction prefix, CLS pooling, L2-normalised) and scored against all paper embeddings.
5. **Fusion.** The top 5 graph papers are kept, then the list is filled to 10 with the best
   dense papers not already selected.

The result is a ranked list of paper ids with DOI, year and journal.

## Install

```bash
pip install -e retrieval/
```

## Data

Download the WaterKG graph release (see the main README) so that you have:

```
WaterKG/graph/
├── parquet/                 # entities, categories, papers, edges_entity_paper, mentions, ...
└── vectors/
    ├── paper_embeddings_bge-large-en-v1.5_fp16.npy
    └── paper_embeddings_rows.parquet
```

Build the SQLite store once (a few minutes, about 1–2 GB on disk):

```bash
python -m waterkg_retrieval build --graph-dir WaterKG/graph --out waterkg.sqlite
```

## Search

The planning steps call an OpenAI-compatible chat API:

```bash
export WATERKG_LLM_API_KEY=...            # or OPENAI_API_KEY
export WATERKG_LLM_MODEL=gpt-5-mini       # default; the model used in the paper
# export WATERKG_LLM_BASE_URL=...         # optional, for other OpenAI-compatible providers

python -m waterkg_retrieval search \
  "Which studies removed both nitrate and phosphate in a single biological reactor?" \
  --store waterkg.sqlite --vectors WaterKG/graph/vectors
```

Useful options:

- `--save-plan plan.json` stores the LLM output; `--plan plan.json` replays it without calling
  the LLM, so a search can be reproduced exactly.
- `--json` prints the full result: ranked papers, per-member channel trace, plan and timings.
- `--graph-rank dense`, `--k`, `--graph-k`, `--dense-k` change the fusion.

From Python:

```python
from waterkg_retrieval import HybridRetriever

r = HybridRetriever("waterkg.sqlite", "WaterKG/graph/vectors")
res = r.search("photocatalytic degradation of endocrine disruptors with immobilized catalysts")
for x in res["results"]:
    print(x["rank"], x["paper_id"], x["source"], x["doi"])
```

