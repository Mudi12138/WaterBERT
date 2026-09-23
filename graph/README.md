# WaterKG — wastewater-treatment literature knowledge graph (v1.0)

WaterKG links **118,131 canonical entities** of six types to **672,380 research papers** on
water and wastewater treatment. It has two layers:

- **Entity layer** — which entities each paper mentions, with every mention's surface form
  and character offsets, and a two-level (L3/L2) taxonomy of the entities.
- **Relation layer** — `removes`, `has` and `removal_rate` relations extracted by
  [WaterBERT-RE](https://huggingface.co/Mudi12137/WaterBERT-RE), with per-paper evidence and graph-level aggregates.

Entities were extracted from titles and abstracts with [WaterBERT-NER](https://huggingface.co/Mudi12137/WaterBERT-NER),
normalised and merged into canonical entities, and classified into 57 L3 / 124 L2 categories.

## Download

The data files (Parquet, CSV.gz, Neo4j import files and paper embeddings; 1.8 GB) are in the
Hugging Face dataset [`Mudi12137/WaterKG`](https://huggingface.co/datasets/Mudi12137/WaterKG):

```python
from huggingface_hub import snapshot_download

snapshot_download("Mudi12137/WaterKG", repo_type="dataset", local_dir="WaterKG/graph")
```

The GitHub repository [Mudi12138/WaterBERT](https://github.com/Mudi12138/WaterBERT) holds this
documentation, the hybrid retrieval package, the Neo4j scripts and a 20-row sample of every
table (`graph/sample/`).

## Contents

```
graph/
├── parquet/          one file per table (typed, recommended)
├── csv/              the same tables as gzip-compressed CSV
├── neo4j/            header files, import_neo4j.sh, constraints.cypher
├── vectors/          BGE-large paper embeddings for the retrieval system
├── summary.json      counts and build statistics
└── checksums.sha256
```

| Table | Rows | One row is |
|---|---:|---|
| `entities` | 118,131 | a canonical entity |
| `entity_aliases` | 146,905 | a surface form of an entity from the curated entity list |
| `categories` | 181 | an L3 (57) or L2 (124) taxonomy category |
| `papers` | 672,380 | a paper |
| `edges_entity_category` | 94,622 | entity → L2 category |
| `edges_category_parent` | 124 | L2 → L3 category |
| `edges_entity_paper` | 4,079,774 | entity mentioned in a paper |
| `mentions` | 8,800,205 | one entity mention in a paper |
| `relation_evidence` | 922,511 | one extracted relation in one paper |
| `relations` | 204,374 | a relation between two entities, aggregated over papers |

## Identifiers

| Prefix | Example | Object |
|---|---|---|
| `E` | `E00001` | canonical entity |
| `P` | `P0000001` | paper (anonymous release id) |
| `<TYPE>-L3-nn` / `<TYPE>-L2-nnn` | `POL-L2-003` | taxonomy category; `TYPE` is POL, WTP, RCT, TRP, MIC or DOS |
| `M` | `M00000001` | mention |
| `V` | `V00000001` | relation evidence |
| `R` | `R0000001` | aggregated relation |

Papers carry DOI, publication year and journal only. 614,235 papers (91.4%) have a DOI;
378 rows share a DOI with another row because the source database holds two records for
them (`doi_shared = true`).

## Tables

### `entities`

| Column | Description |
|---|---|
| `entity_id` | canonical entity id |
| `name` | canonical name (lower-case normalised form) |
| `entity_type` | `Pollutant`, `Wastewater_Treatment_Process`, `Reactor`, `Treatment_Parameter`, `Microorganism`, `Dosed_Material` |
| `l3_category_id`, `l2_category_id` | taxonomy categories (empty when unclassified) |
| `l3_en`, `l2_en` / `l3_zh`, `l2_zh` | category names in English / Chinese |
| `classification_status` | `classified` (94,622) or `classification_unknown` (23,509) |
| `domain_flag` | `in_domain`, `out_of_domain` (category "Out of domain") or `unclassified` |
| `n_surface_forms` | number of distinct surface forms merged into the entity |
| `mention_count` | mentions across the corpus |
| `paper_count` | papers mentioning the entity |

### `entity_aliases`
`entity_id`, `alias` — the curated surface forms of each entity (canonical name included).
Every other spelling observed in the corpus is in `mentions.surface_text`.

### `categories`
`category_id`, `level` (`L3`/`L2`), `entity_type`, `name_en`, `name_zh`, `parent_id` (the L3
of an L2).

### `papers`
`paper_id`, `doi` (lower-case), `year`, `journal`, `doi_shared`.

### `edges_entity_paper`
`entity_id`, `paper_id`, `mentions` (number of mentions of the entity in the paper).

### `mentions`

| Column | Description |
|---|---|
| `mention_id` | mention id |
| `paper_id`, `entity_id`, `entity_type` | where and what; `entity_id` is empty for the 1,726,204 mentions (19.6%) whose surface form was not merged into a canonical entity |
| `surface_text` | the span as written in the paper |
| `char_start`, `char_end` | character offsets into the abstract text of the source record |
| `ner_confidence` | WaterBERT-NER confidence |

Offsets refer to the abstract as exported from the bibliographic database; abstracts are not
redistributed, and abstracts obtained elsewhere may differ slightly.

### `relation_evidence`

| Column | Description |
|---|---|
| `evidence_id`, `paper_id` | evidence id and paper |
| `relation` | `removes`, `has` or `removal_rate` |
| `re_confidence` | WaterBERT-RE probability of the predicted label |
| `head_*`, `tail_*` | for each endpoint: `entity_id`, `type`, `text`, `start`, `end`, `ner_confidence`, `link` |
| `value_percent` | for `removal_rate`: the value as a number when it is a single percentage in 0–100 |

For `removal_rate` the tail is a value (`tail_type = Value`, `tail_entity_id` empty). A
removal efficiency can be attached to a `removes` relation by joining on `paper_id` and the
pollutant's `entity_id`.

`*_link` records how the endpoint was linked to the canonical entity: `span_exact`,
`span_overlap`, `alias_exact`, `alias_loose` or `value`.

### `relations`

| Column | Description |
|---|---|
| `relation_id` | aggregated relation id |
| `head_entity_id`, `relation_type`, `tail_entity_id` | `REMOVES` (process/reactor → pollutant) or `HAS` (process/reactor → parameter) |
| `n_evidence`, `n_papers` | supporting evidence rows and distinct papers |
| `mean_re_confidence`, `max_re_confidence` | over the evidence rows |

## Loading

```python
import pandas as pd

ents = pd.read_parquet("WaterKG/graph/parquet/entities.parquet")
rel = pd.read_parquet("WaterKG/graph/parquet/relations.parquet")
name = ents.set_index("entity_id")["name"]

# processes that remove a pollutant, ranked by number of papers
tc = ents.loc[(ents.name == "tetracycline") & (ents.entity_type == "Pollutant"), "entity_id"].iloc[0]
rm = rel[(rel.relation_type == "REMOVES") & (rel.tail_entity_id == tc)]
print(rm.assign(process=rm.head_entity_id.map(name)).nlargest(10, "n_papers")[["process", "n_papers"]])
```

### Neo4j

```bash
bash WaterKG/graph/neo4j/import_neo4j.sh WaterKG/graph/csv waterkg     # Neo4j 5, offline import into a new database
cypher-shell -d waterkg -f WaterKG/graph/neo4j/constraints.cypher
```

This loads `(:Entity)`, `(:Paper)` and `(:Category)` nodes and the `BELONGS_TO`,
`SUBCLASS_OF`, `MENTIONED_IN`, `REMOVES` and `HAS` relationships. `mentions` and
`relation_evidence` stay as tables.

```cypher
MATCH (p:Entity)-[r:REMOVES]->(t:Entity {name: 'tetracycline'})
RETURN p.name, r.n_papers ORDER BY r.n_papers DESC LIMIT 10;
```

## License

The graph data are released under CC BY 4.0. Bibliographic identifiers (DOI, year, journal)
are factual metadata; no titles, abstracts or authors are included.
