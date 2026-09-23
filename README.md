# WaterBERT

Language models, a knowledge graph and a hybrid retrieval system for water- and
wastewater-treatment research literature.

| Component | What it is | Where |
|---|---|---|
| **WaterBERT** | SciBERT further pretrained on water-treatment literature | [`models/WaterBERT`](models/WaterBERT) · Hugging Face `Mudi12137/WaterBERT` |
| **WaterBERT-NER** | six-type named-entity recognition | [`models/WaterBERT-NER`](models/WaterBERT-NER) · Hugging Face `Mudi12137/WaterBERT-NER` |
| **WaterBERT-RE** | relation classification (`removes`, `removal_rate`, `has`); used to build WaterKG | [`models/WaterBERT-RE`](models/WaterBERT-RE) · Hugging Face `Mudi12137/WaterBERT-RE` |
| **WaterBERT-RE-v1** | first-generation fine-grained relation classification (`targets`, `has_efficiency`, `indicates_removal_of`, `conditioned_by`, `has_value`) | [`models/WaterBERT-RE-v1`](models/WaterBERT-RE-v1) · Hugging Face `Mudi12137/WaterBERT-RE-v1` |
| **WaterKG** | 118,131 entities, 672,380 papers, 4.1 M entity–paper links, 0.9 M relation evidence | [`graph/`](graph) · Hugging Face dataset `Mudi12137/WaterKG` |
| **Hybrid retrieval** | entity-graph + dense (BGE) literature search | [`retrieval/`](retrieval) |

Entity types: `Pollutant`, `Wastewater_Treatment_Process`, `Reactor`, `Treatment_Parameter`,
`Microorganism`, `Dosed_Material`.

## Quick start

**Models**

```python
from transformers import pipeline

ner = pipeline("token-classification", model="Mudi12137/WaterBERT-NER", aggregation_strategy="simple")
ner("Ozonation removed 90% of carbamazepine in a membrane bioreactor effluent.")
```

**Graph**

```python
import pandas as pd

entities = pd.read_parquet("WaterKG/graph/parquet/entities.parquet")
relations = pd.read_parquet("WaterKG/graph/parquet/relations.parquet")
```

A Neo4j bulk-import script is in [`graph/neo4j`](graph/neo4j); the table schema is documented
in [`graph/README.md`](graph/README.md).

**Retrieval**

```bash
pip install -e retrieval/
python -m waterkg_retrieval build --graph-dir WaterKG/graph --out waterkg.sqlite
export WATERKG_LLM_API_KEY=...
python -m waterkg_retrieval search "adsorption of heavy metals and emerging contaminants" \
  --store waterkg.sqlite --vectors WaterKG/graph/vectors
```

See [`retrieval/README.md`](retrieval/README.md).

## Repository layout

```
models/      model cards (weights are on Hugging Face)
graph/       graph documentation, Neo4j scripts, 20-row samples of every table
retrieval/   the waterkg-retrieval Python package
```

## Data and licenses

- Code and models: Apache-2.0 ([LICENSE](LICENSE)); the models inherit SciBERT's Apache-2.0.
- Graph data and embeddings: CC BY 4.0.
- No titles, abstracts or author lists are redistributed. Papers are identified by an
  anonymous release id plus DOI, year and journal. The paper embeddings were computed from
  titles and abstracts; they cannot be decoded back into the text, but embedding-inversion
  research shows that partial reconstruction is possible in principle.

## Citation

If you use these resources, please cite the paper (see [CITATION.cff](CITATION.cff)).
