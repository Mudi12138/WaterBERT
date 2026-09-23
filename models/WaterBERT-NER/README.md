---
license: apache-2.0
language:
  - en
base_model: Mudi12137/WaterBERT
pipeline_tag: token-classification
library_name: transformers
tags:
  - wastewater
  - water-treatment
  - named-entity-recognition
---

# WaterBERT-NER

Named-entity recognition for wastewater- and water-treatment literature, fine-tuned from
[WaterBERT](https://huggingface.co/Mudi12137/WaterBERT). It tags six entity types with a BIO scheme (13 labels):

| Entity type | Covers (examples) |
|---|---|
| `Pollutant` | contaminants and water-quality targets — *ammonium, sulfamethoxazole, COD, microplastics* |
| `Wastewater_Treatment_Process` | treatment processes and unit operations — *adsorption, Fenton process, anaerobic digestion* |
| `Reactor` | reactors, units and devices — *sequencing batch reactor, membrane bioreactor, constructed wetland* |
| `Treatment_Parameter` | operating, reaction and material parameters — *pH, hydraulic retention time, specific surface area* |
| `Microorganism` | microorganisms and microbial groups — *Nitrosomonas, anammox bacteria, microalgae* |
| `Dosed_Material` | added materials, catalysts, adsorbents and reagents — *biochar, persulfate, PAC* |

This is the model used to build the [WaterKG](https://github.com/Mudi12138/WaterBERT) entity graph.

## Usage

```python
from transformers import pipeline

ner = pipeline("token-classification", model="Mudi12137/WaterBERT-NER", aggregation_strategy="simple")
ner("A sequencing batch reactor inoculated with Nitrosomonas removed 95% of ammonium and "
    "sulfamethoxazole at pH 7.5 after dosing biochar.")
# Reactor: sequencing batch reactor | Microorganism: nitrosomonas | Pollutant: ammonium,
# sulfamethoxazole | Treatment_Parameter: ph | Dosed_Material: biochar
```

Inputs longer than 512 tokens are truncated; split long documents into sentences or passages.
The tokenizer lower-cases input, so returned `word` strings are lower-case — use the
`start`/`end` character offsets to recover the original spelling.

## Labels

`O`, then `B-`/`I-` for `Pollutant`, `Wastewater_Treatment_Process`, `Reactor`,
`Treatment_Parameter`, `Microorganism`, `Dosed_Material` (see `label_map.json`).

## Training

Fine-tuned from WaterBERT on manually annotated water-treatment abstracts; the annotation
guideline and data are described in the accompanying paper.

## Limitations

- Numeric values and removal efficiencies are not entity types of this model. The relation
  model [WaterBERT-RE](https://huggingface.co/Mudi12137/WaterBERT-RE) expects value spans from a separate value detector.
- Boundaries of long compound names (e.g. *"rotating disc electrocoagulation system"*) and
  the type of cross-category words (e.g. *adsorption* as a process vs. a parameter) are the
  most common sources of error.

## License

Apache-2.0.
