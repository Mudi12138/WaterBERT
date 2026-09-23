---
license: apache-2.0
language:
  - en
base_model: Mudi12137/WaterBERT
pipeline_tag: text-classification
library_name: transformers
tags:
  - wastewater
  - water-treatment
  - relation-extraction
---

# WaterBERT-RE-v1

Fine-grained relation classification for wastewater- and water-treatment literature,
fine-tuned from [WaterBERT](https://huggingface.co/Mudi12137/WaterBERT). It links treatment
processes, pollutants, removal efficiencies, operating parameters and values, so that a
removal result can be assembled into a structured record (process → efficiency → pollutant,
with its value and conditions).

This is the first-generation relation model. The knowledge graph WaterKG was built with
[WaterBERT-RE](https://huggingface.co/Mudi12137/WaterBERT-RE), which uses a simpler
three-relation schema (`removes`, `removal_rate`, `has`).

## Labels

| Label | Head → tail | Meaning |
|---|---|---|
| `targets` | Process → Pollutant | the process is applied to / removes the pollutant |
| `has_efficiency` | Process → Removal_Efficiency | the efficiency expression belongs to the process |
| `indicates_removal_of` | Removal_Efficiency → Pollutant | the efficiency refers to this pollutant |
| `conditioned_by` | Removal_Efficiency → Treatment_Parameter | the efficiency was obtained under this parameter |
| `has_value` | Treatment_Parameter → Value, Removal_Efficiency → Value | the numeric value of the parameter or efficiency |
| `no_relation` | any valid pair | no relation stated |

Valid entity-type pairs are listed in `label_config.json`.

## Input format

Wrap the head entity in `[E1_X]…[/E1_X]` and the tail in `[E2_Y]…[/E2_Y]`; the markers are
registered as special tokens.

| Code | Entity type | Example span |
|---|---|---|
| `POL` | Pollutant | *carbamazepine* |
| `WTP` | Wastewater_Treatment_Process | *ozonation* |
| `TRP` | Treatment_Parameter | *pH*, *contact time* |
| `REM` | Removal_Efficiency | *removal efficiency*, *degradation rate* |
| `VAL` | Value | *95%*, *7.5* |

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

name = "Mudi12137/WaterBERT-RE-v1"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name).eval()

text = "The [E1_REM]removal efficiency[/E1_REM] of carbamazepine reached [E2_VAL]95%[/E2_VAL] at pH 7."
with torch.no_grad():
    probs = model(**tok(text, return_tensors="pt", truncation=True)).logits.softmax(-1)[0]
print(model.config.id2label[int(probs.argmax())])   # has_value
```

Insert markers from right to left (by character offset) and pass the whole abstract or
passage (up to 512 tokens).

## Where the entity spans come from

`Removal_Efficiency` and `Value` are not entity types of
[WaterBERT-NER](https://huggingface.co/Mudi12137/WaterBERT-NER); they came from an earlier
eight-type NER model. To use this model on new text, supply `REM` spans (efficiency
expressions such as *removal efficiency*, *removal rate*, *degradation*) and `VAL` spans
(numbers, percentages) from your own detector, e.g. keyword and regular-expression rules.

## License

Apache-2.0.
