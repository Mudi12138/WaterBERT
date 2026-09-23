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

# WaterBERT-RE

Relation classification for wastewater- and water-treatment literature, fine-tuned from
[WaterBERT](../WaterBERT). Given a text with two marked entities, it predicts one of four
labels:

| Label | Head → tail | Meaning |
|---|---|---|
| `removes` | Process or Reactor → Pollutant | the process/reactor removes or degrades the pollutant |
| `removal_rate` | Pollutant → Value | the value is a removal efficiency/rate reported for the pollutant |
| `has` | Process or Reactor → Treatment_Parameter | the parameter describes the process/reactor |
| `no_relation` | any valid pair | no relation stated |

Only these entity-type pairs are valid inputs (`label_config.json`):
(Process, Pollutant), (Reactor, Pollutant), (Pollutant, Value), (Process, Parameter),
(Reactor, Parameter).

## Input format

Wrap the head entity in `[E1_X]…[/E1_X]` and the tail in `[E2_Y]…[/E2_Y]`, where `X`/`Y` is
the type code. The markers are already registered as special tokens in the tokenizer.

| Code | Entity type |
|---|---|
| `POL` | Pollutant |
| `WTP` | Wastewater_Treatment_Process |
| `RCT` | Reactor |
| `TRP` | Treatment_Parameter |
| `VAL` | Value (a numeric value, e.g. `92%`) |

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

name = "Mudi12137/WaterBERT-RE"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name).eval()

text = "[E1_WTP]Ozonation[/E1_WTP] effectively removed [E2_POL]carbamazepine[/E2_POL] from secondary effluent."
with torch.no_grad():
    probs = model(**tok(text, return_tensors="pt", truncation=True)).logits.softmax(-1)[0]
print(model.config.id2label[int(probs.argmax())])   # removes
```

Insert markers from right to left (by character offset) so earlier offsets stay valid. Pass the
whole abstract or passage (up to 512 tokens), not only the sentence, so that cross-sentence
relations remain possible.

## Where the entity spans come from

`POL`, `WTP`, `RCT` and `TRP` spans can come from [WaterBERT-NER](../WaterBERT-NER).
`VAL` spans (numbers with units or percentages) are **not** produced by WaterBERT-NER; they
must come from a separate value detector — a regular expression for percentages is enough
for `removal_rate`.

## Training

Fine-tuned from WaterBERT on entity pairs drawn from manually annotated abstracts
(128,221 candidate pairs, of which 16,160 carry a relation), with a class-weighted loss.
Details are in the accompanying paper.

## License

Apache-2.0.
