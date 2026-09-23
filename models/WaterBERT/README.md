---
license: apache-2.0
language:
  - en
base_model: allenai/scibert_scivocab_uncased
pipeline_tag: fill-mask
library_name: transformers
tags:
  - wastewater
  - water-treatment
  - environmental-science
  - scibert
  - domain-adaptive-pretraining
---

# WaterBERT

WaterBERT is a BERT-base encoder for wastewater- and water-treatment literature. It was
obtained by continuing masked-language-model (MLM) pretraining of
[SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased) (`scivocab`, uncased) on a
domain corpus of water-treatment research text. The vocabulary is SciBERT's (31,090 tokens);
only the weights were adapted.

It is the shared backbone of [WaterBERT-NER](../WaterBERT-NER) and
[WaterBERT-RE](../WaterBERT-RE), and is intended as a starting point for fine-tuning on
water-treatment NLP tasks (entity recognition, relation extraction, classification, retrieval).

## Usage

```python
from transformers import pipeline

fill = pipeline("fill-mask", model="Mudi12137/WaterBERT")
fill("The Fenton process generates hydroxyl radicals from hydrogen peroxide and [MASK] ions.")
# top prediction: "iron"
```

For fine-tuning, load it like any BERT checkpoint:

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Mudi12137/WaterBERT")
model = AutoModelForTokenClassification.from_pretrained("Mudi12137/WaterBERT", num_labels=13)
```

## Training

| Setting | Value |
|---|---|
| Initialisation | `allenai/scibert_scivocab_uncased` |
| Objective | masked language modelling |
| Epochs / steps | 30 / 940,440 (this checkpoint is the final step) |
| Effective batch size | 160 (80 per device x 2 gradient-accumulation steps, 1 GPU) |
| Optimiser | AdamW, learning rate 1e-4, linear decay, warmup ratio 0.048, weight decay 0.01 |
| Precision | bf16 |
| Architecture | BERT-base: 12 layers, hidden size 768, 110M parameters, max length 512 |

The pretraining corpus and its construction are described in the accompanying paper.

## Limitations

- English only, lower-cased input (the tokenizer lower-cases automatically).
- The corpus is research literature (mostly abstracts); performance on other genres such as
  plant operation logs or regulations has not been evaluated.

## License

Apache-2.0, the same as the SciBERT base model.

## Citation

See [CITATION.cff](https://github.com/Mudi12138/WaterBERT/blob/main/CITATION.cff) in the project repository.
