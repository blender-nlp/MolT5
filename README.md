# MolT5: Translation between Molecules and Natural Language
Associated repository for "[Translation between Molecules and Natural Language](https://arxiv.org/abs/2204.11817)".


<p align="center">
<img src="https://raw.githubusercontent.com/blender-nlp/MolT5/main/molt5.png" width="70%">
</p>

Table of Contents
 - [Model checkpoints](#model-checkpoints)
 - [Pretraining and Finetuning (MolT5-based models)](#pretraining-and-finetuning-molt5-based-models)
 - [Citation](#citation)

### Model checkpoints

All of our HuggingFace checkpoints are located [here](https://huggingface.co/laituan245).

Pretrained MolT5-based checkpoints include:

+ [molt5-small](https://huggingface.co/laituan245/molt5-small) (~77 million parameters)
+ [molt5-base](https://huggingface.co/laituan245/molt5-base) (~250 million parameters)
+ [molt5-large](https://huggingface.co/laituan245/molt5-large) (~800 million parameters)

You can also easily find our fine-tuned caption2smiles and smiles2caption models. For example, [molt5-large-smiles2caption](https://huggingface.co/laituan245/molt5-large-smiles2caption) is a molt5-large model that has been further fine-tuned for the task of molecule captioning (i.e., smiles2caption).

Example usage for molecule captioning (i.e., smiles2caption):

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-large-smiles2caption", model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-large-smiles2caption')

input_text = 'C1=CC2=C(C(=C1)[O-])NC(=CC2=O)C(=O)O'
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids, num_beams=5, max_length=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Example usage for molecule generation (i.e., caption2smiles):

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-large-caption2smiles", model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-large-caption2smiles')

input_text = 'The molecule is a monomethoxybenzene that is 2-methoxyphenol substituted by a hydroxymethyl group at position 4. It has a role as a plant metabolite. It is a member of guaiacols and a member of benzyl alcohols.'
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids, num_beams=5, max_length=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Pretraining and Finetuning (MolT5-based models)


### Citation
If you found our work useful, please cite:
```bibtex
@article{edwards2022translation,
  title={Translation between Molecules and Natural Language},
  author={Edwards, Carl and Lai, Tuan and Ros, Kevin and Honke, Garrett and Ji, Heng},
  journal={arXiv preprint arXiv:2204.11817},
  year={2022}
}
```
