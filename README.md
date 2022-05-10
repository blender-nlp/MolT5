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

We used the open-sourced [t5x](https://github.com/google-research/t5x) framework for both pretraining and finetuning MolT5-based models.

For pre-training MolT5-based models, please first go over [this document](https://github.com/google-research/t5x/blob/main/docs/usage/pretrain.md). In our work, our pretraining task is a mixture of [c4_v220_span_corruption](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/tasks.py#L45) and also our own task called `zinc_span_corruption`. The pretraining mixture is called `zinc_and_c4_mix`. The code snippet below illustrates how to define `zinc_and_c4_mix` (e.g., you can just add this code snippet to [tasks.py](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/tasks.py)).
```python
...
import tensorflow.compat.v2 as tf
...
seqio.TaskRegistry.add(
    'zinc_span_corruption',
    source=seqio.TFExampleDataSource(
        split_to_filepattern={
            'test': # Path to zinc_smiles_test.tfrecords,
            'validation': # Path to zinc_smiles_val.tfrecords,
            'train': # Path to zinc_smiles_train.tfrecords,
        },
        feature_description={
            'text': tf.io.FixedLenFeature([], dtype=tf.string),
        }),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                'inputs': None,
                'targets': 'text'
            }),
        seqio.preprocessors.tokenize,
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])

seqio.MixtureRegistry.add('zinc_and_c4_mix', [('zinc_span_corruption', 1),
                                              ('c4_v220_span_corruption', 1)])
)
```

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
