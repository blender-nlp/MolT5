# Evaluation Code: Translation between Molecules and Natural Language
Task evaluation code for "[Translation between Molecules and Natural Language](https://arxiv.org/abs/2204.11817)".


## Installation
The requirements for the evaluation code conda environment are in environment_eval.yml. An environment can be created using the following commands: 

```
conda env create -n MolTextTranslationEval -f environment_eval.yml python=3.9
conda activate MolTextTranslationEval
python -m spacy download en_core_web_sm
pip install git+https://github.com/samoturk/mol2vec
```

## Input format
The input format should be a tab-separated txt file with three columns and the header 'SMILES ground truth  output' for smiles2caption or 'description	ground truth	output' for caption2smiles. 

## Evaluation Commands

<table>
  <tr>
    <td>Code</td>
    <td>Evaluation</td>
  </tr>
  <tr>
    <td colspan="2">Evaluating SMILES to Caption</td>
  </tr>
  <tr>
    <td>python text_translation_metrics.py --input_file smiles2caption_example.txt</td>
    <td>Evaluate all NLG metrics.</td>
  </tr>
  <tr>
    <td>python text_text2mol_metric.py --input_file smiles2caption_example.txt</td>
    <td>Evaluate Text2Mol metric for caption generation.</td>
  </tr>
  <tr>
    <td>python text_text2mol_metric.py --use_gt</td>
    <td>Evaluate Text2Mol metric for the ground truth.</td>
  </tr>
  <tr>
    <td colspan="2">Evaluating Caption to SMILES</td>
  </tr>
  <tr>
    <td>python mol_translation_metrics.py --input_file caption2smiles_example.txt</td>
    <td>Evaluate BLEU, Exact match, and Levenshtein metrics.</td>
  </tr>
  <tr>
    <td>python fingerprint_metrics.py --input_file caption2smiles_example.txt</td>
    <td>Evaluate fingerprint metrics.</td>
  </tr>
  <tr>
    <td>./mol_text2mol_metric.sh caption2smiles_example.txt</td>
    <td>Evaluate Text2Mol metric for molecule generation.</td>
  </tr>
  <tr>
    <td>python mol_text2mol_metric.py --use_gt</td>
    <td>Evaluate Text2Mol metric for the ground truth.</td>
  </tr>
  <tr>
    <td>python fcd_metric.py --input_file caption2smiles_example.txt</td>
    <td>Evaluate FCD metric for molecule generation.</td>
  </tr>
</table>



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
