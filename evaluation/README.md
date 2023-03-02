# Evaluation Code: Translation between Molecules and Natural Language
Task evaluation code for "[Translation between Molecules and Natural Language](https://arxiv.org/abs/2204.11817)".

## Updates
10/6: 
* Exact comparison is now done via InChI strings.
* Addressed a small bug in FCD metric code. Results are qualitatively identical, but the new version is much faster and hopefully will be more meaningful in future work. 

## Installation
The requirements for the evaluation code conda environment are in environment_eval.yml. An environment can be created using the following commands: 

```
conda env create -n MolTextTranslationEval -f environment_eval.yml python=3.9
conda activate MolTextTranslationEval
python -m spacy download en_core_web_sm
pip install git+https://github.com/samoturk/mol2vec
```

### Downloads

* [test_outputfinal_weights.320.pt](https://uofi.box.com/s/es16alnhzfy1hpagf55fu48k49f8n29x) should be placed in "evaluation/t2m_output".
It can be downloaded using ```curl -L  https://uofi.box.com/shared/static/es16alnhzfy1hpagf55fu48k49f8n29x --output test_outputfinal_weights.320.pt```

If GitHub LFS fails:
* [cid_to_smiles.pkl](https://uofi.box.com/v/MolT5-cid-to-smiles)

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
@inproceedings{edwards-etal-2022-translation,
    title = "Translation between Molecules and Natural Language",
    author = "Edwards, Carl  and
      Lai, Tuan  and
      Ros, Kevin  and
      Honke, Garrett  and
      Cho, Kyunghyun  and
      Ji, Heng",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.26",
    pages = "375--413",
}

```
