  

# Text2Mol

This is MODIFIED code for the paper [Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries](https://aclanthology.org/2021.emnlp-main.47/)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/text2mol-cross-modal-molecule-retrieval-with/cross-modal-retrieval-on-chebi-20)](https://paperswithcode.com/sota/cross-modal-retrieval-on-chebi-20?p=text2mol-cross-modal-molecule-retrieval-with)

### Installation

Code is written in Python 3. Packages are shown in code/packages.txt. However, the following should suffice:
> pytorch
> transformers
> scikit-learn
> numpy

For processing .sdf files, we recommend [RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html).

### Files

| File      | Description |
| ----------- | ----------- |
| models.py   | The three model definitions: MLP, GCN, and Attention.        |


### Citation
If you found our work useful, please cite:
```bibtex
@inproceedings{edwards2021text2mol,
  title={Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries},
  author={Edwards, Carl and Zhai, ChengXiang and Ji, Heng},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={595--607},
  year={2021},
  url = {https://aclanthology.org/2021.emnlp-main.47/}
}
```
