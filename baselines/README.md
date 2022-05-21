# Baseline Code: Translation between Molecules and Natural Language
Baseline training code for "[Translation between Molecules and Natural Language](https://arxiv.org/abs/2204.11817)".

## Installation
Installation instructions will be uploaded soon. 


## Training Commands

<table>
  <tr>
    <td>Command</td>
    <td>Purpose</td>
  </tr>
  <tr>
    <td>python main_RNN_smiles2caption.py --output_file RNN_caption2smiles.txt --mol_replace</td>
    <td>Run the RNN baseline for molecule captioning.</td>
  </tr>
  <tr>
    <td>python main_transformer_smiles2caption.py --output_file transformer_caption2smiles.txt --mol_replace</td>
    <td>Run the Transformer baseline for molecule captioning.</td>
  </tr>
  
  <tr>
    <td>python main_RNN_smiles2caption.py --output_file RNN_smiles2caption.txt --mol_replace</td>
    <td>Run the RNN baseline for text-based molecule generation.</td>
  </tr>
  <tr>
    <td>python main_transformer_smiles2caption.py --output_file transformer_smiles2caption.txt --mol_replace</td>
    <td>Run the Transformer baseline for text-based molecule generation.</td>
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

### References

https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
https://pytorch.org/tutorials/beginner/translation_transformer.html
