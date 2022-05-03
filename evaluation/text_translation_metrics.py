'''
Code from https://github.com/blender-nlp/MolT5

```bibtex
@article{edwards2022translation,
  title={Translation between Molecules and Natural Language},
  author={Edwards, Carl and Lai, Tuan and Ros, Kevin and Honke, Garrett and Ji, Heng},
  journal={arXiv preprint arXiv:2204.11817},
  year={2022}
}
```
'''

import argparse
import csv

import os.path as osp

import numpy as np

from transformers import BertTokenizerFast

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_metric import PyRouge


parser = argparse.ArgumentParser()

parser.add_argument('--text_model', type=str, default='allenai/scibert_scivocab_uncased', help='Desired language model tokenizer.')

parser.add_argument('--input_file', type=str, default='smiles2caption_example.txt', help='path where test generations are saved')

parser.add_argument('--text_trunc_length', type=str, default=512, help='tokenizer maximum length')

args = parser.parse_args()

outputs = []

with open(osp.join(args.input_file)) as f:
    reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    for n, line in enumerate(reader):
        out_tmp = line['output'][6:] if line['output'].startswith('[CLS] ') else line['output']
        outputs.append((line['SMILES'], line['ground truth'], out_tmp))

text_tokenizer = BertTokenizerFast.from_pretrained(args.text_model)

bleu_scores = []
meteor_scores = []

references = []
hypotheses = []

for i, (smi, gt, out) in enumerate(outputs):
    
    if i % 100 == 0: print(i, 'processed.')


    gt_tokens = text_tokenizer.tokenize(gt, truncation=True, max_length=args.text_trunc_length,
                                        padding='max_length')
    gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
    gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
    gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

    out_tokens = text_tokenizer.tokenize(out, truncation=True, max_length=args.text_trunc_length,
                                        padding='max_length')
    out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
    out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
    out_tokens = list(filter(('[SEP]').__ne__, out_tokens))
   

    references.append([gt_tokens])
    hypotheses.append(out_tokens)

    mscore = meteor_score([gt], out)
    meteor_scores.append(mscore)

bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))

print('BLEU-2 score:', bleu2)
print('BLEU-4 score:', bleu4)
print('Average Meteor score:', np.mean(meteor_scores))

rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)


rouge_scores = []

references = []
hypotheses = []

for i, (smi, gt, out) in enumerate(outputs):

    hypotheses.append(out)
    references.append([gt])
    

rouge_scores = rouge.evaluate(hypotheses, references)

print('ROUGE score:')
print(rouge_scores)

