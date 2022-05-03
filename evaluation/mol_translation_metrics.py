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


import pickle
import argparse
import csv

import os.path as osp

import numpy as np

#load metric stuff

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score

from Levenshtein import distance as lev


parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str, default='caption2smiles_example.txt', help='path where test generations are saved')


args = parser.parse_args()

outputs = []

with open(osp.join(args.input_file)) as f:
    reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    for n, line in enumerate(reader):
        gt_smi = line['ground truth']
        ot_smi = line['output']
        outputs.append((line['description'], gt_smi, ot_smi))


bleu_scores = []
meteor_scores = []

references = []
hypotheses = []

for i, (smi, gt, out) in enumerate(outputs):
    
    if i % 100 == 0: print(i, 'processed.')


    gt_tokens = [c for c in gt]

    out_tokens = [c for c in out]

    references.append([gt_tokens])
    hypotheses.append(out_tokens)

    mscore = meteor_score([gt], out)
    meteor_scores.append(mscore)

    
print('BLEU score:', corpus_bleu(references, hypotheses))
print('Average Meteor score:', np.mean(meteor_scores))


rouge_scores = []

references = []
hypotheses = []

levs = []

num_exact = 0


for i, (smi, gt, out) in enumerate(outputs):
    

    hypotheses.append(out)
    references.append(gt)
    
    if out == gt: num_exact += 1

    levs.append(lev(out, gt))


print('Exact Match:')
print(num_exact/(i+1))

print('Levenshtein:')
print(np.mean(levs))

