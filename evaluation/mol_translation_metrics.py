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
#from nltk.translate.meteor_score import meteor_score

from Levenshtein import distance as lev

from rdkit import Chem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def evaluate(input_fp, verbose=False):
    outputs = []

    with open(osp.join(input_fp)) as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for n, line in enumerate(reader):
            gt_smi = line['ground truth']
            ot_smi = line['output']
            outputs.append((line['description'], gt_smi, ot_smi))


    bleu_scores = []
    #meteor_scores = []

    references = []
    hypotheses = []

    for i, (smi, gt, out) in enumerate(outputs):

        if i % 100 == 0:
            if verbose:
                print(i, 'processed.')


        gt_tokens = [c for c in gt]

        out_tokens = [c for c in out]

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        # mscore = meteor_score([gt], out)
        # meteor_scores.append(mscore)

    # BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    if verbose: print('BLEU score:', bleu_score)

    # Meteor score
    # _meteor_score = np.mean(meteor_scores)
    # print('Average Meteor score:', _meteor_score)

    rouge_scores = []

    references = []
    hypotheses = []

    levs = []

    num_exact = 0

    bad_mols = 0

    for i, (smi, gt, out) in enumerate(outputs):

        hypotheses.append(out)
        references.append(gt)

        try:
            m_out = Chem.MolFromSmiles(out)
            m_gt = Chem.MolFromSmiles(gt)

            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt): num_exact += 1
            #if gt == out: num_exact += 1 #old version that didn't standardize strings
        except:
            bad_mols += 1

        

        levs.append(lev(out, gt))


    # Exact matching score
    exact_match_score = num_exact/(i+1)
    if verbose:
        print('Exact Match:')
        print(exact_match_score)

    # Levenshtein score
    levenshtein_score = np.mean(levs)
    if verbose:
        print('Levenshtein:')
        print(levenshtein_score)
        
    validity_score = 1 - bad_mols/len(outputs)
    if verbose:
        print('validity:', validity_score)

    return bleu_score, exact_match_score, levenshtein_score, validity_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='caption2smiles_example.txt', help='path where test generations are saved')
    args = parser.parse_args()
    evaluate(args.input_file, verbose=True)
