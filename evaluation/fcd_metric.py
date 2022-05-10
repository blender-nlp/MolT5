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

from tqdm import tqdm
from rdkit import Chem
from fcd import get_fcd, load_ref_model

def evaluate(input_file, verbose=False):
    outputs = []
    bad_mols = 0

    with open(osp.join(input_file)) as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for n, line in enumerate(reader):
            try:
                gt_smi = line['ground truth']
                ot_smi = line['output']
                gt_m = Chem.MolFromSmiles(gt_smi)
                ot_m = Chem.MolFromSmiles(ot_smi)
                if ot_m == None: raise ValueError('Bad SMILES')
                outputs.append((line['description'], gt_m, ot_m))
            except:
                bad_mols += 1
    if verbose:
        print('validity:', len(outputs)/(len(outputs)+bad_mols))


    model = load_ref_model()

    fcd_sims = []

    enum_list = outputs

    for i, (desc, gt_m, ot_m) in enumerate(tqdm(enum_list)):

        if i % 100 == 0:
            if verbose: print(i, 'processed.')


        gt_smi, ot_smi = Chem.MolToSmiles(gt_m), Chem.MolToSmiles(ot_m)

        #fix a bug in FCD where the covariance matrix of a single character smiles string is NaN:
        if len(gt_smi) == 1: gt_smi = '[' + gt_smi + ']'
        if len(ot_smi) == 1: ot_smi = '[' + ot_smi + ']'

        fcd_sims.append(get_fcd(gt_smi, ot_smi, model))

    fcd_sim_score = np.mean(fcd_sims)
    if verbose:
        print('Average FCD Similarity:', fcd_sim_score)

    return fcd_sim_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='caption2smiles_example.txt', help='path where test generations are saved')
    args = parser.parse_args()
    evaluate(args.input_file, True)
