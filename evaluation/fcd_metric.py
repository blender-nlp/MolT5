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

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from fcd import get_fcd, load_ref_model, canonical_smiles

def evaluate(input_file, verbose=False):
    gt_smis = []
    ot_smis = []

    with open(osp.join(input_file)) as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for n, line in enumerate(reader):
            gt_smi = line['ground truth']
            ot_smi = line['output']
            if len(ot_smi) == 0: ot_smi = '[]'

            gt_smis.append(gt_smi)
            ot_smis.append(ot_smi)


    model = load_ref_model()

    canon_gt_smis = [w for w in canonical_smiles(gt_smis) if w is not None]
    canon_ot_smis = [w for w in canonical_smiles(ot_smis) if w is not None]

    fcd_sim_score = get_fcd(canon_gt_smis, canon_ot_smis, model)
    if verbose:
        print('FCD Similarity:', fcd_sim_score)

    return fcd_sim_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='caption2smiles_example.txt', help='path where test generations are saved')
    args = parser.parse_args()
    evaluate(args.input_file, True)
