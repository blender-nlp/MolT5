
from torch.utils.data import Dataset


import os.path as osp

import csv
import pickle

import spacy


class TextMoleculeDataset(Dataset):
    def __init__(self, data_path, split, tokenizer):
        self.data_path = data_path
        
        self.tokenizer = tokenizer

        self.cids = []
        self.descriptions = {}
    
        self.mol2vec = {}
        self.smiles = {}
        
        #load data
        with open(osp.join(data_path, split+'.txt')) as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames = ['cid', 'mol2vec', 'desc'])
            for n, line in enumerate(reader):
                self.descriptions[line['cid']] = line['desc']
                self.mol2vec[line['cid']] = line['mol2vec']
                self.cids.append(line['cid'])

        #load smiles
        with open('../evaluation/cid_to_smiles.pkl', 'rb') as f:
            self.cids_to_smiles = pickle.load(f)

        #remove '*' smiles from dataset
        cids_to_remove = []
        for cid in self.cids_to_smiles:
            if self.cids_to_smiles[cid] == '*':
                cids_to_remove.append(cid)

        for cid in cids_to_remove:
            self.cids_to_smiles.pop(cid, None)
            if cid in self.cids:
                self.cids.remove(cid)


    def __len__(self):
        return len(self.cids)


    def __getitem__(self, idx):

        cid = self.cids[idx]

        smiles = self.cids_to_smiles[cid]

        description = self.descriptions[cid]

        text = self.tokenizer(description, padding="max_length", max_length=512, truncation=True, return_tensors='pt')

        smiles_tokens = self.smiles_tokenizer.get_tensor(smiles)

        return {'smiles':smiles, 'description':description, 'smiles_tokens':smiles_tokens, #'smiles_mask':smiles_mask, 
            'text':text['input_ids'].squeeze(), 'text_mask':text['attention_mask'].squeeze()}



class TextMoleculeReplaceDataset(Dataset): #This dataset replaces the name of the molecule at the beginning of the description
    def __init__(self, data_path, split, tokenizer):
        self.data_path = data_path
        
        self.tokenizer = tokenizer

        self.cids = []
        self.descriptions = {}
    
        self.mol2vec = {}
        self.smiles = {}
        
        #load data
        with open(osp.join(data_path, split+'.txt')) as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames = ['cid', 'mol2vec', 'desc'])
            for n, line in enumerate(reader):
                self.descriptions[line['cid']] = line['desc']
                self.mol2vec[line['cid']] = line['mol2vec']
                self.cids.append(line['cid'])

        #load smiles            
        with open('../evaluation/cid_to_smiles.pkl', 'rb') as f:
            self.cids_to_smiles = pickle.load(f)

        #remove '*' smiles from dataset
        cids_to_remove = []
        for cid in self.cids_to_smiles:
            if self.cids_to_smiles[cid] == '*':
                cids_to_remove.append(cid)
        

        for cid in cids_to_remove:
            self.cids_to_smiles.pop(cid, None)
            if cid in self.cids:
                self.cids.remove(cid)

        nlp = spacy.load("en_core_web_sm")

        for cid in self.descriptions:
            desc = self.descriptions[cid]
            doc = nlp(desc)
            for token in doc:
                if token.text == 'is':
                    desc = 'The molecule ' + desc[token.idx:]
                    break

            self.descriptions[cid] = desc

    def __len__(self):
        return len(self.cids)


    def __getitem__(self, idx):

        cid = self.cids[idx]

        smiles = self.cids_to_smiles[cid]

        description = self.descriptions[cid]

        text = self.tokenizer(description, padding="max_length", max_length=512, truncation=True, return_tensors='pt')

        smiles_tokens = self.smiles_tokenizer.get_tensor(smiles)

        return {'smiles':smiles, 'description':description, 'smiles_tokens':smiles_tokens, 
            'text':text['input_ids'].squeeze(), 'text_mask':text['attention_mask'].squeeze()}


