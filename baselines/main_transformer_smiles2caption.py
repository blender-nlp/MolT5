
#Based on https://pytorch.org/tutorials/beginner/translation_transformer.html

from lib2to3.pgen2 import token
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer
from torch import nn
import torch.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from torch.utils.data.dataloader import default_collate

from transformers.optimization import get_linear_schedule_with_warmup

import numpy as np

import pickle

import argparse


from dataloader import TextMoleculeDataset, TextMoleculeReplaceDataset
from models_baseline import Seq2SeqTransformer

#import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--hidden_size', type=int, default=2048, help='hidden size')
parser.add_argument('--nlayers', type=int, default=6, help='number of layers')
parser.add_argument('--emb_size', type=int, default=512, help='input dimension size')
parser.add_argument('--max_length', type=int, default=512, help='max length')
parser.add_argument('--max_smiles_length', type=int, default=512, help='max smiles length')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--nhead', type=int, default=8, help='num attention heads')


parser.add_argument('--data_path', type=str, default='../evaluation/text2mol_data/', help='path where data is located =')
parser.add_argument('--saved_path', type=str, default='saved_models/', help='path where weights are saved')

parser.add_argument('--text_model', type=str, default='allenai/scibert_scivocab_uncased', help='Desired language model.')

parser.add_argument('--use_scheduler', type=bool, default=True, help='Use linear scheduler')
parser.add_argument('--num_warmup_steps', type=int, default=400, help='Warmup steps for linear scheduler, if enabled.')

parser.add_argument('--output_file', type=str, default='out.txt', help='path where test generations are saved')


parser.add_argument('--mol_replace', action=argparse.BooleanOptionalAction)


args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(args.text_model)

if args.mol_replace:
    train_data = TextMoleculeReplaceDataset(args.data_path, 'training', tokenizer)
    val_data = TextMoleculeReplaceDataset(args.data_path, 'val', tokenizer)
    test_data = TextMoleculeReplaceDataset(args.data_path, 'test', tokenizer)
else:
    train_data = TextMoleculeDataset(args.data_path, 'training', tokenizer)
    val_data = TextMoleculeDataset(args.data_path, 'val', tokenizer)
    test_data = TextMoleculeDataset(args.data_path, 'test', tokenizer)

def build_smiles_vocab(dicts):
    smiles = []
    for d in dicts:
        for cid in d:
            smiles.append(d[cid])

    char_set = set()

    for smi in smiles:
        for c in smi:
            char_set.add(c)

    return ''.join(char_set)

class SmilesTokenizer():

    def __init__(self, smiles_vocab, max_len=512):
        self.smiles_vocab = smiles_vocab
        self.max_len = max_len
        self.vocab_size = len(smiles_vocab) + 3 #SOS, EOS, pad
        
        self.SOS = self.vocab_size - 2
        self.EOS = self.vocab_size - 1
        self.pad = 0

    def letterToIndex(self, letter):
        return self.smiles_vocab.find(letter) + 1 #skip 0 == [PAD]
        
    def ind2Letter(self, ind):
        if ind == self.SOS: return '[SOS]'
        if ind == self.EOS: return '[EOS]'
        if ind == self.pad: return '[PAD]'
        return self.smiles_vocab[ind-1]
        
    def decode(self, iter):
        return "".join([self.ind2Letter(i) for i in iter]).replace('[SOS]','').replace('[EOS]','').replace('[PAD]','')

    def __len__(self):
        return self.vocab_size

    def get_tensor(self, smi):
        tensor = torch.zeros(1, args.max_smiles_length, dtype=torch.int64)
        tensor[0,0] = smiles_tokenizer.SOS
        for li, letter in enumerate(smi):
            tensor[0,li+1] = self.letterToIndex(letter)
            if li + 3 == args.max_smiles_length: break
        tensor[0, li+2] = self.EOS

        return tensor

smiles_vocab = build_smiles_vocab((train_data.cids_to_smiles, val_data.cids_to_smiles, test_data.cids_to_smiles))
smiles_tokenizer = SmilesTokenizer(smiles_vocab)

train_data.smiles_tokenizer = smiles_tokenizer
val_data.smiles_tokenizer = smiles_tokenizer
test_data.smiles_tokenizer = smiles_tokenizer


train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)#, collate_fn=pad_collate)
val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)#, collate_fn=pad_collate)
test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)#, collate_fn=pad_collate)


model = Seq2SeqTransformer(args, len(smiles_tokenizer), len(tokenizer))


model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

num_training_steps = args.epochs * len(train_dataloader) - args.num_warmup_steps
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.num_warmup_steps, num_training_steps = num_training_steps) 


PAD_IDX = 0 #note that both vocabularies share the same padding token

criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

SOS_token = 102
EOS_token = 103
MAX_LENGTH = args.max_length

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(dataloader, model, optimizer, epoch):


    model.train()
    losses = 0
    
    for j, d in enumerate(dataloader):

        if j % 1000 == 0: print('Step:', j)

        smiles_tokens = d['smiles_tokens'].squeeze().to(device).transpose(0,1)

        text = d['text'].to(device).transpose(0,1)

        tgt_input = text[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(smiles_tokens, tgt_input)

        logits = model(smiles_tokens, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = text[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        scheduler.step()
        
        #wandb.log({'total steps':epoch*len(dataloader) + j, 'step':j,'loss' : loss})
        
        losses += loss.item()


    return losses / len(dataloader)


def eval(dataloader, model, epoch):

    model.eval()

    losses = 0

    with torch.no_grad():
        for j, d in enumerate(dataloader):

            if j % 100 == 0: print('Val Step:', j)

            smiles_tokens = d['smiles_tokens'].squeeze().to(device).transpose(0,1)

            text = d['text'].to(device).transpose(0,1)

            real_text = d['description']

            tgt_input = text[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(smiles_tokens, tgt_input)

            logits = model(smiles_tokens, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = text[1:, :]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

            #wandb.log({'val total steps':epoch*len(dataloader) + j, 'step':j,'val loss' : loss})

            if j == 0: 
                inds = logits.argmax(dim=2).transpose(0,1)
                sents = [tokenizer.decode(s) for s in inds]
                data = [[smi, rt, ot] for smi, rt, ot in zip(d['smiles'], real_text, sents)]
                #table = wandb.Table(columns=["val smiles", "val ground truth", "val output"], data=data)
                #wandb.log({'val outputs':table})
    return losses/len(dataloader)

config = vars(args)

#wandb.init(
#    entity="",
#    project="", 
#    config=config)


for i in range(args.epochs):
    
    print('Epoch:', i)

    train_epoch(train_dataloader, model, optimizer, i)

    eval(val_dataloader, model, i)
    

torch.save(model.state_dict(), args.saved_path + 'transformer_smiles2caption_baseline_epoch' + str(args.epochs) + '.pt')

with open(args.saved_path + 'transformer_smiles2caption_vocab_baseline_epoch' + str(args.epochs) + '.pkl', 'wb') as f:
    pickle.dump(smiles_tokenizer, f)

softmax = nn.LogSoftmax(dim=1)

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_token:
            break
    return ys

def beam_decode_fast(model, src, src_mask, max_len, start_symbol, k = 5):
    src = src.to(device)
    src_mask = src_mask.to(device)

    final_beams = []

    memory = model.encode(src, src_mask)
    memory = memory.to(device)
    #ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    beams = [(torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device), 0, start_symbol)]
    with torch.no_grad():
        for i in range(max_len-1):

            if len(beams) == 0: break
            
            next_beams = []
        
            ys = torch.cat([b[0] for b in beams], dim = 1)

            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(device)
            out = model.decode(ys, memory.repeat(1,ys.size(1),1), tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            prob = softmax(prob)
            
            for jb, pb in enumerate(prob):
                probs, inds = torch.topk(pb, k)
                for ind, p in zip(inds, probs):

                    next_beams.append((torch.cat([ys[:,jb].view(-1,1), torch.ones(1, 1).type_as(src.data).fill_(ind)], dim=0), p.item()+beams[jb][1], ind))


            beams = sorted(next_beams, key=lambda tup:tup[1], reverse=True)[:k]
            for b in beams:
                if b[2] == EOS_token: 
                    final_beams.append(b)
                    k = k - 1
            beams = [b for b in beams if b[2] != EOS_token]
            
    if len(beams) != 0: final_beams.extend(beams)

    final_beams = sorted(final_beams, key=lambda tup:tup[1], reverse=True)

    return final_beams[0][0]

def translate(model: torch.nn.Module, smi: str):
    model.eval()
    src = smiles_tokenizer.get_tensor(smi).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = beam_decode_fast(
        model,  src, src_mask, max_len=args.max_length, start_symbol=SOS_token).flatten()
        

    sent = tokenizer.decode(tgt_tokens)

    #print(smi, sent)
    sent = sent.replace('[CLS] ', '').replace(' [SEP]', '').replace('[CLS]', '').replace('[SEP]', '').strip()

    return sent


def test_eval(dataloader, model):

    model.eval()

    smiles = []
    test_outputs = []
    test_gt = []

    with torch.no_grad():
        for j, d in enumerate(dataloader):

            if j % 100 == 0: print('Test Step:', j)

            real_text = d['description']

            smiles.extend(d['smiles'])
            test_gt.extend(real_text)
            test_outputs.extend([translate(model, smi) for smi in d['smiles']])

            #wandb.log({'test total steps':len(dataloader) + j, 'step':j,'test loss' : loss})

    return smiles, test_gt, test_outputs

smiles, test_gt, test_outputs = test_eval(test_dataloader, model)

with open(args.output_file, 'w') as f:
    f.write('SMILES' + '\t' + 'ground truth' + '\t' + 'output' + '\n')
    for smi, rt, ot in zip(smiles, test_gt, test_outputs):
        f.write(smi + '\t' + rt + '\t' + ot + '\n')


#wandb.finish()