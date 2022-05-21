
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer
from torch import nn

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from torch.utils.data.dataloader import default_collate

import random
import argparse

import numpy as np
import pickle

from dataloader import TextMoleculeDataset, TextMoleculeReplaceDataset
from models_baseline import EncoderRNN, DecoderRNN

#import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--n_layers', type=int, default=4, help='number of layers')
parser.add_argument('--hidden_size', type=int, default=512, help='hidden size')
parser.add_argument('--max_length', type=int, default=512, help='max length')
parser.add_argument('--max_smiles_length', type=int, default=512, help='max smiles length')
parser.add_argument('--clip', type=float, default=50.0, help='clip value')


parser.add_argument('--data_path', type=str, default='../evaluation/text2mol_data/', help='path where data is located =')
parser.add_argument('--saved_path', type=str, default='saved_models/', help='path where weights are saved')

parser.add_argument('--text_model', type=str, default='allenai/scibert_scivocab_uncased', help='Desired language model.')

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

def pad_collate(batch):

    coll_list = []

    for b in batch:
        coll_list.append(b['smiles_tokens'].squeeze())

    lengths = [(c!=smiles_tokenizer.pad).sum() for c in coll_list]

    padded = pad_sequence(coll_list, padding_value=smiles_tokenizer.pad)

    for b, l in zip(batch, lengths):
        b['smiles_tokens'] = 0 #temporarily replace for normal batch
        b['lengths'] = l
    #collate
    batch = default_collate(batch)

    #add back in smiles tokens
    batch['smiles_tokens'] = padded


    return batch


train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)


encoder = EncoderRNN(args, len(tokenizer), device)
decoder = DecoderRNN(args, len(smiles_tokenizer), device)

encoder.to(device)
decoder.to(device)

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

criterion = nn.NLLLoss(reduction = 'none')

teacher_forcing_ratio = 0.5


def maskNLLLoss(output, target, mask):
    nTotal = mask.sum()
    loss = criterion(output, target)
    loss = loss.to(device)
    loss = loss * mask
    loss = loss.mean()
    return loss, nTotal.item()

def train(smiles_tokens, lengths, text, text_mask, encoder, decoder, encoder_optimizer, decoder_optimizer):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    batch_size = len(lengths)

    # Set device options
    smiles_tokens = smiles_tokens.to(device).transpose(0,1)
    text = text.to(device).transpose(0,1)
    text_lengths = text_mask.sum(dim=1)
    text_mask = text_mask.to(device)
    # Lengths for rnn packing should always be on the cpu

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0


    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(text, text_lengths)
    
    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[smiles_tokenizer.SOS] for _ in range(batch_size)])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(args.max_smiles_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, batch_size, decoder_hidden,
            )
            # Teacher forcing: next input is current target
            decoder_input = smiles_tokens[:,t].view(1, -1)
            # Calculate and accumulate loss
            smiles_mask = torch.full((batch_size,), t) < lengths
            smiles_mask = smiles_mask.to(device)
            mask_loss, nTotal = maskNLLLoss(decoder_output, smiles_tokens[:,t], smiles_mask)#[:,t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(args.max_smiles_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, batch_size, decoder_hidden, 
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            smiles_mask = torch.full((batch_size,), t) < lengths
            smiles_mask = smiles_mask.to(device)
            mask_loss, nTotal = maskNLLLoss(decoder_output, smiles_tokens[:,t], smiles_mask)#[:,t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    
    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def eval(smiles_tokens, lengths, text, text_mask, encoder, decoder):

    # Set device options
    smiles_tokens = smiles_tokens.to(device).transpose(0,1)
    text = text.to(device).transpose(0,1)
    text_lengths = text_mask.sum(dim=1)
    text_mask = text_mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    #lengths = lengths.to("cpu")

    batch_size = len(lengths)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0


    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(text, text_lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[smiles_tokenizer.SOS for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = False #True if random.random() < teacher_forcing_ratio else False

    decoder_outputs = []

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(args.max_smiles_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, batch_size, decoder_hidden,
            )
            # Teacher forcing: next input is current target
            decoder_input = text[:,t].view(1, -1)
            # Calculate and accumulate loss
            smiles_mask = torch.full((batch_size,), t) < lengths
            smiles_mask = smiles_mask.to(device)
            mask_loss, nTotal = maskNLLLoss(decoder_output, smiles_tokens[:,t], smiles_mask)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

            decoder_outputs.append(decoder_output.detach().cpu())
    else:
        for t in range(args.max_smiles_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, batch_size, decoder_hidden, 
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            smiles_mask = torch.full((batch_size,), t) < lengths
            smiles_mask = smiles_mask.to(device)
            mask_loss, nTotal = maskNLLLoss(decoder_output, smiles_tokens[:,t], smiles_mask)#[:,t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

            decoder_outputs.append(decoder_output.detach().cpu())


    inds = torch.stack(decoder_outputs, dim=1).argmax(dim=2).numpy()

    sents = [smiles_tokenizer.decode(s) for s in inds]

    return sum(print_losses) / n_totals, sents


config = vars(args)

#wandb.init(
#    entity="",
#    project="", 
#    config=config)


for i in range(args.epochs):
    
    print('Epoch:', i)
    
    encoder.train()
    decoder.train()
    
    for j, d in enumerate(train_dataloader):

        if j % 100 == 0: print('Step:', j)

        smiles_tokens = d['smiles_tokens']
        lengths = d['lengths']

        text = d['text']
        text_mask = d['text_mask']

        loss = train(smiles_tokens, lengths, text, text_mask, encoder, decoder, encoder_optimizer, decoder_optimizer)
        
        #wandb.log({'total steps':i*len(train_dataloader) + j, 'step':j,'loss' : loss})
    
    

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for j, d in enumerate(val_dataloader):

            if j % 100 == 0: print('Val Step:', j)

            smiles_tokens = d['smiles_tokens']
            lengths = d['lengths']

            text = d['text']
            text_mask = d['text_mask']

            real_text = d['description']

            loss, outputs = eval(smiles_tokens, lengths, text, text_mask, encoder, decoder)
            
            #wandb.log({'val total steps':i*len(val_dataloader) + j, 'step':j,'val loss' : loss})

            if j == 0: 
                data = [[smi, rt, ot] for smi, rt, ot in zip(d['smiles'], real_text, outputs)]
                #table = wandb.Table(columns=["val smiles", "val ground truth", "val output"], data=data)
                #wandb.log({'val outputs':table})
                

torch.save(encoder.state_dict(), args.saved_path + 'rnn_caption2smiles_encoder_baseline_epoch' + str(args.epochs) + '_layers'+str(args.n_layers)+'.pt')
torch.save(decoder.state_dict(), args.saved_path + 'rnn_caption2smiles_decoder_baseline_epoch' + str(args.epochs) + '_layers'+str(args.n_layers)+ '.pt')

with open(args.saved_path + 'rnn_caption2smiles_vocab_baseline_epoch' + str(args.epochs) + '.pkl', 'wb') as f:
    pickle.dump(smiles_tokenizer, f)


def translate(smiles_tokens, lengths, text, text_mask, encoder, decoder, k=5):

    # Set device options
    smiles_tokens = smiles_tokens.to(device).view(-1,1)
    text = text.to(device).view(-1,1)
    text_mask = text_mask.to(device).view(-1,1)
    text_lengths = text_mask.sum().view(-1).cpu()
    text_mask = text_mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    #lengths = lengths.to("cpu")


    encoder.eval()
    decoder.eval()
    

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(text, text_lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[smiles_tokenizer.SOS]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    start_symbol=smiles_tokenizer.SOS

    EOS_token = smiles_tokenizer.EOS
    
    final_beams = []

    beams = [(decoder_hidden, decoder_input, 0, start_symbol, [])]
    with torch.no_grad():
        for i in range(args.max_smiles_length-1):

            if len(beams) == 0: break
            
            next_beams = []
            
            decoder_hidden = torch.cat([b[0] for b in beams], dim = 1)
            decoder_input = torch.cat([b[1] for b in beams], dim = 1)

            decoder_output, decoder_hidden = decoder(
                decoder_input, len(beams), decoder_hidden, 
            )
            for jb, pb in enumerate(decoder_output):
                probs, inds = torch.topk(pb, k)
                for ind, p in zip(inds, probs):

                    decoder_input = torch.LongTensor([ind,1]).to(device)

                    tmp = beams[jb][4].copy()
                    tmp.append(ind)
                    next_beams.append((decoder_hidden[:,jb:jb+1,:], torch.LongTensor([[ind]]).to(device), p.item()+beams[jb][2], ind, tmp)) #hidden state, prob, word idx


            beams = sorted(next_beams, key=lambda tup:tup[2], reverse=True)[:k]
            for b in beams:
                if b[3] == EOS_token: 
                    final_beams.append(b)
                    k = k - 1
            beams = [b for b in beams if b[3] != EOS_token]
            
    if len(beams) != 0: final_beams.extend(beams)

    final_beams = sorted(final_beams, key=lambda tup:tup[2], reverse=True)

    tgt_tokens = final_beams[0][4]

    sent = smiles_tokenizer.decode(tgt_tokens)

    sent = sent.replace('[EOS]', '').replace('[SOS]', '').replace('[PAD]', '')
    
    return sent


smiles = []
test_outputs = []
descriptions = []
    
with torch.no_grad():
    for j, d in enumerate(test_dataloader):

        if j % 10 == 0: print('Test Step:', j)

        smiles_tokens = d['smiles_tokens']
        lengths = d['lengths']

        text = d['text']
        text_mask = d['text_mask']

        real_text = d['description']

        sents = [translate(st, l, t, tm, encoder, decoder) for st, l, t, tm in zip(smiles_tokens, lengths, text, text_mask)]

        smiles.extend(d['smiles'])
        descriptions.extend(real_text)
        test_outputs.extend(sents)
        

with open(args.output_file, 'w') as f:
    f.write('description' + '\t' + 'ground truth' + '\t' + 'output' + '\n')
    for desc, rt, ot in zip(descriptions, smiles, test_outputs):
        f.write(desc + '\t' + rt + '\t' + ot + '\n')

#wandb.finish()