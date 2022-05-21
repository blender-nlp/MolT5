
import torch
from torch import nn

from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import Transformer
from torch import nn, Tensor

import math



class EncoderRNN(nn.Module):
    def __init__(self, args, vocab_size, device):
        super(EncoderRNN, self).__init__()
        
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        
        self.vocab_size = vocab_size
        self.device=device

        self.n_layers = args.n_layers
        self.dropout = 0

        #layers:

        self.embedding = nn.Embedding(vocab_size, self.hidden_size)

        self.logits = nn.Linear(self.hidden_size, vocab_size)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers,
                          dropout=(0 if self.n_layers == 1 else self.dropout), bidirectional=True)

    def forward(self, input, lengths, hidden=None):

        emb = self.embedding(input)

        emb = pack_padded_sequence(emb, lengths, enforce_sorted=False)

        output, hidden = self.gru(emb)

        output, _ = pad_packed_sequence(output)

        output = output[:, :, :self.hidden_size] + output[:, : ,self.hidden_size:]

        return output, hidden

    def initHidden(self, batch_size = 1):
        return torch.zeros(self.n_layers*2, batch_size, self.hidden_size, device= self.device)


class DecoderRNN(nn.Module):
    def __init__(self, args, vocab_size, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = args.hidden_size
        self.vocab_size = vocab_size
        self.device = device

        self.n_layers = args.n_layers
        self.dropout = 0

        self.embedding = nn.Embedding(vocab_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.n_layers)
        self.out = nn.Linear(self.hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, batch_size, hidden=None):
        output = self.embedding(input).view(1, batch_size, self.hidden_size)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size = 1):
        return torch.zeros(self.n_layers, batch_size, self.hidden, device=self.device)




################Begin Transformer Model #################################
#See https://pytorch.org/tutorials/beginner/translation_transformer.html

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, args,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()


        self.transformer = Transformer(d_model=args.emb_size,
                                       nhead=args.nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=args.hidden_size,
                                       dropout=dropout)
        self.generator = nn.Linear(args.emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, args.emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, args.emb_size)
        self.positional_encoding = PositionalEncoding(
            args.emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):

        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
