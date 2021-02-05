import math
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F


class TransformerModel(nn.Module):
    """Container module with an encoder, transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, nclasses, idropout=0.1, hdropout=0.5, 
                 layer_norm=0, src_scale=0, mlp=0):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        if ninp > 0:
            self.encoder = nn.Linear(ntoken, ninp)
        else:
            self.encoder = None
            ninp = nhid = ntoken
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, hdropout, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.mlp = mlp
        if mlp:
            self.lins = nn.ModuleList()
            self.lins.append(nn.Linear(nhid, nhid * 4))
            self.lins.append(nn.Linear(nhid * 4, nclasses))
            self.bns = nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(nhid * 4))
            self.hdropout = hdropout
        else:
            self.decoder = nn.Linear(ninp, nclasses)
        self.dropout = nn.Dropout(p=idropout)
        self.src_scale = src_scale
        self.layer_norm = layer_norm
        if layer_norm:
            self.input_layer_norm = nn.LayerNorm(ninp)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None, padding=None, pe=None):
        src = src.transpose(0, 1)
        if self.encoder is not None:
            src = self.encoder(src)
        if self.src_scale:
            src = src * math.sqrt(self.ninp)
        if pe is not None:
            pe = pe.transpose(0, 1)
            src = src + pe
        if self.layer_norm:
            src = self.input_layer_norm(src)
        src = self.dropout(src)
        x = self.transformer_encoder(src, src_mask, src_key_padding_mask=padding)[0]
        if self.mlp:
            for i, lin in enumerate(self.lins[:-1]):
                x = lin(x)
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.hdropout, training=self.training)
            output = self.lins[-1](x)
        else:
            output = self.decoder(x)

        return output

