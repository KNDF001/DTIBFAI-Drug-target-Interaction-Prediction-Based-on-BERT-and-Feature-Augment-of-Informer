# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from decoder import Decoder, DecoderLayer
from attn import FullAttention, ProbAttention, AttentionLayer
from embed import DataEmbedding
torch.manual_seed(1)
np.random.seed(1)

class BIN_Interaction_Flat(nn.Sequential):
    '''
        Interaction Network with 2D interaction map
    '''
    
    def __init__(self, **config):
        super(BIN_Interaction_Flat, self).__init__()
        self.max_d = config['max_drug_seq']
        self.max_p = config['max_protein_seq']
        self.emb_size = config['emb_size']
        self.dropout_rate = config['dropout_rate']
        
        #densenet
        self.scale_down_ratio = config['scale_down_ratio']
        self.growth_rate = config['growth_rate']
        self.transition_rate = config['transition_rate']
        self.num_dense_blocks = config['num_dense_blocks']
        self.kernal_dense_size = config['kernal_dense_size']
        self.batch_size = config['batch_size']
        self.input_dim_drug = config['input_dim_drug']
        self.input_dim_target = config['input_dim_target']
        self.gpus = max(1, torch.cuda.device_count())
        self.n_layer = 2
        #encoder
        self.hidden_size = config['emb_size']
        self.intermediate_size = config['intermediate_size']
        self.num_attention_heads = config['num_attention_heads']
        self.attention_probs_dropout_prob = config['attention_probs_dropout_prob']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        
        self.flatten_dim = config['flat_dim'] 
        self.icnn = nn.Conv2d(1, 3, 3, padding = 0) 
        self.decoder = nn.Sequential(
            nn.Linear(6096, 1024),
            nn.ReLU(True),

            nn.BatchNorm1d(1024),
            nn.Linear(1024, 256),
            nn.ReLU(True),

            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            
            #output layer
            nn.Linear(32, 1)
        ) 
        self.dmodel = Informer(7, 7, 7, self.max_d, 71, True)
        self.pmodel = Informer(7, 7, 7, self.max_p, 513, True)
        
    def forward(self, d, p, d_mask, p_mask):
        ex_d_mask = d_mask.unsqueeze(1).unsqueeze(2)
        ex_p_mask = p_mask.unsqueeze(1).unsqueeze(2)

        ex_d_mask = (1.0 - ex_d_mask) * -10000.0
        ex_p_mask = (1.0 - ex_p_mask) * -10000.0

        d_encoded_layers = self.dmodel(d.float(), ex_d_mask.float())
        p_encoded_layers = self.pmodel(p.float(), ex_p_mask.float())

        d_aug = torch.unsqueeze(d_encoded_layers, 2).repeat(1, 1, 129, 1) 
        p_aug = torch.unsqueeze(p_encoded_layers, 1).repeat(1, 18, 1, 1) 
        
        i = d_aug * p_aug 
        i_v = i.view(int(self.batch_size/1), -1, 18, 129) 
        i_v = torch.sum(i_v, dim = 1)
        i_v = torch.unsqueeze(i_v, 1)
        i_v = F.dropout(i_v, p = self.dropout_rate)        
        f = self.icnn(i_v)
        f = f.view(int(self.batch_size/1), -1)
        score = self.decoder(f)
        return score    


class Informer(nn.Module): 
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=768, n_heads=12, e_layers=3, d_layers=2, d_ff=768, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = True, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        Attn = ProbAttention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Sequential(
            nn.Linear(512, 1), 
            nn.Sigmoid() 
        )
        
    def forward(self, x_enc, x_mark_enc, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out, attns = self.encoder(x_enc, attn_mask=enc_self_mask)
        return enc_out
