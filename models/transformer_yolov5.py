# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 15:40:24 2021

@author: jankos

This file defines the Bounding Box Transformer

Used https://github.com/jankrepl/mildlyoverfitted/blob/master/github_adventures/vision_transformer/custom.py
as example in some parts
"""
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import math
import torch
import time

#%%
class BoxEmbedder(nn.Module): #TBD: try different embedding approaches
    """
    creates embeddings of the bounding boxes
    take the bounding boxes, apply conv with kernel size = img size, flatten
    """
    def __init__(self, img_size, embed_dim=96):
        super().__init__()
        self.Conv = nn.Conv2d(3, embed_dim, kernel_size=img_size,
                              stride=img_size)
    
    def forward(self, x):
        x = self.Conv(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x
        
class PositionalEncoding(nn.Module):
    """
    Applied positional encoding to the stack of boxes so the model can use
    positional (i.e. which box came after which box) information
    from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class BBTransformer(nn.Module):
    """
    The transofrmer model itself
    
    img_size: bounding box width and height, assumed to be the same
    d_embed: dimension of the embedded bbox
    n_head: number of heads
    n_layers: number of encoder layers in the transformer
    n_classes: number of predicted classes
    tran_droput: dropout prob for the transformer pos encoder
    
    """
    
    def __init__(self, img_size, d_embed, n_head, n_layers, n_classes, tran_dropout=0.3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_embed,
                                                        nhead=n_head,
                                                        batch_first=True)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer,
                                                   num_layers=n_layers)
        self.pos_encoder = PositionalEncoding(d_embed, dropout=tran_dropout,
                                              max_len=d_embed)
        self.norm = nn.LayerNorm(d_embed, eps=1e-6)
        self.head = nn.Linear(d_embed, n_classes)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_embed))
        self.box_embedder = BoxEmbedder(img_size=img_size, embed_dim=d_embed)
    def forward(self, x):
        embedded_stack = []
        for i in range(x.size()[0]):
            #embed batches, find faster way?
            x0 = self.box_embedder(x[i])
            embedded_stack.append(x0)
        x = torch.stack(embedded_stack).squeeze(dim=2)
        n_samples = x.shape[0]
        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_encoder(x)
        x = self.trans_encoder(x)
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x
        
