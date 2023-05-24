# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, n_vocab, embed, num_classes, filter_sizes=(2, 3, 4), num_filters=256):
        super(Model, self).__init__()
        # if config.embedding_pretrained is not None:
        #     self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        # else:
        #     self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.embedding = nn.Embedding(n_vocab, embed, padding_idx=n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed)) for k in filter_sizes])
        self.dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
