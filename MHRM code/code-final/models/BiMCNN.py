# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

### This is our own model Bi-LSTM, multi-scale CNN and adaptive attention fusion

class Model(nn.Module):
    def __init__(self, n_vocab, embed, num_classes, num_class_2,
                 filter_sizes=(2, 3, 4), num_filters=256,
                 hidden_size=100, hidden_size2=50, num_layers=2):
        super(Model, self).__init__()
        # if config.embedding_pretrained is not None:
        #     self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        # else:
        #     self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.embedding = nn.Embedding(n_vocab, embed, padding_idx=n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed)) for k in filter_sizes])
        self.dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(num_filters * len(filter_sizes), embed)

        self.lstm = nn.LSTM(embed, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=0.8)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_size * 2, embed)
        self.fc2 = nn.Linear(embed*2, num_classes)
        self.fc3 = nn.Linear(embed*2, num_class_2)

        self.m = nn.Sigmoid()
        # self.loss = torch.nn.BCELoss()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out_ori = self.embedding(x)

        ## CNN model
        out = out_ori.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out_CNN = self.fc(out)

        ## BilSTM attention model
        H, _ = self.lstm(out_ori)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out_LSTM = self.fc1(out)

        # print("out_CNN", out_LSTM.shape, out_CNN.shape)
        out_final = torch.cat((out_CNN, out_LSTM), dim=-1)
        out_resut = self.fc2(out_final)  # [128, 64]

        out_result2 = self.m(self.fc3(out_final))

        # result_2 = self.loss(self.m(out_result2), target)


        return out_resut, out_result2, alpha

