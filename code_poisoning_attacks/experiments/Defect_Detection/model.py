# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = x.reshape(-1, x.size(-1) * 2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.classifier = RobertaClassificationHead(config)
        # self.dropout = nn.Dropout(0)

    def forward(self, input_ids=None, input_embeddings=None, input_mask=None, labels=None):
        if input_ids != None and input_embeddings == None and input_mask == None:
            outputs = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(1))[0]
        elif input_embeddings != None and input_mask != None:
            outputs = self.encoder(inputs_embeds=input_embeddings, attention_mask=input_mask)[0]
        logits = self.classifier(outputs)
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob
