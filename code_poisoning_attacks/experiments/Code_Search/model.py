# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, encoder, config):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config

    def forward(self, source_inputs=None, source_embeddings=None, source_mask=None):
        if source_inputs is not None:
            outputs = self.encoder(source_inputs, attention_mask=source_inputs.ne(1))[0]
            outputs = (outputs * source_inputs.ne(1)[:, :, None]).sum(1) / source_inputs.ne(1).sum(-1)[:, None]
        elif source_embeddings is not None:
            outputs = self.encoder(inputs_embeds=source_embeddings, attention_mask=source_mask)[0]
            outputs = (outputs * source_mask[:, :, None]).sum(1) / source_mask.sum(-1)[:, None]
        else:
            pass
        return torch.nn.functional.normalize(outputs, p=2, dim=1)
