import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_nbow(m):
    if hasattr(m, 'weight'):
        nn.init.xavier_uniform_(m.weight.data)

class NBOW(nn.Module):
    def __init__(self,
                 vocab_size,
                 hid_dim,
                 dropout,
                 pad_idx,
                 device):
        super().__init__()

        self.device = device
        
        self.embedding = nn.Embedding(vocab_size, hid_dim, padding_idx = pad_idx)

        self.dropout = nn.Dropout()

        self.apply(initialize_nbow)

    def forward(self, src, src_len, src_mask):

        src = src.permute(1, 0)
        src_mask = src_mask.permute(1, 0)

        # src = [batch size, src len]
        # src_len = [batch size]
        # src_mask = [batch size, src len]

        src = self.dropout(self.embedding(src))

        #src = [batch size, src len, hid dim]

        return src