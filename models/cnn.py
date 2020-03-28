import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_cnn(m):
    if hasattr(m, 'weight'):
        nn.init.xavier_uniform_(m.weight.data)

class CNN1D(nn.Module):
    def __init__(self,
                 vocab_size,
                 hid_dim,
                 filter_size,
                 n_layers,
                 dropout,
                 pad_idx,
                 max_length,
                 device):
        super().__init__()

        self.tok_embedding = nn.Embedding(vocab_size, hid_dim, padding_idx = pad_idx)
        self.pos_embedding = nn.Embedding(max_length, hid_dim, padding_idx = pad_idx)

        if filter_size % 2 == 0:
            padding = (filter_size // 2) - 1
        else:
            padding = filter_size // 2

        self.layers = nn.ModuleList([nn.Conv1d(hid_dim, hid_dim, filter_size, padding=padding) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.filter_size = filter_size
        self.pad_idx = pad_idx
        self.device = device

    def forward(self, src, src_lens, src_mask):

        # src = [src len, batch size]
        # src_lens = [batch size]
        # src_mask = [src len, batch size]

        tok_embedded = self.tok_embedding(src)
        tok_embedded = tok_embedded.permute(1, 0, 2)

        batch_size = src.shape[1]
        src_len = src.shape[0]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        pos_embedded = self.pos_embedding(pos)

        embedded = self.dropout(tok_embedded + pos_embedded)

        #embedded = [batch size, src len, emb dim]

        embedded = embedded.permute(0, 2, 1)

        #embedded = [batch size, emb dim, src len]

        hid_dim = embedded.shape[1]

        for layer in self.layers:

            _embedded = layer(embedded)

            if self.filter_size % 2 == 0:
                padding = torch.zeros(batch_size, hid_dim, 1).fill_(self.pad_idx).to(self.device)
                _embedded = torch.cat((_embedded, padding), dim=-1)

            _embedded += embedded

            embedded = self.dropout(torch.tanh(_embedded))

        #embedded = [batch size, emb dim, src len]

        embedded = embedded.permute(0, 2, 1)

        #embedded = [batch size, src len, emb dim]

        return embedded