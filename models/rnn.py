import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_rnn(m):
    if hasattr(m, 'weight'):
        nn.init.xavier_uniform_(m.weight.data)

class BiRNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 hid_dim,
                 n_layers,
                 rnn_type,
                 dropout,
                 pad_idx,
                 device):
        super().__init__()

        self.device = device
        
        self.embedding = nn.Embedding(vocab_size, hid_dim, padding_idx = pad_idx)

        if rnn_type == 'GRU':
            self.rnn = nn.GRU(hid_dim, hid_dim//2, num_layers=n_layers, bidirectional=True, dropout=dropout)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hid_dim, hid_dim//2, num_layers=n_layers, bidirectional=True, dropout=dropout)
        else:
            raise ValueError(f'RNN type must be LSTM or GRU, got {rnn_type}')

        self.dropout = nn.Dropout()

        self.apply(initialize_rnn)

    def forward(self, src, src_len, src_mask):

        #src = [src len, batch size]
        #src_len = [batch size]
        #src_mask = [src len, batch size]

        embedded = self.dropout(self.embedding(src))

        #embedded = [src len, batch size, hid dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len, enforce_sorted=False)

        packed_outputs, _ = self.rnn(packed_embedded)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        #output = [seq len, batch, hid dim]

        outputs = outputs.permute(1, 0, 2)

        #src = [batch size, src len, hid dim]

        return outputs