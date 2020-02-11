import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModelHead(nn.Module):
    def __init__(self,
                 vocab_size,
                 hid_dim):
        super().__init__()

        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, x):

        # x = [batch size, src len, hid dim]

        x = self.fc(x)

        return x
