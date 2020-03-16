import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingPooler(nn.Module):
    def __init__(self,
                 emb_dim):
        super().__init__()

        self.fc = nn.Linear(emb_dim, 1, bias = False)

    def forward(self, 
                embeddings,
                embeddings_len,
                mask):

        #embeddings = [batch, seq len, emb_dim]
        #mask = [seq len, batch]

        _, seq_len, _ = embeddings.shape

        mask = mask.permute(1, 0).unsqueeze(-1)

        #mask = [batch, seq len, 1]

        weights = torch.sigmoid(self.fc(embeddings))
        #weighs = [batch, seq len, 1]
        weighted = embeddings * weights
        #weighted = [batch, seq len, emb dim]
        weighted = weighted.masked_fill(mask == 0, 0)
        #weighted = [batch, seq len, emb dim]
        weighted = weighted.permute(0, 2, 1)
        #weighted = [batch, emb dim, seq len]
        pooled = F.avg_pool1d(weighted,
                              kernel_size = seq_len)

        #pooled = [batch size, emb dim, 1]

        pooled = pooled.squeeze(-1)

        #pooled = [batch size, emb dim]

        return pooled, weights
