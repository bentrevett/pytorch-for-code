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
                mask):

        #embeddings = [batch, seq len, emb_dim]
        #mask = [batch, seq len]

        _, seq_len, _ = embeddings.shape

        mask = mask.unsqueeze(1)

        #mask = [batch, 1, seq len]

        #_embeddings = [batch, seq len, emb dim]
        weights = torch.sigmoid(self.fc(embeddings))
        #weighs = [batch, seq len, 1]
        weights = weights.permute(0, 2, 1)
        #weights = [batch, 1, seq len]
        weighted = embeddings * weights
        weighted = weighted.masked_fill(mask == 0, 0)
        #weighted = [batch, emb dim, seq len]
        pooled = F.avg_pool1d(weighted,
                              kernel_size = seq_len)

        #pooled = [batch size, emb dim, 1]

        pooled = pooled.squeeze(-1)

        #pooled = [batch size, emb dim]

        return pooled
