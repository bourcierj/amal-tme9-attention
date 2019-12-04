"""Different models with attention."""

import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """
    Simple linear model.
    Represents an input as the mean of its embeddings, and pass it
    to a simple linear layer.
    """
    def __init__(self, glove_embeddings, num_classes=2):

        super(BaseModel, self).__init__()
        glove_embeddings = torch.as_tensor(glove_embeddings, dtype=torch.float)
        num_embs, emb_size = glove_embeddings.size()
        self.embedding = torch.nn.Embedding.from_pretrained(glove_embeddings,
                                                            freeze=True)
        self.lin = nn.Linear(emb_size, num_classes)

    def forward(self, x, lengths):

        # get mean embeddings vectors for each batch sample
        embed = self.embedding(x)
        mean = torch.sum(embed, 0) / lengths[:, None]
        out = self.lin(mean)
        return out

        # mask = torch.stack([torch.arange(seq_length) < l for l in lengths], 1)
        # torch.sum(out * mask, 0).div(lengths).squeeze(0)
