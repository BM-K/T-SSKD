import math
import torch
import torch.nn as nn


class AuxRecoveryMask(nn.Module):
    def __init__(self, args, d_model, vocab_size=0):
        super(AuxRecoveryMask, self).__init__()

        self.decoder = nn.Linear(d_model, vocab_size)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, logits, auxiliary):
        masked_pos = auxiliary['mask']['position'][:, :, None].expand(-1, -1, logits.size(-1))
        h_masked = torch.gather(logits, 1, masked_pos)
        h_masked = self.decoder(self.norm(self.gelu(self.linear(h_masked))))

        return h_masked

    def gelu(self, x):
        "Implementation of the gelu activation function by Hugging Face"
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
