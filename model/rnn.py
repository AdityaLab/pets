import numpy as np
import torch
import torch.nn as nn

class vanilla_rnn(nn.Module):
    def __init__(self, in_dim=12, window=16, meta_dim=48):
        super(vanilla_rnn, self).__init__()

        self.gru = nn.GRU(input_size=in_dim, hidden_size=64, num_layers=4, bidirectional=True, batch_first=True, dropout=0.2)
        linear = [
            nn.Linear(in_features=128+meta_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        ]
        self.linear = nn.Sequential(*linear)

    def forward(self, x, meta=None):
        latent = self.gru(x)[0]
        if meta is None:
            out = self.linear(latent)
        else:
            out = self.linear(torch.cat((latent, meta), dim=-1))
        return torch.squeeze(out, dim=-1)


class perform_rnn(nn.Module):
    def __init__(self, in_dim=12, window=16, meta_dim=48, perf_dim=6):
        super(perform_rnn, self).__init__()

        self.perf_dim = perf_dim

        self.gru1 = nn.GRU(input_size=perf_dim, hidden_size=64, num_layers=4, bidirectional=True, batch_first=True, dropout=0.2)
        linear1 = [
            nn.Linear(in_features=128+meta_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=perf_dim)
        ]
        self.linear1 = nn.Sequential(*linear1)


        self.gru2 = nn.GRU(input_size=in_dim, hidden_size=64, num_layers=4, bidirectional=True, batch_first=True, dropout=0.2)
        linear2 = [
            nn.Linear(in_features=128+meta_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        ]
        self.linear2 = nn.Sequential(*linear2)

    def forward(self, x, meta=None):
        if meta is None:
            latent = self.gru1(x[:,:,:self.perf_dim])[0]
            out1 = self.linear1(latent)
            latent = self.gru2(torch.cat((out1, x[:,:,self.perf_dim:]), dim=-1))[0]
            out2 = self.linear2(latent)
            return out1, torch.squeeze(out2, dim=-1)
        else:
            latent = self.gru1(x[:,:,:self.perf_dim])[0]
            out1 = self.linear1(torch.cat((latent, meta), dim=-1))
            latent = self.gru2(torch.cat((out1, x[:,:,self.perf_dim:]), dim=-1))[0]
            out2 = self.linear2(torch.cat((latent, meta), dim=-1))
            return out1, torch.squeeze(out2, dim=-1)
