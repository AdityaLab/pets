import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_length=100):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, input_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2).float() * (-math.log(10000.0) / input_dim))

        if input_dim % 2 == 0:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        
        if input_dim % 2 == 1:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return x
    


class vanilla_transformer(nn.Module):
    def __init__(self, in_dim=12, window=16, meta_dim=48):
        super(vanilla_transformer, self).__init__()

        self.embedding = nn.GRU(input_size=in_dim, hidden_size=64, num_layers=4,\
            batch_first=True, dropout=0.2)
        self.positional_encoding = PositionalEncoding(64)

        self.encoder_layers = nn.TransformerEncoderLayer(d_model=64 + meta_dim, \
            nhead=8, dropout=0.2, batch_first=True)

        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=3)

        linear = [
            nn.Linear(in_features=64 + meta_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        ]
        self.linear = nn.Sequential(*linear)

    def forward(self, x, meta=None):
        x = self.embedding(x)[0]
        x = self.positional_encoding(x)
        if meta is None:
            x = self.encoder(x)
        else:
            x = self.encoder(torch.cat((x, meta), dim=-1))
        x = self.linear(x)
        return torch.squeeze(x, dim=-1)
    


class perform_transformer(nn.Module):
    def __init__(self, in_dim=12, window=16, meta_dim=48, perf_dim=6):
        super(perform_transformer, self).__init__()
        self.perf_dim = perf_dim

        self.gru1 = nn.GRU(input_size=perf_dim, hidden_size=64, num_layers=4, bidirectional=True, batch_first=True, dropout=0.2)
        linear1 = [
            nn.Linear(in_features=128 + meta_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=perf_dim)
        ]
        self.linear1 = nn.Sequential(*linear1)


        self.embedding = nn.GRU(input_size=in_dim, hidden_size=64, num_layers=4,\
            batch_first=True, dropout=0.2)
        self.positional_encoding = PositionalEncoding(64)

        self.encoder_layers = nn.TransformerEncoderLayer(d_model=64 + meta_dim, \
            nhead=8, dropout=0.2, batch_first=True)

        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=3)

        linear2 = [
            nn.Linear(in_features=64 + meta_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        ]
        self.linear2 = nn.Sequential(*linear2)

    def forward(self, x, meta=None):
        if meta is None:
            latent = self.gru1(x[:,:,:self.perf_dim])[0]
            out1 = self.linear1(latent)
            x = self.embedding(torch.cat((out1, x[:,:,self.perf_dim:]), dim=-1))[0]
            x = self.positional_encoding(x)
            x = self.encoder(x)
            out2 = self.linear2(x)
            return out1, torch.squeeze(out2, dim=-1)
        
        else:
            latent = self.gru1(x[:,:,:self.perf_dim])[0]
            out1 = self.linear1(torch.cat((latent, meta), dim=-1))
            x = self.embedding(torch.cat((out1, x[:,:,self.perf_dim:]), dim=-1))[0]
            x = self.positional_encoding(x)
            x = self.encoder(torch.cat((x, meta), dim=-1))
            out2 = self.linear2(x)
            return out1, torch.squeeze(out2, dim=-1)