import torch
import torch.nn as nn
import torch.nn.functional as F

class vanilla_lstnet(nn.Module):
    def __init__(self, in_dim=12, window=16, meta_dim=48):
        super(vanilla_lstnet, self).__init__()
        self.P = window
        self.m = in_dim
        self.hidR = 64
        self.hidC = 64
        self.hidS = 64
        self.Ck = in_dim
        self.skip = 4
        self.pt = (self.P - self.Ck) / self.skip
        self.hw = 64

        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m), padding='same')
        self.linear = nn.Linear(self.m * self.hidC, self.hidC)
        self.GRU1 = nn.GRU(self.hidC, self.hidR, batch_first=True)
        self.dropout = nn.Dropout(p = 0.2)
   
        self.GRUskip = nn.GRU(int(self.pt * self.skip), self.hidS, batch_first=True)
        self.linear1 = nn.Linear(self.hidR + self.hidS + meta_dim, self.m)

        self.highway = nn.GRU(self.m, self.hw, batch_first=True)
        self.linearhw = nn.Linear(self.hw + meta_dim, self.m)

        self.outlinear= nn.Linear(self.m, 1)

    def forward(self, x, meta=None):
        if meta is None:
            batch_size = x.size(0)
            
            #CNN
            c = x.view(-1, 1, self.P, self.m)
            c = F.relu(self.conv1(c))
            c = self.dropout(c)
            c = c.view(batch_size, self.P, self.m*self.hidC)
            c = self.linear(c)

            # RNN 
            r = self.GRU1(c)[0]
            if batch_size != 1:
                r = self.dropout(torch.squeeze(r,0))
            else:
                r = self.dropout(r)

            #skip-rnn
            s = c[:,:,int(-self.pt * self.skip):].contiguous()
            s = self.GRUskip(s)[0]
            s = self.dropout(s)
            r = torch.cat((r,s),2)

            res = self.linear1(r)
            
            #highway
            z = self.highway(x)[0]
            z = self.linearhw(z)
            res = res + z

            res = self.outlinear(torch.tanh(res))
            return torch.squeeze(res, dim=-1)
        
        else:
            batch_size = x.size(0)
            
            #CNN
            c = x.view(-1, 1, self.P, self.m)
            c = F.relu(self.conv1(c))
            c = self.dropout(c)
            c = c.view(batch_size, self.P, self.m*self.hidC)
            c = self.linear(c)

            # RNN 
            r = self.GRU1(c)[0]
            if batch_size != 1:
                r = self.dropout(torch.squeeze(r,0))
            else:
                r = self.dropout(r)

            #skip-rnn
            s = c[:,:,int(-self.pt * self.skip):].contiguous()
            s = self.GRUskip(s)[0]
            s = self.dropout(s)
            r = torch.cat((r,s,meta),2)

            res = self.linear1(r)
            
            #highway
            z = self.highway(x)[0]
            z = self.linearhw(torch.cat((z,meta), dim=-1))
            res = res + z

            res = self.outlinear(torch.tanh(res))
            return torch.squeeze(res, dim=-1)
    

class perform_lstnet(nn.Module):
    def __init__(self, in_dim=12, window=16, meta_dim=48, perf_dim=6):
        super(perform_lstnet, self).__init__()
        self.P = window
        self.m = in_dim
        self.hidR = 64
        self.hidC = 64
        self.hidS = 64
        self.Ck = in_dim
        self.skip = 4
        self.pt = (self.P - self.Ck) / self.skip
        self.hw = 64

        self.perf_dim = perf_dim

        self.gru1 = nn.GRU(input_size=perf_dim, hidden_size=64, num_layers=4, bidirectional=True, batch_first=True, dropout=0.2)
        linear1 = [
            nn.Linear(in_features=128 + meta_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=perf_dim)
        ]
        self.linear1 = nn.Sequential(*linear1)

        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m), padding='same')
        self.linear = nn.Linear(self.m * self.hidC, self.hidC)
        self.GRU1 = nn.GRU(self.hidC, self.hidR, batch_first=True)
        self.dropout = nn.Dropout(p = 0.2)
   
        self.GRUskip = nn.GRU(int(self.pt * self.skip), self.hidS, batch_first=True)
        self.linearskip = nn.Linear(self.hidR + self.hidS + meta_dim, self.m)

        self.highway = nn.GRU(self.m, self.hw, batch_first=True)
        self.linearhw = nn.Linear(self.hw + meta_dim, self.m)

        self.outlinear= nn.Linear(self.m, 1)

    def forward(self, x, meta=None):
        if meta is None:
            batch_size = x.size(0)

            latent = self.gru1(x[:,:,:self.perf_dim])[0]
            out1 = self.linear1(latent)
            x = torch.cat((out1, x[:,:,self.perf_dim:]), dim=-1)
            
            #CNN
            c = x.view(-1, 1, self.P, self.m)
            c = F.relu(self.conv1(c))
            c = self.dropout(c)
            c = c.view(batch_size, self.P, self.m*self.hidC)
            c = self.linear(c)

            # RNN 
            r = self.GRU1(c)[0]
            if batch_size != 1:
                r = self.dropout(torch.squeeze(r,0))
            else:
                r = self.dropout(r)

            #skip-rnn
            s = c[:,:,int(-self.pt * self.skip):].contiguous()
            s = self.GRUskip(s)[0]
            s = self.dropout(s)
            r = torch.cat((r,s),2)
            res = self.linearskip(r)
            
            #highway
            z = self.highway(x)[0]
            z = self.linearhw(z)
            res = res + z

            res = self.outlinear(torch.tanh(res))
            return out1, torch.squeeze(res, dim=-1)
        else:
            batch_size = x.size(0)

            latent = self.gru1(x[:,:,:self.perf_dim])[0]
            out1 = self.linear1(torch.cat((latent, meta), dim=-1))
            x = torch.cat((out1, x[:,:,self.perf_dim:]), dim=-1)
            
            #CNN
            c = x.view(-1, 1, self.P, self.m)
            c = F.relu(self.conv1(c))
            c = self.dropout(c)
            c = c.view(batch_size, self.P, self.m*self.hidC)
            c = self.linear(c)

            # RNN 
            r = self.GRU1(c)[0]
            if batch_size != 1:
                r = self.dropout(torch.squeeze(r,0))
            else:
                r = self.dropout(r)

            #skip-rnn
            s = c[:,:,int(-self.pt * self.skip):].contiguous()
            s = self.GRUskip(s)[0]
            s = self.dropout(s)
            r = torch.cat((r,s,meta),2)
            res = self.linearskip(r)
            
            #highway
            z = self.highway(x)[0]
            z = self.linearhw(torch.cat((z, meta), dim=-1))
            res = res + z

            res = self.outlinear(torch.tanh(res))
            return out1, torch.squeeze(res, dim=-1)