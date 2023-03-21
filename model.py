import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, _, x):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)

        x = self.act_fn(x)
        x = self.layer2(x)

        return x


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x):

        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](graph, x))
        x = self.convs[-1](graph, x)

        return x


class PMGAE(nn.Module):
    def __init__(self,in_dim, hid_dim, out_dim,n_layers=2):
        super(PMGAE, self).__init__()

        self.encoder1 =GCN(in_dim, hid_dim, out_dim,n_layers=n_layers)
        self.encoder2 =GCN(in_dim, hid_dim, out_dim,n_layers=n_layers)

    def get_embedding(self,graph,raw):
        code1 = self.encoder1(graph, raw)
        code2 = self.encoder2(graph, raw)
        code=torch.cat((code1,code2),dim=1)
        return code.detach()

    def forward(self,graph1,raw1,graph2,raw2,index):
        code1= self.encoder1(graph1,raw1)
        code2 = self.encoder2(graph2, raw2)
        code1=code1[index]
        code2=code2[index]
        code1 = F.normalize(code1, dim=1)
        code2=F.normalize(code2,dim=1)

        before = torch.cuda.memory_allocated()

        z1=torch.mm(code1,code1.T)
        z2=torch.mm(code2, code2.T)
        z3 = torch.mm(code1, code2.T)
        z3 = (z3 + z3.T)*0.5

        z=(z1+z2+z3)*(1/3)

        after =  torch.cuda.memory_allocated()

        z= torch.triu(z, diagonal=0)

        return z,after-before
