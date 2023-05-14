import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj, epoach):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, epoach) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj, epoach))
        return F.log_softmax(x, dim=1)
    
    # 不适合直接替换第一个 F.dropout, x此时是独热编码；单独替换第二个时，效果也不好，有待继续探究
    def Dropkey(self, x):
        m_r = torch.ones_like(x) * self.dropout
        return x + torch.bernoulli(m_r) * (-1e12)


