import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
path = "D:/MADDPG/GAT"


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # torch.empty 创建一个未初始化的矩阵
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, epoach):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        if epoach == 99:
            if not os.path.exists(path):  # 如果不存在路径，则创建这个路径
                os.makedirs(path)
            self.show_heatmaps(torch.unsqueeze(e,0).unsqueeze(0), 'x', 'y')
            plt.savefig(path + '/' + 'attention_origion' + '.jpg')
            plt.close()
        attention = torch.where(adj > 0, e, zero_vec)
        if epoach == 99:
            self.show_heatmaps(torch.unsqueeze(attention, 0).unsqueeze(0), 'x', 'y')
            plt.savefig(path + '/' + 'attention_afterwhere' + '.jpg')
            plt.close()
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        # torch.matmul 矩阵乘法
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

    def show_heatmaps(self, matrices, xlabel, ylabel, titles=None, figsize=(30, 30), cmap='Reds'):
        """显示矩阵热图"""
        num_rows, num_cols = matrices.shape[0], matrices.shape[1]
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
        for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
            for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
                pcm = ax.imshow(matrix.detach().cpu().numpy(), cmap=cmap)
                if i == num_rows - 1:
                    ax.set_xlabel(xlabel)
                if j == 0:
                    ax.set_ylabel(ylabel)
                if titles:
                    ax.set_title(titles[j])
        fig.colorbar(pcm, ax=axes, shrink=0.6)

