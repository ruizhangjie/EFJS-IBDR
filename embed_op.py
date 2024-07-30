import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class HetGATLayer(nn.Module):
    def __init__(self, in_features_op, in_features_ma, in_features_edge, out_features, hidden_size):
        super(HetGATLayer, self).__init__()
        self.weight_op = nn.Parameter(torch.Tensor(in_features_op, out_features))
        self.weight_ma = nn.Parameter(torch.Tensor(in_features_ma, out_features))
        self.weight_edge = nn.Parameter(torch.Tensor(in_features_edge, out_features))
        self.attention_op = nn.Parameter(torch.Tensor(1, 2 * out_features))
        self.attention_ma = nn.Parameter(torch.Tensor(1, 3 * out_features))
        self.hidden1 = nn.Linear(2 * out_features, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_op)
        nn.init.xavier_uniform_(self.weight_ma)
        nn.init.xavier_uniform_(self.weight_edge)
        nn.init.xavier_uniform_(self.attention_op)
        nn.init.xavier_uniform_(self.attention_ma)
        nn.init.orthogonal_(self.hidden1.weight, gain=1.0)  # default gain for tanh
        nn.init.zeros_(self.hidden1.bias)
        nn.init.orthogonal_(self.hidden2.weight, gain=1.0)  # default gain for tanh
        nn.init.zeros_(self.hidden2.bias)
        nn.init.orthogonal_(self.output.weight, gain=1.0)  # default gain for tanh
        nn.init.zeros_(self.output.bias)

    def forward(self, x, y, z, op_adj_in, ma_adj_in, op_ma_adj):
        # x: [Batch, N, in_features]
        # op_adj_in, ma_adj_in: [Batch, N, N]
        batch_size, op_nodes, _ = x.shape
        ma_nodes = y.shape[1]

        # 计算op邻居贡献
        # 特征变换
        h = torch.einsum('bni,io->bno', x, self.weight_op)
        h_repeat_in_dim2 = h.repeat_interleave(op_nodes, dim=1)
        h_repeat_in_dim1 = h.repeat(1, op_nodes, 1)
        all_combinations = torch.cat([h_repeat_in_dim1, h_repeat_in_dim2], dim=2).view(batch_size, op_nodes,
                                                                                       op_nodes, -1)
        e_op = torch.matmul(all_combinations, self.attention_op.t()).view(batch_size, op_nodes, op_nodes)
        e_op = F.leaky_relu(e_op)
        # 考虑工序顺序和机器顺序的入和出邻接矩阵
        adj = (torch.transpose(op_adj_in, 1, 2) + op_adj_in + torch.transpose(ma_adj_in, 1, 2) +
               ma_adj_in)
        mask = torch.all(op_ma_adj == 0, dim=-1).unsqueeze(-1).expand_as(adj)  # 指示padding的op
        adj[mask] = True
        # 计算注意力系数
        attention = torch.where(adj > 0, e_op, float('-inf'))
        attention_in_op = F.softmax(attention, dim=2)
        # Combine attentions and aggregate neighbor information
        h_prime_op = torch.bmm(attention_in_op, h)


        # 计算ma邻居贡献
        h_op = h.unsqueeze(-2).expand(-1, -1, ma_nodes, -1)
        h_ma = torch.einsum('bni,io->bno', y, self.weight_ma).unsqueeze(1).expand_as(h_op)
        h_edge = torch.einsum('bnmi,io->bnmo', z, self.weight_edge)
        all_comb = torch.cat([h_op, h_ma, h_edge], dim=-1)
        e_ma = torch.matmul(all_comb, self.attention_ma.t()).view(batch_size, op_nodes, ma_nodes)
        e_ma = F.leaky_relu(e_ma)
        adj = copy.deepcopy(op_ma_adj)
        mask = torch.all(op_ma_adj == 0, dim=-1).unsqueeze(-1).expand_as(adj)  # 指示padding的op
        adj[mask] = 1
        attention = torch.where(adj > 0, e_ma, float('-inf'))
        attention_in_ma = F.softmax(attention, dim=2)
        h_ma_edge = h_ma + h_edge
        attention_in_ma_ex = attention_in_ma.unsqueeze(-2)  # 形状变为(batch,op_nodes, 1, ma_nodes)
        # 执行批量矩阵乘法
        h_prime_ma = torch.matmul(attention_in_ma_ex, h_ma_edge).squeeze(-2) + h

        # 级联不同邻居并进行特征转换
        h_prime = torch.cat([h_prime_op, h_prime_ma], dim=-1)
        embed = F.elu(self.hidden1(h_prime))
        embed = F.elu(self.hidden2(embed))
        embed = self.output(embed)
        mask = torch.all(op_ma_adj == 0, dim=-1).unsqueeze(-1).expand_as(embed)
        embed[mask] = 0
        return embed


class EmdedOP(nn.Module):
    def __init__(self, in_features_op, in_features_ma, in_features_edge, out_features, hidden_size):
        super(EmdedOP, self).__init__()
        self.gat_layer = HetGATLayer(in_features_op, in_features_ma, in_features_edge, out_features, hidden_size)

    def forward(self, x, y, z, op_adj_in, ma_adj_in, op_ma_adj):
        ui = self.gat_layer(x, y, z, op_adj_in, ma_adj_in, op_ma_adj)
        return ui
