import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeGATLayer(nn.Module):
    def __init__(self, in_features_op, in_features_ma, in_features_edge, out_features):
        super(EdgeGATLayer, self).__init__()
        self.weight_op = nn.Parameter(torch.Tensor(in_features_op, out_features))
        self.weight_ma = nn.Parameter(torch.Tensor(in_features_ma, out_features))
        self.weight_edge = nn.Parameter(torch.Tensor(in_features_edge, out_features))
        self.attention = nn.Parameter(torch.Tensor(1, 3 * out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_op)
        nn.init.xavier_uniform_(self.weight_ma)
        nn.init.xavier_uniform_(self.weight_edge)
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x, y, z, op_ma_adj):
        batch_size, op_nodes, _ = x.shape
        ma_nodes = y.shape[1]
        # (batch_size,ma_nodes,op_nodes,out_features)
        h_op = torch.einsum('bni,io->bno', x, self.weight_op).unsqueeze(1).expand(-1, ma_nodes, -1, -1)
        # (batch_size,ma_nodes,out_features)
        h = torch.einsum('bni,io->bno', y, self.weight_ma)
        # (batch_size,ma_nodes,op_nodes,out_features)
        h_ma = h.unsqueeze(-2).expand_as(h_op)
        # (batch_size,ma_nodes,op_nodes,out_features)
        h_edge = torch.einsum('bnmi,io->bnmo', z, self.weight_edge).transpose(1, 2)
        # (batch_size,ma_nodes,op_nodes,3*out_features)
        all_comb = torch.cat([h_op, h_ma, h_edge], dim=-1)
        # (batch_size,ma_nodes,op_nodes)
        e_ma = torch.matmul(all_comb, self.attention.t()).view(batch_size, ma_nodes, op_nodes)
        e_ma = F.leaky_relu(e_ma)
        adj = copy.deepcopy(op_ma_adj).transpose(1, 2)
        mask = torch.all(adj == 0, dim=-1).unsqueeze(-1).expand_as(e_ma)  # 指示调度完成也未安排工序的机器
        attention = torch.where(adj > 0, e_ma, float('-inf'))
        attention[mask] = 0
        attention = F.softmax(attention, dim=-1)
        h_op_edge = h_op + h_edge
        # (batch, ma_nodes, 1, op_nodes) * (batch_size,ma_nodes,op_nodes,out_features)
        # 执行批量矩阵乘法
        h_prime = torch.matmul(attention.unsqueeze(-2), h_op_edge).squeeze(-2)
        mask = torch.all(adj == 0, dim=-1).unsqueeze(-1).expand_as(h_prime)  # 指示调度完成也未安排工序的机器
        h_prime[mask] = 0
        h_prime += h
        embed = F.elu(h_prime)
        return embed


class EmdedMA(nn.Module):
    def __init__(self, in_features_op, in_features_ma, in_features_edge, out_features):
        super(EmdedMA, self).__init__()
        self.gat_layer = EdgeGATLayer(in_features_op, in_features_ma, in_features_edge, out_features)

    def forward(self, x, y, z, op_ma_adj):
        vi = self.gat_layer(x, y, z, op_ma_adj)
        return vi
