import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, JumpingKnowledge


class CFLP(nn.Module):
    def __init__(self, dim_feat, dim_h, dim_z, dropout, gnn_type='GCN', jk_mode='mean', dec='hadamard'):
        super(CFLP, self).__init__()
        gcn_num_layers = 3
        self.encoder = GNN(dim_feat, dim_h, dim_z, dropout, gnn_type=gnn_type, num_layers = gcn_num_layers, jk_mode=jk_mode)
        if jk_mode == 'cat':
            dim_in = dim_h * (gcn_num_layers-1) + dim_z
        else:
            dim_in = dim_z
        self.decoder = Decoder(dec, dim_in, dim_h=dim_h)
        self.init_params()

    def forward(self, adj, features, edges, T_f_batch, T_cf_batch):
        z = self.encoder(adj, features)
        z_i = z[edges.T[0]]
        z_j = z[edges.T[1]]
        logits_f = self.decoder(z_i, z_j, T_f_batch)
        logits_cf = self.decoder(z_i, z_j, T_cf_batch)
        return z, logits_f, logits_cf

    def init_params(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()


class GNN(nn.Module):
    def __init__(self, dim_feat, dim_h, dim_z, dropout, gnn_type='GCN', num_layers=3, jk_mode='mean', batchnorm=True):
        super(GNN, self).__init__()

        assert jk_mode in ['max','sum','mean','lstm','cat','none']
        self.act = nn.ELU()
        self.dropout = dropout
        self.linear = torch.nn.Linear(dim_h, dim_z)

        if gnn_type == 'SAGE':
            gnnlayer = SAGEConv
        elif gnn_type == 'GCN':
            gnnlayer = GCNConv
        self.convs = torch.nn.ModuleList()
        self.convs.append(gnnlayer(dim_feat, dim_h))
        for _ in range(num_layers - 2):
            self.convs.append(gnnlayer(dim_h, dim_h))
        self.convs.append(gnnlayer(dim_h, dim_z))

        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(dim_h) for _ in range(num_layers)])

        self.jk_mode = jk_mode
        if self.jk_mode in ['max', 'lstm', 'cat']:
            self.jk = JumpingKnowledge(mode=self.jk_mode, channels=dim_h, num_layers=num_layers)
        elif self.jk_mode == 'mean':
            self.weights = torch.nn.Parameter(torch.randn((len(self.convs))))

    def forward(self, adj, features):
        out = features
        out_list = []

        for i in range(len(self.convs)):
            out = self.convs[i](out, adj)
            if self.batchnorm:
                 out = self.bns[i](out)
            out = self.act(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            out_list += [out]

        if self.jk_mode in ['max', 'lstm', 'cat']:
            out = self.jk(out_list)
        elif self.jk_mode == 'mean':
            sftmax = F.softmax(self.weights, dim=0)
            for i in range(len(out_list)):
                out_list[i] = out_list[i] * sftmax[i]
                out = sum(out_list)
        elif self.jk_mode == 'sum':
            out_stack = torch.stack(out_list, dim=0)
            out = torch.sum(out_stack, dim=0)
        elif self.jk_mode == 'none':
            out = out_list[-1]
        return out

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()
        if self.jk_mode in ['max', 'lstm', 'cat']:
            self.jk.reset_parameters()


class Decoder(nn.Module):
    def __init__(self, dec, dim_z, dim_h=64):
        super(Decoder, self).__init__()
        self.dec = dec
        if dec == 'innerproduct':
            dim_in = 2
        elif dec == 'hadamard':
            dim_in = dim_z + 1
        elif dec == 'mlp':
            dim_in = 1 + 2*dim_z
        self.mlp_out = nn.Sequential(
            nn.Linear(dim_in, dim_h, bias=True),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(dim_h, 1, bias=False)
        )

    def forward(self, z_i, z_j, T):
        if self.dec == 'innerproduct':
            z = (z_i * z_j).sum(1).view(-1, 1)
            h = torch.cat((z, T.view(-1, 1)), dim=1)
        elif self.dec == 'mlp':
            h = torch.cat((z_i, z_j, T.view(-1, 1)), dim=1)
        elif self.dec == 'hadamard':
            z = z_i * z_j
            h = torch.cat((z, T.view(-1, 1)), dim=1)
        h = self.mlp_out(h).squeeze()
        return h

    def reset_parameters(self):
        for lin in self.mlp_out:
            try:
                lin.reset_parameters()
            except:
                continue
