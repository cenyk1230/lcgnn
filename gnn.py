
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv, GATConv


class ConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats, conv_type, activation=None,
                 residual=True, batchnorm=True, dropout=0.,
                 num_heads=1, negative_slope=0.2):
        super(ConvLayer, self).__init__()

        self.activation = activation
        self.conv_type = conv_type
        if conv_type == 'gcn':
            self.graph_conv = GraphConv(in_feats=in_feats, out_feats=out_feats,
                                       norm='both', activation=activation)
        elif conv_type == 'sage':
            self.graph_conv = SAGEConv(in_feats=in_feats, out_feats=out_feats,
                                       aggregator_type='mean', norm=None, activation=activation)
        elif conv_type == 'gat':
            assert out_feats % num_heads == 0
            self.graph_conv = GATConv(in_feats=in_feats, out_feats=out_feats // num_heads, 
                                      num_heads=num_heads, feat_drop=dropout, attn_drop=dropout, 
                                      negative_slope=negative_slope, activation=activation)
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.graph_conv.reset_parameters()
        if self.residual:
            self.res_connection.reset_parameters()
        if self.bn:
            self.bn_layer.reset_parameters()

    def forward(self, g, feats):
        new_feats = self.graph_conv(g, feats)
        if self.conv_type == 'gat':
            new_feats = new_feats.view(new_feats.shape[0], -1)
        if self.residual:
            res_feats = self.res_connection(feats)
            if self.activation is not None:
                res_feats = self.activation(res_feats)
            new_feats = new_feats + res_feats
        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats


class GNNModel(nn.Module):
    def __init__(
        self,
        conv_type,
        input_size=128,
        hidden_size=64,
        num_layers=2,
        num_classes=40,
        idropout=0.0,
        dropout=0.0,
        batch_norm=False,
        residual=False,
        linear_layer=True,
        norm='none',
        num_heads=1,
    ):
        super(GNNModel, self).__init__()
        self.dropout = nn.Dropout(p=idropout)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                ConvLayer(
                    in_feats=hidden_size if i > 0 else input_size,
                    out_feats=hidden_size,
                    conv_type=conv_type,
                    activation=F.gelu,
                    residual=residual,
                    batchnorm=batch_norm,
                    dropout=dropout,
                    num_heads=num_heads,
                )
            )
        self.linear_layer = linear_layer 
        if self.linear_layer is None:
            # None for returning hidden state of GNN's final layer
            pass
        elif self.linear_layer:
            self.linear = nn.Linear(hidden_size, num_classes)
        else:
            self.layers.append(
                ConvLayer(
                    in_feats=hidden_size,
                    out_feats=num_classes,
                    conv_type=conv_type,
                    activation=None,
                    residual=False,
                    batchnorm=False,
                    dropout=0.,
                    num_heads=1,
                )
            )
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.linear_layer:
            self.linear.reset_parameters()

    def forward(self, g):
        out = g.ndata['feat']
        out = self.dropout(out)
        for layer in self.layers:
            out = layer(g, out)

        batch_num_nodes = g.batch_num_nodes().tolist()
        out = torch.split(out, batch_num_nodes, dim=0)
        out = torch.nn.utils.rnn.pad_sequence(out, batch_first=True, padding_value=0.0)
        out = out[:, 0]
        if self.linear_layer:
            out = self.linear(out)
        return out


if __name__ == "__main__":
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 0, 0, 1, 1, 2], [0, 1, 1, 2, 2, 2])
    g.ndata['feat'] = torch.rand(3, 64)
    model = GNNModel('gcn', input_size=64)
    print(model)
    print(model(g).shape)
    model = GNNModel('sage', input_size=64)
    print(model)
    print(model(g).shape)
    model = GNNModel('gat', input_size=64, num_layers=2)
    print(model)
    print(model(g).shape)
