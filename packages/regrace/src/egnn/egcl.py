"""
Code from https://github.com/vgsatorras/egnn
"""
import torch
from torch import nn


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class E_GCL(nn.Module):

    def __init__(
            self,
            input_nf,  # number of input features
            output_nf,  # number of output features
            hidden_nf,  # number of hidden features
            edges_in_d=0,  # number of edge features
            act_fn=nn.SiLU(),
            residual=True,
            attention=False,
            normalize=False,
            coords_agg='mean',
            tanh=False):
        super(E_GCL, self).__init__()

        # define the number of input features for edge model
        input_edge = input_nf * 2  # input features for edge model
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1  # dimension of distance vector

        # define the edge and node models
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        # define the coordinate model
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer,
            nn.Tanh() if self.tanh else nn.Identity(),
        )

        # define the attention model
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        # add edge attributes if they are present
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        # pass through the edge model
        out = self.edge_mlp(out)
        # apply attention if needed
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        # aggregate features
        row, _ = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        # add node attributes if they are present
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        # pass through the node model
        out = self.node_mlp(agg)
        # add residual connection
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        # encode the edge features into the coordinates
        row, _ = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        # aggregate the transformed coordinates
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        # transform the coordinates into radial coordinates
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2,
                           1).unsqueeze(1)  # distance between nodes

        if self.normalize:
            norm = torch.sqrt(
                radial).detach() + self.epsilon  # avoid division by zero
            coord_diff = coord_diff / norm  # normalize the distance vector

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr):
        # make sure the edge index is correct given torch_geometric conventions
        edge_index = edge_index[[1, 0]]
        row, col = edge_index
        # transform the coordinates into radial coordinates
        radial, coord_diff = self.coord2radial(edge_index, coord)
        # pass through the edge, coordinate and node models
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, _ = self.node_model(h, edge_index, edge_feat, None)

        return h, coord

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
