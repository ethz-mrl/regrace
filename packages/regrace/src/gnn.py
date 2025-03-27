from typing import Optional

import torch
import torch.nn.backends
import torch.utils.data
import torch_geometric.data
import torch_geometric.nn
import torch_geometric.utils

from .config import YAMLConfig
from .egnn.egcl import E_GCL
from .pooling import GeneralizedMeanPooling
from .sgpr.tensornn import TensorNetworkModule


class GNN(torch.nn.Module):

    def __init__(self,
                 degree_tensor: torch.Tensor,
                 embedding_size: int,
                 node_feature_size: int,
                 edge_feature_size: int,
                 node_hidden_size: int,
                 node_output_size: int,
                 edge_hidden_size: int,
                 edge_output_size: int,
                 gnn_output_size: int,
                 hidden_output_size: int,
                 n_towers: int,
                 n_conv_layers: int,
                 aggregators: list[str],
                 scalers: list[str],
                 pre_layers: int,
                 post_layers: int,
                 pooling: str,
                 conv_type: str,
                 nearest_neighbors: Optional[int] = None,
                 concat_semantics: bool = False,
                 use_TNN: bool = False):
        super().__init__()
        # save the parameters
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        
        # create the embedding MLPs prior to the GNN
        self.node_emb = torch.nn.Sequential(
            torch.nn.Linear(node_feature_size, node_hidden_size),
            torch.nn.BatchNorm1d(node_hidden_size), torch.nn.ReLU(),
            torch.nn.Linear(node_hidden_size, node_output_size))
        self.edge_emb = torch.nn.Sequential(
            torch.nn.Linear(edge_feature_size, edge_hidden_size),
            torch.nn.BatchNorm1d(edge_hidden_size), torch.nn.ReLU(),
            torch.nn.Linear(edge_hidden_size, edge_output_size))
        
        # create the GNN layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        layers_size = [
            node_output_size * pow(2, i) for i in range(n_conv_layers)
        ]
        layers_size.append(gnn_output_size)
        for i in range(n_conv_layers):
            if conv_type == 'pna':
                conv = torch_geometric.nn.PNAConv(in_channels=layers_size[i],
                                                  out_channels=layers_size[i +
                                                                           1],
                                                  aggregators=aggregators,
                                                  scalers=scalers,
                                                  deg=degree_tensor,
                                                  edge_dim=edge_output_size,
                                                  towers=n_towers,
                                                  pre_layers=pre_layers,
                                                  post_layers=post_layers,
                                                  divide_input=False)
            elif conv_type == 'egnn':
                assert edge_feature_size == 1, "EGNN only supports 1 edge feature, must be distance"
                conv = E_GCL(
                    input_nf=layers_size[i],  # input features
                    output_nf=layers_size[i + 1],  # output features
                    hidden_nf=layers_size[i],  # hidden features
                    edges_in_d=edge_output_size,  # number of edge features
                    act_fn=torch.nn.SiLU(),
                    residual=False,
                    attention=False,
                    normalize=False,
                    tanh=False,
                )
            elif conv_type == 'dynedgeconv':
                assert nearest_neighbors is not None, "Number of nearest neighbors must be provided to use 'edgeconv'"
                conv = torch_geometric.nn.conv.DynamicEdgeConv(
                    nn=torch.nn.Sequential(
                        torch.nn.Linear(layers_size[i] * 2,
                                        layers_size[i + 1]),
                        torch.nn.BatchNorm1d(layers_size[i + 1]),
                        torch.nn.ReLU(),
                    ),
                    k=nearest_neighbors,
                    aggr='max',
                )
            else:
                raise ValueError(
                    f"Convolution type {conv_type} not supported. Please choose between 'pna', 'egnn, 'dynedgeconv'"
                )
            self.convs.append(conv)
            self.batch_norms.append(
                torch_geometric.nn.BatchNorm(layers_size[i + 1]))
            
        # create the MLP after the GNN
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(gnn_output_size, hidden_output_size),
            torch.nn.BatchNorm1d(hidden_output_size), torch.nn.ReLU(),
            torch.nn.Linear(hidden_output_size, embedding_size))
        self.semantic_emb = torch.nn.Sequential(
            torch.nn.Linear(20, 64), torch.nn.BatchNorm1d(64), torch.nn.ReLU(),
            torch.nn.Linear(64, 128)) if concat_semantics else None
        self.semantic_output_mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_size + 128, embedding_size),
            torch.nn.BatchNorm1d(embedding_size), torch.nn.ReLU(),
            torch.nn.Linear(embedding_size,
                            embedding_size)) if concat_semantics else None
        
        # create the TensorNN
        self.TNN = TensorNetworkModule(
            embedding_size,
            int(embedding_size / 2),
            int(embedding_size / 2),
        ) if use_TNN else None
        
        # create pooling layer
        if pooling == 'genmean':
            self.global_pool = GeneralizedMeanPooling()
        elif pooling == 'global_mean':
            self.global_pool = torch_geometric.nn.global_mean_pool
        elif pooling == 'global_add':
            self.global_pool = torch_geometric.nn.global_add_pool
        else:
            raise ValueError(
                f"Pooling {pooling} not supported. Please choose between 'genmean', 'global_mean', 'global_add'"
            )

    @classmethod
    def from_config(cls, config: YAMLConfig, degree_tensor: torch.Tensor):
        return cls(degree_tensor=degree_tensor,
                   n_conv_layers=config.n_conv_layers,
                   embedding_size=config.output_mpl_output_layer_size,
                   node_feature_size=config.node_features,
                   edge_feature_size=config.edge_features,
                   node_hidden_size=config.node_mpl_hidden_layer_size,
                   node_output_size=config.node_mpl_output_layer_size,
                   edge_hidden_size=config.edge_mpl_hidden_layer_size,
                   edge_output_size=config.edge_mpl_output_layer_size,
                   gnn_output_size=config.gnn_output_layer_size,
                   hidden_output_size=config.output_mpl_hidden_layer_size,
                   n_towers=config.gnn_n_towers,
                   aggregators=config.pna_aggr,
                   scalers=config.pna_scaler,
                   pre_layers=config.gnn_pre_mpl_n_layers,
                   post_layers=config.gnn_post_mpl_n_layers,
                   pooling=config.pooling,
                   conv_type=config.conv_type,
                   nearest_neighbors=config.k_nearest_neighbors,
                   concat_semantics=config.use_semantics_in_graph_features,
                   use_TNN=(config.loss_type in ['bce', 'both']))

    def forward(
        self, query: torch_geometric.data.Batch,
        meta_info: dict[str, torch.Tensor | list[torch.Tensor]]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # get info
        x = query.x  #type: ignore
        edge_attr = query.edge_attr  #type: ignore
        edge_index = query.edge_index  #type: ignore
        batch = query.batch  #type: ignore
        pos = query.pos  #type: ignore
        position_of_x = pos[edge_index[0]]
        
        # apply the embedding MLPs
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        
        # apply the GNN layers
        for (conv, batch_norm) in zip(self.convs, self.batch_norms):
            # run network
            if isinstance(conv, E_GCL):
                x, position_of_x = conv(x, edge_index, position_of_x,
                                        edge_attr)
            elif isinstance(conv, torch_geometric.nn.conv.DynamicEdgeConv):
                x = conv(x, batch)
            else:
                x = conv(x, edge_index, edge_attr)
            x = torch.nn.functional.relu(batch_norm(x))

        # apply the final MLP
        x = self.mlp(x)

        # apply the semantic embedding
        if self.semantic_emb is not None:
            x = self.semantic_output_mlp(
                torch.cat(
                    (
                        x,
                        self.semantic_emb(
                            torch.cat(
                                meta_info['label_probabilities']  #type: ignore
                            ).to(x.device))),
                    dim=1))

        # pool before the final MLP
        x = self.global_pool(x, batch)  # type: ignore

        # return the embeddings
        if self.TNN is not None:
            # roll through the possible pairs
            index_pairs = torch.combinations(torch.arange(x.size(0)),
                                             with_replacement=False).T
            # forward pass through the network
            scores = self.TNN(x[index_pairs[0]].unsqueeze(2),
                              x[index_pairs[1]].unsqueeze(2))
            return x, scores
        return x, None

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch_geometric.nn.PNAConv):
                m.reset_parameters()
            elif isinstance(m, torch_geometric.nn.GATv2Conv):
                m.reset_parameters()
            elif isinstance(m, torch_geometric.nn.BatchNorm):
                m.reset_parameters()
            elif isinstance(m, E_GCL):
                m.reset_parameters()
            elif isinstance(m, torch_geometric.nn.conv.DynamicEdgeConv):
                m.reset_parameters()
