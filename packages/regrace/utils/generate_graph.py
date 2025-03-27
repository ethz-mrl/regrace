import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn.pool import knn_graph


def create_graph(
    node_features: torch.Tensor,
    nodes_positions: torch.Tensor,
    label_probabilities: torch.Tensor,
    knn: int,
    with_angles: bool,
    with_semantics: bool,
) -> Data:
    # sanity check
    assert node_features.shape[0] == nodes_positions.shape[0]
    
    # get edge index
    edge_index = knn_graph(x=nodes_positions.float(),
                           k=knn,
                           batch=None,
                           loop=False,
                           flow='source_to_target',
                           cosine=False)
    
    # get edge features
    dist = torch.cdist(nodes_positions, nodes_positions)
    if with_angles:
        angleXY = (torch.atan2(
            nodes_positions[None, :, 1] - nodes_positions[None, :, 1].T,
            nodes_positions[None, :, 0] - nodes_positions[None, :, 0].T) +
                   2 * torch.pi) % (2 * torch.pi) / (2 * torch.pi)
        angleXZ = (torch.atan2(
            nodes_positions[None, :, 2] - nodes_positions[None, :, 2].T,
            nodes_positions[None, :, 0] - nodes_positions[None, :, 0].T) +
                   2 * torch.pi) % (2 * torch.pi) / (2 * torch.pi)
        edge_features = torch.stack([
            dist[edge_index[0], edge_index[1]],
            angleXY[edge_index[0], edge_index[1]], angleXZ[edge_index[0],
                                                           edge_index[1]]
        ],
                                    dim=1)
    else:
        edge_features = torch.stack([dist[edge_index[0], edge_index[1]]],
                                    dim=1)
        
    # get node features
    if with_semantics:
        node_features = torch.cat((node_features, label_probabilities), dim=1)
    else:
        node_features = node_features
        
    # sanity check
    if edge_index.shape[1] == 0:
        raise ValueError("No edges were created")
    
    # create the data object
    return Data(
        x=node_features,
        edge_index=edge_index.long(),
        edge_attr=edge_features.reshape(edge_index.shape[1], -1).float(),
        pos=nodes_positions.float(),
    )


def create_graph_batch(
    node_features: torch.Tensor,
    nodes_positions: list[torch.Tensor],
    label_probabilities: list[torch.Tensor],
    batch_split_index_list: list[int],
    knn: int,
    with_angles: bool,
    with_semantics: bool,
) -> Batch:
    # split the node features
    node_features_per_graph = torch.split(node_features,
                                          batch_split_index_list)
    
    # iterate over the node features
    graphs = [
        create_graph(node_features, nodes_positions, label_probability, knn,
                     with_angles, with_semantics)
        for node_features, nodes_positions, label_probability in zip(
            node_features_per_graph, nodes_positions, label_probabilities)
    ]
    
    # create the batch
    return Batch.from_data_list(graphs)  # type: ignore
