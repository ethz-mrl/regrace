import torch

from ..utils.generate_graph import create_graph_batch
from .config import YAMLConfig
from .gnn import GNN
from .riconv2_cls import RIConvClassification


class Model(torch.nn.Module):
    gnn: GNN
    riconv: RIConvClassification
    device: torch.device
    knn: int
    use_angles_in_edge_features: bool
    use_semantics_in_node_features: bool

    def __init__(self, config: YAMLConfig, histogram: torch.Tensor,
                 device: torch.device):
        # initialize the super class
        super(Model, self).__init__()
        # create the GNN
        self.gnn = GNN.from_config(config, histogram).to(device)
        # create the RIConv
        assert config.n_points_to_sample % 512 == 0, "Number of points to sample must be multiple of 512"
        self.riconv = RIConvClassification(int(config.n_points_to_sample /
                                               512)).to(device)
        # store parameters
        self.device = device
        self.knn = config.k_nearest_neighbors
        self.use_angles_in_edge_features = config.use_angles_in_edge_features
        self.use_semantics_in_node_features = config.use_semantics_in_node_features

    def forward(
        self,
        batch: torch.Tensor,
        meta_info: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # get the node features
        node_features = self.riconv(batch.to(self.device)).to("cpu")
        # create the graph batch
        query = create_graph_batch(
            node_features=node_features,
            nodes_positions=meta_info['cluster_centers'],  # type: ignore
            label_probabilities=meta_info[
                'label_probabilities'],  # type: ignore
            batch_split_index_list=meta_info['batch_split_index'].tolist(),
            knn=self.knn,
            with_angles=self.use_angles_in_edge_features,
            with_semantics=self.use_semantics_in_node_features,
        ).to(  # type: ignore
            self.device)
        # get the embeddings and scores
        query_embeddings, scores = self.gnn(query, meta_info)
        return query_embeddings, scores

    def get_gnn_state_dict(self):
        return self.gnn.state_dict()

    def get_riconv_state_dict(self):
        return self.riconv.state_dict()
