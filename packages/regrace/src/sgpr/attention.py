""" 
Code by https://github.com/kxhit/SG_PR
"""
import torch
import torch_geometric.nn


class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """

    def __init__(self, size: int):
        """
        :param size: Size of the input tensor.
        """
        super(AttentionModule, self).__init__()
        self.size = size
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(self.size, self.size))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embeddings, batch, eps):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector. 
        """
        # get global graph context
        graph_mean_embedding = torch_geometric.nn.global_mean_pool(
            embeddings.clamp(min=eps), batch)  # type: ignore
        c = torch.tanh(torch.matmul(graph_mean_embedding, self.weight_matrix))

        # get graph embedding
        _, batch_node_count = torch.unique(batch,
                                           return_inverse=False,
                                           return_counts=True)
        repeated_c = c.repeat_interleave(batch_node_count, dim=0)
        sigmoid_scores = torch.sigmoid(torch.matmul(embeddings, repeated_c.T))
        e = torch_geometric.nn.global_mean_pool(
            torch.matmul(sigmoid_scores, embeddings), batch)  # type: ignore
        return e
