""" 
Code by https://github.com/kxhit/SG_PR
"""
import torch


class TensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """

    def __init__(self, attention_size: int, n_tensor_neurons: int,
                 bottleneck_size: int):
        """
        :param args: Arguments object.
        """
        super(TensorNetworkModule, self).__init__()
        self.attention_size = attention_size
        self.n_tensor_neurons = n_tensor_neurons
        self.bottleneck_size = bottleneck_size
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(self.attention_size, self.attention_size,
                         self.n_tensor_neurons))
        self.weight_matrix_block = torch.nn.Parameter(
            torch.Tensor(self.n_tensor_neurons, 2 * self.attention_size))
        self.bias = torch.nn.Parameter(torch.Tensor(self.n_tensor_neurons, 1))
        self.fully_connected_first = torch.nn.Linear(self.n_tensor_neurons,
                                                     self.bottleneck_size)
        self.scoring_layer = torch.nn.Linear(self.bottleneck_size, 1)

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)
        torch.nn.init.xavier_uniform_(self.fully_connected_first.weight)
        torch.nn.init.xavier_uniform_(self.scoring_layer.weight)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.    bxfx1
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        batch_size = embedding_1.shape[0]  #gxfx1
        scoring = torch.matmul(
            embedding_1.permute(0, 2, 1),  #gx1xf
            self.weight_matrix.view(self.attention_size, -1)).view(
                batch_size,
                self.attention_size,
                self.n_tensor_neurons,
            )
        scoring = torch.matmul(scoring.permute(0, 2, 1), embedding_2)  # bxfx1
        combined_representation = torch.cat((embedding_1, embedding_2),
                                            dim=1)  # bx2fx1
        block_scoring = torch.matmul(
            self.weight_matrix_block,
            combined_representation)  # bxtensor_neuronsx1
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        scores = scores.permute(0, 2, 1)  # bx1xf
        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores)).reshape(-1)
        return score
