import torch
import torch_geometric.nn
import torch_geometric.nn.aggr


class GeneralizedMeanPooling(torch.nn.Module):
    # By Joshua Knights, CSIRO
    p: torch.nn.Parameter
    eps: float

    def __init__(self, p: float = 3, eps: float = 1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        features = x.clamp(min=self.eps).pow(self.p)
        features = torch_geometric.nn.global_mean_pool(features,
                                                       batch)  # type: ignore
        return features.pow(1. / self.p)
