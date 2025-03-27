from typing import Any, Optional, Sequence, Union

import torch
import torch.utils.data
from scipy.spatial import distance
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter

from .data import Data
from .dataset import Dataset


class Collater:

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        max_dist_2_positive: float,
        min_dist_2_negative: float,
        normalization_constant: float,
        augument_data: bool,
        follow_batch: Optional[list[str]] = None,
        exclude_keys: Optional[list[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.max_dist_2_positive = max_dist_2_positive
        self.min_dist_2_negative = min_dist_2_negative
        self.normalization_constant = normalization_constant
        self.augument_data = augument_data

    def __call__(
        self, data_batch: list[dict[str, Data]]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | list[torch.Tensor]]]:
        # initialize the query batch
        points: list[torch.Tensor] = []
        cluster_centers: list[torch.Tensor] = []
        label_probabilities: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        batch_split_index = torch.ones(len(data_batch) * 2) * -1

        for i, data in enumerate(data_batch):
            # get the query data
            query_points, query_cluster_centers, query_label_probabilities = data[
                "query"].to_collate_format(self.normalization_constant,
                                           self.augument_data)
            points.extend(query_points)
            cluster_centers.append(query_cluster_centers)
            label_probabilities.append(query_label_probabilities)
            batch_split_index[2 * i] = len(query_points)
            # get the positive data
            positive_points, positive_cluster_centers, positive_label_probabilities = data[
                "positive"].to_collate_format(self.normalization_constant,
                                              self.augument_data)
            points.extend(positive_points)
            cluster_centers.append(positive_cluster_centers)
            label_probabilities.append(positive_label_probabilities)
            batch_split_index[2 * i + 1] = len(positive_points)
            # get the positions
            positions.extend([
                torch.from_numpy(data["query"].position.reshape(-1)[:2]),
                torch.from_numpy(data["positive"].position.reshape(-1)[:2])
            ])
            
        # sanity check
        assert torch.all(batch_split_index >= 0)
        
        # create batch
        batch = torch.utils.data.default_collate(points)
        
        # create masks
        dist_matrix = torch.tensor(distance.cdist(positions, positions))
        positive_mask = (dist_matrix
                         < self.max_dist_2_positive).fill_diagonal_(
                             0)  # avoid self-matching
        negative_mask = (dist_matrix > self.min_dist_2_negative)
        
        # create meta info dict
        meta_info = {
            "positive_mask": positive_mask,
            "negative_mask": negative_mask,
            "distances": dist_matrix,
            "batch_split_index": batch_split_index.int(),
            "cluster_centers": cluster_centers,
            "label_probabilities": label_probabilities,
        }

        return batch, meta_info

    def collate_fn(
        self, batch: list[Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | list[torch.Tensor]]]:
        return self(batch)


class DataLoader(torch.utils.data.DataLoader):

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        max_dist_2_positive: float,
        min_dist_2_negative: float,
        normalization_constant: float,
        augument_data: bool,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[list[str]] = None,
        exclude_keys: Optional[list[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.collator = Collater(dataset, max_dist_2_positive,
                                 min_dist_2_negative, normalization_constant,
                                 augument_data, follow_batch, exclude_keys)

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collator.collate_fn,
            **kwargs,
        )
