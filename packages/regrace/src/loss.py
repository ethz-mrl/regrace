import torch
from pytorch_metric_learning import distances, losses, reducers


class Miner():
    device: torch.device
    distance: distances.LpDistance
    max_positive_pair_dist: float = 0
    max_negative_pair_dist: float = 0
    min_positive_pair_dist: float = float('inf')
    min_negative_pair_dist: float = float('inf')
    avg_positive_pair_dist: float = 0
    avg_negative_pair_dist: float = 0

    # By Joshua Knights, CSIRO

    def __init__(self, device: torch.device, distance: distances.LpDistance):
        self.device = device
        self.distance = distance

    def __call__(
        self, query_embeddings: torch.Tensor, positive_masks: torch.Tensor,
        negative_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # init
        assert query_embeddings.dim(
        ) == 2  # check if the tensor has 2 dimensions (batch_size, embedding_size  )
        query_embeddings_detach = query_embeddings.detach()

        # mine
        with torch.no_grad():
            hard_triplets = self.mine(query_embeddings_detach, positive_masks,
                                      negative_masks)
        return hard_triplets

    @staticmethod
    def get_min_per_row(
            matrix: torch.Tensor,
            mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # get the minimum value per row
        valid_rows = torch.any(mask, dim=1)
        masked_matrix = matrix.clone()
        masked_matrix[~mask] = float('inf')
        return torch.min(masked_matrix, dim=1), valid_rows

    @staticmethod
    def get_max_per_row(
            matrix: torch.Tensor,
            mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # get the maximum value per row
        valid_rows = torch.any(mask, dim=1)
        masked_matrix = matrix.clone()
        masked_matrix[~mask] = 0
        return torch.max(masked_matrix, dim=1), valid_rows

    def mine(
        self, query_embeddings: torch.Tensor, positive_masks: torch.Tensor,
        negative_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # compute the distance matrix
        dist_matrix = self.distance(query_embeddings)

        # get index of valid rows and best positives and negatives
        (hardest_negatives_distances,
         hardest_negatives_index), valid_index_1 = self.get_min_per_row(
             dist_matrix, negative_masks)
        (hardest_positives_distances,
         hardest_positives_index), valid_index_2 = self.get_max_per_row(
             dist_matrix, positive_masks)
        valid_index = torch.where(
            torch.logical_and(valid_index_1, valid_index_2))

        # filter out invalid rows
        query_index = torch.arange(dist_matrix.size(0),
                                   device=self.device)[valid_index]
        positive_index = hardest_positives_index[valid_index]
        negative_index = hardest_negatives_index[valid_index]

        # collect stats
        self.max_positive_pair_dist = torch.max(
            hardest_positives_distances[valid_index]).item()
        self.max_negative_pair_dist = torch.max(
            hardest_negatives_distances[valid_index]).item()
        self.min_positive_pair_dist = torch.min(
            hardest_positives_distances[valid_index]).item()
        self.min_negative_pair_dist = torch.min(
            hardest_negatives_distances[valid_index]).item()
        self.avg_positive_pair_dist = torch.mean(
            hardest_positives_distances[valid_index]).item()
        self.avg_negative_pair_dist = torch.mean(
            hardest_negatives_distances[valid_index]).item()
        return query_index, positive_index, negative_index


class TripletLossWithMiner():
    device: torch.device
    distance: distances.LpDistance
    miner: Miner
    loss_fn: losses.TripletMarginLoss

    def __init__(self, device: torch.device, margin: float, normalize: bool):
        self.device = device
        self.distance = distances.LpDistance(normalize_embeddings=normalize,
                                             collect_stats=True)
        self.miner = Miner(self.device, self.distance)
        reducer_fn = reducers.AvgNonZeroReducer(collect_stats=True)
        self.loss_fn = losses.TripletMarginLoss(margin,
                                                swap=True,
                                                distance=self.distance,
                                                reducer=reducer_fn,
                                                collect_stats=True)

    @staticmethod
    def nonzero_loss_average(
            x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.nonzero(x)
        nonzero_count = len(mask)
        return (x.sum() / nonzero_count), mask

    @staticmethod
    def nonzero_distance_average(x: torch.Tensor, y: torch.Tensor,
                                 index: torch.Tensor) -> torch.Tensor:
        return torch.dist(x[index], y[index], p=2).sum() / len(index)

    def __call__(
        self, query_embeddings: torch.Tensor, meta_info: dict[str,
                                                              torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # From Joshua Knights, CSIRO
        hard_triplets = self.miner(query_embeddings,
                                   meta_info['positive_mask'].to(self.device),
                                   meta_info['negative_mask'].to(self.device))
        positive_embeddings = query_embeddings[hard_triplets[1]]
        loss = self.loss_fn(query_embeddings,
                            indices_tuple=hard_triplets,
                            ref_emb=positive_embeddings)

        # collect stats
        stats = {
            'Loss': loss.item(),
            'Embedding Norm': self.loss_fn.distance.final_avg_query_norm,
            'Non Zero Triplets': self.loss_fn.reducer.num_past_filter,
            'Positive Pair Distance': self.miner.avg_positive_pair_dist,
            'Negative Pair Distance': self.miner.avg_negative_pair_dist,
            'Max Positive Pair Distance': self.miner.max_positive_pair_dist,
            'Max Negative Pair Distance': self.miner.max_negative_pair_dist,
            'Min Positive Pair Distance': self.miner.min_positive_pair_dist,
            'Min Negative Pair Distance': self.miner.min_negative_pair_dist
        }
        return loss, stats


class BCELoss():
    device: torch.device
    loss_fn: torch.nn.BCELoss

    def __init__(self, device: torch.device):
        self.device = device
        self.loss_fn = torch.nn.BCELoss(reduction='none')

    def __call__(
        self,
        scores: torch.Tensor,
        meta_info: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # roll through the possible pairs
        index = torch.combinations(torch.arange(
            meta_info["positive_mask"].size(0)),
                                   with_replacement=False).T

        # create gt labels
        labels = meta_info["positive_mask"][index[0], index[1]].to(self.device)

        # calculate loss
        loss = self.loss_fn(scores.squeeze(), labels.float())

        # count non zero elements
        valid_mask = ~torch.isnan(loss)
        valid_count = len(valid_mask)
        valid_loss = loss[valid_mask].mean()

        # collect stats
        stats = {
            'BCE Loss':
            valid_loss.item(),
            'Valid Pairs':
            valid_count,
            'Correct Guess Percentage':
            (scores.squeeze().round() == labels).float().mean()
        }
        return valid_loss, stats
