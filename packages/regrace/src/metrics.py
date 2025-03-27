import click
import joblib
import pandas as pd
import torch
import wandb
import wandb.plot
from tabulate import tabulate
from tqdm import tqdm

from .config import YAMLConfig
from .dataset import Dataset
from .registration import compute_consistency


def pr_score(
    embeddings: torch.Tensor,
    dataset: Dataset,
    node_features: list[torch.Tensor],
    node_positions: list[torch.Tensor],
    timestamps: torch.Tensor,
    positions: torch.Tensor,
    config: YAMLConfig,
    top_k: int = 20,
) -> None:
    # check input
    assert embeddings.dim(
    ) == 2, "Embeddings should be a 2D tensor"  # should be shape (N, embedding_dim)\
    assert timestamps.dim() == 1 and positions.dim(
    ) == 2, "Timestamps should be a 1D tensor and positions should be a 2D tensor"
    positions = positions.clone()[:, :2]
    timestamps = timestamps.clone().reshape(-1, 1).float()
    assert positions.shape[
        1] == 2, "Positions should have shape (N, 2)"  # only XY positions are required
    assert timestamps.shape[0] == embeddings.shape[0] and timestamps.shape[
        0] == positions.shape[
            0], "Timestamps and positions should have the same length as embeddings"
    assert timestamps.shape[0] == len(
        node_features
    ), "Node features should have the same length as embeddings"
    assert len(node_features) == len(
        node_positions
    ), "Node positions should have the same length as embeddings"
    start_time = timestamps[0]

    # verbose
    click.echo(
        click.style(
            f">> Evaluating metrics for top-{top_k} closest submaps",
            fg='blue',
            bold=True))
    click.echo(
        click.style(
            f">> Max distance to consider a relocation as correct: {config.max_dist_2_true_positive}",
            fg='blue',
            bold=True))

    # get tresholds
    embedding_tresholds_to_evaluate = torch.linspace(
            config.min_consistency_treshold, config.max_consistency_treshold,
            config.num_consistency_tresholds).float().tolist()

    # init DF with columns (tresh, TP, FP, FN, TN) for each treshold
    df = pd.DataFrame({
        "tresh":
        pd.Series(embedding_tresholds_to_evaluate),
        "TP":
        pd.Series(
            torch.zeros(len(embedding_tresholds_to_evaluate)).int().tolist()),
        "FP":
        pd.Series(
            torch.zeros(len(embedding_tresholds_to_evaluate)).int().tolist()),
        "FN":
        pd.Series(
            torch.zeros(len(embedding_tresholds_to_evaluate)).int().tolist()),
        "TN":
        pd.Series(
            torch.zeros(len(embedding_tresholds_to_evaluate)).int().tolist()),
        "Precision":
        pd.Series(
            torch.zeros(
                len(embedding_tresholds_to_evaluate)).float().tolist()),
        "Recall":
        pd.Series(
            torch.zeros(
                len(embedding_tresholds_to_evaluate)).float().tolist()),
        "F1":
        pd.Series(
            torch.zeros(
                len(embedding_tresholds_to_evaluate)).float().tolist()),
    })

    # create timestamp difference matrix
    positions_diff = torch.cdist(positions, positions)
    embeddings_diff = torch.cdist(embeddings, embeddings)
    num_total_reloc = 0

    # store correct relocations
    correct_relocations: dict[float, list[int]] = {}
    all_revisits: set[int] = set()

    # sanity check
    assert config.time_window > 0, "Time window should be greater than 0"

    # evaluate each item
    for eval_idx in tqdm(range(embeddings.shape[0]),
                         colour='CYAN',
                         desc='Collecting metrics',
                         dynamic_ncols=True):

        # sanity check
        assert timestamps[
            eval_idx] - start_time >= 0, "Timestamps should be sorted"

        # get the seen embeddings
        time_diff = timestamps[
            eval_idx] - timestamps  # negative for future timestamps
        seen_mask = (time_diff >= config.time_window).bool().view(-1)

        # if no seen embeddings, skip
        if torch.sum(seen_mask) == 0: continue
        mask_idx = torch.where(seen_mask)[0]

        # get the distance to the seen embeddings
        dist_to_seen_positions = positions_diff[eval_idx][seen_mask]
        dist_to_seen_embeddings = embeddings_diff[eval_idx][seen_mask]

        # compute if there are revisits
        there_are_revisits = False
        if torch.any(
                dist_to_seen_positions <= config.max_dist_2_true_positive):
            there_are_revisits = True
            all_revisits.update([
                int(mask_idx[i].item())
                for i in torch.where(dist_to_seen_positions <=
                                     config.max_dist_2_true_positive)[0].int()
            ])
            num_total_reloc += 1

        # get the best top-k candidate in the seen embeddings
        topk_idx = torch.topk(dist_to_seen_embeddings,
                              k=min(top_k, len(mask_idx)),
                              largest=False)[1]

        # rerank
        if topk_idx.shape[0] > 1:
            consistency = joblib.Parallel(
                n_jobs=config.num_workers,
                return_as='list')(joblib.delayed(compute_consistency)(
                    node_features[eval_idx],
                    node_positions[eval_idx],
                    node_features[mask_idx[idx]],
                    node_positions[mask_idx[idx]],
                    dataset.furthest_dist_between_points,
                    1,
                ) for idx in topk_idx)
            top1_idx = topk_idx[torch.argmax(torch.tensor(consistency))]
            topk_idx = topk_idx[torch.argsort(torch.tensor(consistency),
                                              descending=True)]
            top_consistency = torch.tensor(consistency).max()
        else:
            top1_idx = topk_idx[0]
            consistency = compute_consistency(
                node_features[eval_idx], node_positions[eval_idx],
                node_features[mask_idx[top1_idx]],
                node_positions[mask_idx[top1_idx]],
                dataset.furthest_dist_between_points, 1)
            top_consistency = torch.tensor(consistency).max()
        top1_position_dist, top1_embedding_dist = dist_to_seen_positions[
            top1_idx], dist_to_seen_embeddings[top1_idx]

        # evaluate the top-k (like Logg3DNet)
        for eval_tresh_row in range(len(embedding_tresholds_to_evaluate)):
            row = df.loc[eval_tresh_row].to_numpy(
            )  # pandas does not support += operator
            eval_tresh = row[0]
            # get positive predictions
            is_positive = (top_consistency >= eval_tresh)
            # evaluate the prediction
            if is_positive:  # get positive predictions
                if top1_position_dist <= config.max_dist_2_true_positive:
                    row[1] += 1  # True Positive
                    # store correct relocations
                    correct_relocations[eval_tresh] = correct_relocations.get(
                        eval_tresh,
                        []) + [int(mask_idx[top1_idx].item()), eval_idx]
                elif top1_position_dist > config.min_dist2negative:
                    row[2] += 1  # False Positive
            else:  # get negative predictions
                if there_are_revisits:
                    row[3] += 1  # False Negative
                else:
                    row[4] += 1  # True Negative
            # update the row
            df.loc[eval_tresh_row] = row

    # if no relocs, raise error
    if num_total_reloc == 0:
        raise ValueError("No relocations found in evaluation the dataset")

    # F1Max = F1 score at top-1
    f1_max_score, f1_max_tresh = 0, float('inf')
    for eval_tresh_row in range(len(embedding_tresholds_to_evaluate)):
        row = df.loc[eval_tresh_row].to_numpy()
        eval_tresh = row[0]
        # check there are TP
        if row[1] > 0:
            # compute metrics
            precision = row[1] / (row[1] + row[2])  # TP / (TP + FP)
            recall = row[1] / (row[1] + row[3])  # TP / (TP + FN)
            f1_score = 2 * (precision * recall) / (precision + recall)
             # update F1Max
            if f1_score >= f1_max_score:
                f1_max_score = f1_score
                f1_max_tresh = eval_tresh
            # update the row
            row[5] = precision
            row[6] = recall
            row[7] = f1_score
            df.loc[eval_tresh_row] = row

    # report metrics
    click.echo(click.style(f"{'='*15} Metrics {'='*15}", fg='green',
                           bold=True))
    click.echo(f">> F1-Max: {f1_max_score} at tresh: {f1_max_tresh}")

    # echo pandas dataframe
    click.echo(tabulate(
        df,  # type: ignore
        headers='keys',
        tablefmt='psql'))

    # log to wandb
    if config.wandb_logging:
        data_table = wandb.Table(data=df)
        metrics_dict = {
            "precision-recall":
            wandb.plot.line(data_table,
                            "Recall",
                            "Precision",
                            title="Precision-Recall Curve"),
            "F1-Max":
            f1_max_score,
            "metrics":
            data_table
        }
        wandb.log(metrics_dict)
