import os
import random
from pathlib import Path

import click
import numpy as np
import torch
import torch.utils
import torch.utils.data

os.environ["WANDB_SILENT"] = "true"

import wandb
import wandb.plot

from ..src.config import YAMLConfig
from ..src.dataloader import DataLoader
from ..src.gnn import GNN
from ..src.metrics import pr_score
from ..src.riconv2_cls import RIConvClassification
from ..src.trainer import Trainer
from .generate_data import load_triplets


def create_wandb_instance(config: YAMLConfig) -> None:

    # rename the conv type if it is dynedgeconv
    if config.conv_type == "dynedgeconv":
        name = "edge-conv"
    else:
        name = config.conv_type

    # set entity and project name
    project = 'REGRACE'
    raise NotImplementedError(
        "Change the entity to your own entity, then remove this line.")
    entity = 'YOUR WANDB ENTITY HERE'

    # load the wandb instance if it exists
    if len(config.wandb_id) > 0:
        wandb.init(project=project,
                   id=config.wandb_id,
                   resume=True,
                   entity=entity)
    # create the config otherwise
    wandb.init(
        project=project,
        config=config.model_dump(),
        group=name,
        tags=
        f"{config.conv_type}_{config.n_conv_layers}L_{config.k_nearest_neighbors}KNN_{int(config.use_semantics_in_node_features)}semnode_{config.debug_train_on_test_set}traindata_{config.n_epochs}epoch_{''.join([str(m) for m in config.decay_epochs])}decay_{config.pooling}_{config.loss_type}_{int(config.initialize_weigths_xavier)}xavier_{config.loss_margin}margin_{int(config.use_semantics_in_graph_features)}semgraph_{int(config.use_angles_in_edge_features)}anglesinedges"
        .split("_"),
        name=config.date_time.strftime('%Y-%m-%d_%H-%M-%S'))


def seed_worker(worker_id):
    # This function is used to set the seed for each worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(config: YAMLConfig, config_filename: Path) -> None:
    '''
    This function trains the model
    
    Args:
    config: config object given by the user
    config_filename: the path to the config file that will be updated with the new checkpoint paths
    '''

    # create a wandb instance
    if config.wandb_logging:
        create_wandb_instance(config)

    # load the dataset and max cardinality
    if config.debug_train_on_test_set:
        train_dataset = load_triplets(config, "test")
        click.echo(
            click.style(
                "WARNING: Training on test set. This is only for debugging purposes.",
                fg='red',
                bold=True))
    else:
        train_dataset = load_triplets(config)
    test_dataset = load_triplets(config, "test")
    click.echo(click.style("=" * 50, fg='green', bold=True))

    # ensure reproducibility
    g_train = torch.Generator()
    g_train.manual_seed(0)
    train_loader = DataLoader(
        train_dataset,
        max_dist_2_positive=config.max_dist2positive,
        min_dist_2_negative=config.min_dist2negative,
        normalization_constant=train_dataset.furthest_dist_between_points,
        batch_size=config.batch_size,
        shuffle=True,
        generator=g_train,
        augument_data=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        num_workers=config.num_workers)

    # ensure reproducibility
    g_val = torch.Generator()
    g_val.manual_seed(0)
    val_loader = DataLoader(
        test_dataset,
        max_dist_2_positive=config.max_dist2positive,
        min_dist_2_negative=config.min_dist2negative,
        normalization_constant=train_dataset.furthest_dist_between_points,
        batch_size=config.batch_size,
        shuffle=True,
        generator=g_val,
        worker_init_fn=seed_worker,
        augument_data=False,
        num_workers=config.num_workers)
    click.echo(
        click.style(f"Train dataloader with size: {len(train_dataset)}",
                    fg='blue',
                    bold=True))
    click.echo(
        click.style(f"Validation dataloader with size: {len(test_dataset)}",
                    fg='blue',
                    bold=True))

    # set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    click.echo(click.style(f"Device: {device}", fg='blue', bold=True))
    click.echo(click.style("=" * 50, fg='blue', bold=True))

    # pre process histogram
    histogram = train_dataset.histogram[:config.k_nearest_neighbors + 1]
    histogram[-1] = train_dataset.histogram[(config.k_nearest_neighbors +
                                             1):].sum()

    # create the model
    gnn = GNN.from_config(config, histogram).to(device)
    assert config.n_points_to_sample % 512 == 0, "Number of points to sample must be multiple of 512"
    embedding_net = RIConvClassification(int(config.n_points_to_sample /
                                             512)).to(device)

    # create the trainer
    trainer = Trainer(config, gnn, embedding_net, train_loader, val_loader,
                      device, config_filename)

    # train the model
    trainer.train()

    # change config model path
    config.checkpoint_path = Path(f"{trainer.save_path}/checkpoint-final.pt")


def test(config: YAMLConfig) -> None:

    # load the dataset and max cardinality
    train_dataset = load_triplets(config)
    test_dataset = load_triplets(config, "test")
    click.echo(click.style("=" * 50, fg='green', bold=True))

    # create the data loader
    val_loader = DataLoader(
        test_dataset,
        normalization_constant=train_dataset.furthest_dist_between_points,
        max_dist_2_positive=config.max_dist2positive,
        min_dist_2_negative=config.min_dist2negative,
        batch_size=1,
        shuffle=False,
        augument_data=False,
        num_workers=config.num_workers)
    click.echo(
        click.style(f"Validation dataloader with size: {len(test_dataset)}",
                    fg='blue',
                    bold=True))

    # set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    click.echo(click.style(f"Device: {device}", fg='blue', bold=True))
    click.echo(click.style("=" * 50, fg='blue', bold=True))

    # create the model
    gnn = GNN.from_config(config, train_dataset.histogram).to(device)
    assert config.n_points_to_sample % 512 == 0, "Number of points to sample must be multiple of 512"
    embedding_net = RIConvClassification(int(config.n_points_to_sample /
                                             512)).to(device)

    # create the trainer
    trainer = Trainer(config,
                      gnn,
                      embedding_net,
                      val_loader,
                      val_loader,
                      device,
                      config_filename=None)

    # create the instance of the run
    if config.wandb_logging:
        if wandb.run is None:
            create_wandb_instance(config)

    # collect the distance matrix
    if config.loss_type == "triplet" or config.loss_type == "both":
        embeddings, node_features, node_positions = trainer.test()
        torch.cuda.empty_cache()
        # get the metrics for PR
        pr_score(embeddings=embeddings,
                 dataset=test_dataset,
                 node_features=node_features,
                 node_positions=node_positions,
                 timestamps=test_dataset.timestamps,
                 positions=test_dataset.positions,
                 config=config,
                 top_k=20)
    else:
        raise ValueError("We only support triplet loss for now :(")
