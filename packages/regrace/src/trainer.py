import itertools
from pathlib import Path
from typing import Optional

import click
import torch
import torch.utils.data
import wandb
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from ..utils.generate_graph import create_graph_batch
from .config import YAMLConfig
from .dataloader import DataLoader
from .gnn import GNN
from .loss import BCELoss, TripletLossWithMiner
from .riconv2_cls import RIConvClassification


class Trainer():

    config: YAMLConfig
    config_filename: Optional[Path]
    gnn: GNN
    embedding_net: RIConvClassification
    train_dataloader: DataLoader  # type: ignore
    validation_dataloader: DataLoader  # type: ignore
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.MultiStepLR
    loss_functions: list[TripletLossWithMiner | BCELoss]
    device: torch.device
    step: int = 0
    logging: bool

    def __init__(
        self,
        config: YAMLConfig,
        gnn: GNN,
        embedding_net: RIConvClassification,
        train_dataloader: DataLoader,  # type: ignore
        val_dataloader: DataLoader,  # type: ignore
        device: torch.device,
        config_filename: Optional[Path] = None,
    ) -> None:

        # save the parameters
        self.config = config
        self.config_filename = config_filename
        self.gnn = gnn
        self.embedding_net = embedding_net
        self.validation_dataloader = val_dataloader
        self.train_dataloader = train_dataloader

        # create the optimizer only on the active parameters
        if self.config.freeze_riconv:
            self.optimizer = torch.optim.Adam(self.gnn.parameters(),
                                              lr=config.learning_rate)
        else:
            self.optimizer = torch.optim.Adam(itertools.chain(
                self.gnn.parameters(), self.embedding_net.parameters()),
                                              lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.decay_epochs,
            gamma=config.lr_decay_factor)
        # create the loss functions
        self.loss_functions = []
        if config.loss_type in ['triplet', 'both']:
            self.loss_functions.append(
                TripletLossWithMiner(device, config.loss_margin,
                                     config.normalize_embeddings))
        if config.loss_type in ['bce', 'both']:
            self.loss_functions.append(BCELoss(device))
        self.save_freq = 10
        self.save_path = f"./data/checkpoints/{config.date_time}"
        self.device = device
        self.logging = config.wandb_logging

        # check/create ./data/checkpoints/{date_time}
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

        # init weights
        if config.initialize_weigths_xavier:
            self.embedding_net.init_weights()
            self.gnn.init_weights()

        # freeze the riconv
        if self.config.freeze_riconv:
            self.embedding_net.freeze()

        # load the pretrained weights
        if self.config.init_from_checkpoint:
            self.load_weights()

    def load_weights(self):
        assert Path(self.config.checkpoint_path).exists(
        ), "Checkpoint file does not exist"
        # load the model
        try:
            state_dict = torch.load(self.config.checkpoint_path)
            self.gnn.load_state_dict(state_dict['gnn_model_state_dict'])
            self.embedding_net.load_state_dict(
                state_dict['embedding_model_state_dict'])
            if (state_dict['optimizer_state_dict']
                    is not None) and not self.config.freeze_riconv:
                # if the riconv is frozen, the optimizer is not loaded
                self.optimizer.load_state_dict(
                    state_dict['optimizer_state_dict'])
            if state_dict['scheduler_state_dict'] is not None:
                self.scheduler.load_state_dict(
                    state_dict['scheduler_state_dict'])
            self.step = state_dict['step']
            self.epoch = state_dict['epoch']
            click.echo(
                click.style(
                    f"Loaded weights from {self.config.checkpoint_path}",
                    fg='green'))
        except Exception as e:
            click.echo(click.style(f"Error: {e}", fg='red'))
            raise ValueError(
                f"Error loading weights from {self.config.checkpoint_path}")

    def save_weights(self, final: bool = False):
        # save the model
        torch.save(
            {
                'step': self.step,
                'gnn_model_state_dict': self.gnn.state_dict(),
                'embedding_model_state_dict': self.embedding_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.epoch,
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, f"{self.save_path}/checkpoint-final.pt"
            if final else f"{self.save_path}/checkpoint-{self.epoch}.pt")
        click.echo(
            click.style(f"Saved weights to {self.save_path}", fg='green'))

        # change the config object
        config = self.config.model_copy()
        config.checkpoint_path = Path(
            f"{self.save_path}/checkpoint-{self.epoch}.pt")
        config.init_from_checkpoint = True
        if final:
            config.train = False
            config.test = True

        # dump the config
        assert self.config_filename is not None, \
            "File to dump updated config is not set. Please set it using the config_filename parameter for the Trainer class."
        config.dump_to_yaml(self.config_filename)

    def before_training(self) -> None:
        if not self.config.init_from_checkpoint:
            self.epoch = 0
            self.step = 0

    def before_epoch(self) -> None:
        self.epoch += 1

    def validate_epoch(self) -> float:
        # total loss
        val_loss = []
        for batch, meta_info in tqdm(
                self.validation_dataloader,
                desc="Validating...",
                colour='CYAN',
                dynamic_ncols=True,
        ):
            # get the loss of the batch
            loss = self.single_batch(batch, meta_info, validation=True)
            if loss != -1:
                val_loss.append(loss)

        # return the average loss
        return sum(val_loss) / len(val_loss) if len(val_loss) > 0 else -1

    def train_epoch(self) -> None:
        for batch, meta_info in tqdm(
                self.train_dataloader,
                desc=f"Training Epoch {self.epoch}",
                colour='MAGENTA',
                dynamic_ncols=True,
        ):
            # train the model on the batch
            _ = self.single_batch(batch, meta_info)
            self.step += 1

    def after_epoch(self):
        # update the learning rate
        self.scheduler.step()
        # save periodically
        if self.epoch % self.save_freq == 0:
            # validate the model
            val_loss = self.validate_epoch()
            assert val_loss != -1, 'The batch must have positive and negative examples'
            # save the model
            self.save_weights()

    def single_batch(
        self,
        batch: torch.Tensor,
        meta_info: dict[str, torch.Tensor],
        validation: bool = False,
    ) -> float:
        # make sure that the batch has positive and negative examples
        n_sum_pos = torch.sum(meta_info['positive_mask'])
        n_sum_neg = torch.sum(meta_info['negative_mask'])
        if n_sum_pos == 0 or n_sum_neg == 0:
            click.echo(
                click.style(
                    f"Invalid batch with no positive examples {n_sum_pos} or negative examples {n_sum_neg}",
                    fg='red'))
            return -1

        # initialize the total loss and total distances
        self.gnn.eval() if validation else self.gnn.train()
        self.embedding_net.eval() if validation else self.embedding_net.train()

        # get embeddings
        node_features = self.embedding_net(batch.to(self.device))

        # get graph batch
        query = create_graph_batch(
            node_features=node_features,
            nodes_positions=meta_info['cluster_centers'],  # type: ignore
            label_probabilities=meta_info[
                'label_probabilities'],  # type: ignore
            batch_split_index_list=meta_info['batch_split_index'].tolist(),
            knn=self.config.k_nearest_neighbors,
            with_angles=self.config.use_angles_in_edge_features,
            with_semantics=self.config.use_semantics_in_node_features,
        )

        # forward pass
        query_embeddings, scores = self.gnn(query, meta_info)

        # compute the loss
        if self.config.loss_type == 'both':
            triplet_loss, tmp_stats_triplets = self.loss_functions[0](
                query_embeddings, meta_info)
            bce_loss, tmp_stats_bce = self.loss_functions[1](scores, meta_info)
            loss = triplet_loss + bce_loss
            tmp_stats = {
                **tmp_stats_triplets,
                **tmp_stats_bce, 'Total Loss': loss.item()
            }
        else:
            if self.config.loss_type == 'triplet':
                loss, tmp_stats = self.loss_functions[0](query_embeddings,
                                                         meta_info)
            else:
                loss, tmp_stats = self.loss_functions[0](scores, meta_info)

        # backpropagate
        if not validation:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # logging
        if self.logging:
            if not validation:
                wandb.log(tmp_stats, step=self.step)
            else:
                validation_dict = {}
                if self.config.loss_type == "triplet" or self.config.loss_type == "both":
                    validation_dict.update({
                        'Validation Loss':
                        tmp_stats['Loss'],
                        'Validation Positive Pair Distance':
                        tmp_stats['Positive Pair Distance'],
                        'Validation Negative Pair Distance':
                        tmp_stats['Negative Pair Distance'],
                        'Validation Non Zero Triplets':
                        tmp_stats['Non Zero Triplets'],
                        'Learning Rate':
                        self.scheduler.get_last_lr()[0]
                    })
                if self.config.loss_type == "bce" or self.config.loss_type == "both":
                    validation_dict.update({
                        'Validation BCE Loss':
                        tmp_stats['BCE Loss'],
                        'Validation Valid Pairs':
                        tmp_stats['Valid Pairs'],
                        'Validation Correct Guess Percentage':
                        tmp_stats['Correct Guess Percentage'],
                        'Learning Rate':
                        self.scheduler.get_last_lr()[0]
                    })
                wandb.log(validation_dict, step=self.step)
        return loss.item()

    def after_training(self) -> None:
        self.save_weights(final=True)

    def train(self):
        self.before_training()
        for _ in range(self.config.n_epochs):
            self.before_epoch()
            self.train_epoch()
            self.after_epoch()
        self.after_training()

    def return_embeddings(
        self
    ) -> tuple[list[torch.Tensor] | list[tuple[Data, torch.Tensor]],
               list[torch.Tensor], list[torch.Tensor]]:
        embeddings = []
        nodes_feature_list = []
        nodes_positions = []
        for batch, meta_info in tqdm(self.validation_dataloader,
                                     desc="Collecting embeddings",
                                     colour='magenta',
                                     dynamic_ncols=True):

            # get embeddings
            node_features = self.embedding_net(batch.to(self.device))
            nodes_feature_list.append(
                node_features[:meta_info['batch_split_index'][0]])
            nodes_positions.append(meta_info['cluster_centers'][0])

            # get graph batch
            query = create_graph_batch(
                node_features=node_features[:meta_info['batch_split_index']
                                            [0]],
                nodes_positions=[meta_info['cluster_centers'][0]],
                label_probabilities=[meta_info['label_probabilities'][0]
                                     ],  # type: ignore
                batch_split_index_list=[
                    meta_info['batch_split_index'].tolist()[0]
                ],
                knn=self.config.k_nearest_neighbors,
                with_angles=self.config.use_angles_in_edge_features,
                with_semantics=self.config.use_semantics_in_node_features,
            )

            # forward pass and get the embeddings for triplet loss
            meta_info['label_probabilities'] = [
                meta_info['label_probabilities'][0]
            ]
            query_embedding = self.gnn(query, meta_info)[0][0, :].reshape(
                1, -1
            )  # get the first element, __getitem__ returns a tuple (query, positive)
            assert query_embedding.shape[0] == 1, \
                "The query embedding must have a single element"
            assert query_embedding.shape[1] == self.config.output_mpl_output_layer_size, \
                "The query embedding must have the same size as the embedding size"
            embeddings.append(query_embedding.detach().cpu())  # move to cpu

        return embeddings, nodes_feature_list, nodes_positions  # type: ignore

    def test(
        self, ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:

        # load the model
        self.load_weights()

        # set the model to evaluation mode
        self.gnn.eval()
        self.embedding_net.eval()

        # get the embeddings of the validation set
        with torch.no_grad():
            embeddings, node_features, nodes_positions = self.return_embeddings(
            )
            assert isinstance(embeddings[0], torch.Tensor), \
                    "The embeddings must be a tensor"
            embeddings = torch.stack(embeddings).squeeze()  # type: ignore
            return embeddings, node_features, nodes_positions
