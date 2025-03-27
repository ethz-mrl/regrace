from datetime import datetime
from pathlib import Path

import click
import pydantic
import yaml


class YAMLConfig(pydantic.BaseModel):

    # dataset
    node_features: int
    edge_features: int
    max_dist2positive: float = pydantic.Field(..., gt=0)
    min_dist2negative: float = pydantic.Field(..., gt=0)
    preprocessing_folder: str
    n_points_to_sample: int
    n_points_in_local_neighborhood: int
    n_jobs: int

    # model
    node_mpl_hidden_layer_size: int
    node_mpl_output_layer_size: int
    edge_mpl_hidden_layer_size: int
    edge_mpl_output_layer_size: int
    gnn_output_layer_size: int
    gnn_n_towers: int
    gnn_pre_mpl_n_layers: int
    gnn_post_mpl_n_layers: int
    pna_aggr: list[str]
    pna_scaler: list[str]
    output_mpl_hidden_layer_size: int
    output_mpl_output_layer_size: int
    pooling: str
    conv_type: str
    n_conv_layers: int
    k_nearest_neighbors: int

    # training
    batch_size: int
    learning_rate: float
    n_epochs: int
    decay_epochs: list[int]
    lr_decay_factor: float
    loss_margin: float
    loss_norm_order: int
    initialize_weigths_xavier: bool
    num_workers: int
    loss_type: str
    checkpoint_path: Path = Path("")

    # data folder
    output_path: Path
    train_folders: list[Path]
    test_folders: list[Path]

    # flags
    generate_triplets: bool = False
    normalize_embeddings: bool = False
    train: bool = False
    test: bool = False
    wandb_logging: bool = False
    use_angles_in_edge_features: bool = False
    use_semantics_in_node_features: bool = False
    debug_train_on_test_set: bool = False
    use_semantics_in_graph_features: bool = False
    init_from_checkpoint: bool = False
    freeze_riconv: bool = False

    # unique identifier
    date_time: datetime

    # evaluation parameters
    min_consistency_treshold: float
    max_consistency_treshold: float
    num_consistency_tresholds: int
    time_window: float
    wandb_id: str = ""
    max_dist_2_true_positive: float = pydantic.Field(..., gt=0)

    @pydantic.field_validator('pna_aggr')
    def check_pna_aggr(cls, value):
        for aggr in value:
            if aggr not in ['mean', 'max', 'min', 'std']:
                raise ValueError(f'{aggr} is not a valid aggregation function')
        return value

    @pydantic.field_validator('pna_scaler')
    def check_pna_scaler(cls, value):
        for scaler in value:
            if scaler not in ['identity', 'amplification', 'attenuation']:
                raise ValueError(f'{scaler} is not a valid scaling function')
        return value

    @pydantic.field_validator('pooling')
    def check_pooling(cls, value):
        if value not in [
                'genmean',
                'global_mean',
                'global_add',
        ]:
            raise ValueError(
                f"Pooling {value} not supported. Please choose between 'genmean', 'global_mean', 'global_add'"
            )
        return value

    @pydantic.field_validator('loss_type')
    def check_loss(cls, value):
        if value not in ['triplet', 'bce', 'both']:
            raise ValueError(
                f"Loss type {value} not supported. Please choose between 'triplet', 'bce', 'both'"
            )
        return value

    @pydantic.field_validator('conv_type')
    def check_conv(cls, value):
        if value not in ['pna', 'egnn', 'dynedgeconv']:
            raise ValueError(
                f"Convolution type {value} not supported. Please choose between 'pna', 'egnn, 'dynedgeconv'"
            )
        return value

    @classmethod
    def from_yaml(cls, file_path: Path):
        # load the yaml file
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        # extract the data from the yaml file and create the object
        training = data['training']
        model = data['model']
        dataset = data['dataset']
        evaluation = data['evaluation']
        flags = data['flags']
        assert len(
            dataset['test_folders']
        ) == 1, "Only one test sequence is supported"  # leave-one-out approach
        evaluated_sequence = int(
            Path(dataset['test_folders'][0]).parts[-3][-2:])
        click.echo(
            click.style(
                f">> Testing on sequence {str(evaluated_sequence).zfill(2)}",
                fg='blue',
                bold=True))
        return cls(
            node_features=dataset['node_features'],
            edge_features=dataset['edge_features'],
            train_folders=list[Path](map(Path, dataset['train_folders'])),
            test_folders=list[Path](map(Path, dataset['test_folders'])),
            n_points_to_sample=dataset['n_points_to_sample'],
            n_points_in_local_neighborhood=dataset[
                'n_points_in_local_neighborhood'],
            n_jobs=dataset['n_jobs'],
            preprocessing_folder=dataset['preprocessing_folder'],
            max_dist2positive=dataset['max_distance_to_positive'],
            min_dist2negative=dataset['min_distance_to_negative'],
            debug_train_on_test_set=dataset['debug_train_on_test_set'],
            node_mpl_hidden_layer_size=model['node_preprocessing']
            ['node_hidden_size'],
            node_mpl_output_layer_size=model['node_preprocessing']
            ['node_output_size'],
            edge_mpl_hidden_layer_size=model['edge_preprocessing']
            ['edge_hidden_size'],
            edge_mpl_output_layer_size=model['edge_preprocessing']
            ['edge_output_size'],
            gnn_output_layer_size=model['gnn']['gnn_output_size'],
            gnn_n_towers=model['gnn']['n_towers'],
            gnn_pre_mpl_n_layers=model['gnn']['pre_layers'],
            gnn_post_mpl_n_layers=model['gnn']['post_layers'],
            pna_aggr=model['gnn']['pna_aggregators'],
            pna_scaler=model['gnn']['pna_scalers'],
            n_conv_layers=model['number_conv_layers'],
            k_nearest_neighbors=model['number_k_nearest_neighbors'],
            output_mpl_hidden_layer_size=model['output_mlp']
            ['hidden_output_size'],
            output_mpl_output_layer_size=model['output_mlp']['embedding_size'],
            batch_size=training['batch_size'],
            learning_rate=training['optimizer']['lr'],
            n_epochs=training['epochs'],
            num_workers=training['num_workers'],
            decay_epochs=list(training['scheduler']['milestones']),
            lr_decay_factor=training['scheduler']['decay_rate'],
            loss_margin=training['loss']['margin'],
            loss_type=training['loss']['type'],
            loss_norm_order=training['loss']['p'],
            date_time=datetime.now(),
            output_path=Path(
                f"./data/pickle_list/eval_seq_{str(evaluated_sequence).zfill(2)}"
            ),
            generate_triplets=flags['generate_triplets'],
            wandb_logging=flags['wandb_logging'],
            normalize_embeddings=flags['normalize_embeddings'],
            use_angles_in_edge_features=flags['use_angles_in_edge_features'],
            use_semantics_in_node_features=flags[
                'use_semantics_in_node_features'],
            use_semantics_in_graph_features=flags[
                'use_semantics_in_graph_features'],
            initialize_weigths_xavier=flags['initialize_weigths_xavier'],
            min_consistency_treshold=evaluation['consistency_tresh_min'],
            max_consistency_treshold=evaluation['consistency_tresh_max'],
            num_consistency_tresholds=evaluation['num_consistency_tresh'],
            checkpoint_path=training['checkpoint_path'],
            wandb_id=evaluation['wandb_id'],
            train=flags['train'],
            test=flags['test'],
            freeze_riconv=flags.get('freeze_riconv', False),
            pooling=model['pooling'],
            conv_type=model['conv_type'],
            time_window=evaluation['previous_time_window_to_evaluate'],
            max_dist_2_true_positive=evaluation['max_dist_2_true_positive'])

    def dump_to_yaml(self, file_path: Path):
        # use the structure of the class to create the yaml file in runtime
        dataset_dict = {
            "node_features": self.node_features,
            "edge_features": self.edge_features,
            "train_folders": [str(folder) for folder in self.train_folders],
            "test_folders": [str(folder) for folder in self.test_folders],
            "preprocessing_folder": self.preprocessing_folder,
            "max_distance_to_positive": self.max_dist2positive,
            "min_distance_to_negative": self.min_dist2negative,
            "n_points_to_sample": self.n_points_to_sample,
            "n_points_in_local_neighborhood":
            self.n_points_in_local_neighborhood,
            "n_jobs": self.n_jobs,
            "debug_train_on_test_set": self.debug_train_on_test_set
        }
        model_dict = {
            "node_preprocessing": {
                "node_hidden_size": self.node_mpl_hidden_layer_size,
                "node_output_size": self.node_mpl_output_layer_size
            },
            "edge_preprocessing": {
                "edge_hidden_size": self.edge_mpl_hidden_layer_size,
                "edge_output_size": self.edge_mpl_output_layer_size
            },
            "gnn": {
                "gnn_output_size": self.gnn_output_layer_size,
                "n_towers": self.gnn_n_towers,
                "pre_layers": self.gnn_pre_mpl_n_layers,
                "post_layers": self.gnn_post_mpl_n_layers,
                "pna_aggregators": self.pna_aggr,
                "pna_scalers": self.pna_scaler
            },
            "number_conv_layers": self.n_conv_layers,
            "number_k_nearest_neighbors": self.k_nearest_neighbors,
            "output_mlp": {
                "hidden_output_size": self.output_mpl_hidden_layer_size,
                "embedding_size": self.output_mpl_output_layer_size
            },
            "pooling": self.pooling,
            "conv_type": self.conv_type,
        }
        training_dict = {
            "batch_size": self.batch_size,
            "checkpoint_path": str(self.checkpoint_path),
            "optimizer": {
                "lr": self.learning_rate
            },
            "epochs": self.n_epochs,
            "num_workers": self.num_workers,
            "scheduler": {
                "milestones": self.decay_epochs,
                "decay_rate": self.lr_decay_factor
            },
            "loss": {
                "margin": self.loss_margin,
                "p": self.loss_norm_order,
                "type": self.loss_type
            }
        }
        flags_dict = {
            "generate_triplets": self.generate_triplets,
            "wandb_logging": self.wandb_logging,
            "normalize_embeddings": self.normalize_embeddings,
            "use_angles_in_edge_features": self.use_angles_in_edge_features,
            "use_semantics_in_node_features":
            self.use_semantics_in_node_features,
            "initialize_from_checkpoint": self.init_from_checkpoint,
            "initialize_weigths_xavier": self.initialize_weigths_xavier,
            "train": self.train,
            "test": self.test,
            "use_semantics_in_graph_features":
            self.use_semantics_in_graph_features,
            "freeze_riconv": self.freeze_riconv
        }
        evaluation_dict = {
            "previous_time_window_to_evaluate": self.time_window,
            "wandb_id": self.wandb_id,
            "consistency_tresh_min": self.min_consistency_treshold,
            "consistency_tresh_max": self.max_consistency_treshold,
            "num_consistency_tresh": self.num_consistency_tresholds,
            "max_dist_2_true_positive": self.max_dist_2_true_positive
        }
        data = {
            "dataset": dataset_dict,
            "model": model_dict,
            "training": training_dict,
            "flags": flags_dict,
            "evaluation": evaluation_dict
        }
        # dump the data to the yaml file
        with open(file_path, 'w') as file:
            yaml.dump(data, file)
