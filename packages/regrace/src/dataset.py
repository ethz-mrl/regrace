import pickle
import random
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
import torch
import torch_geometric.data
from sklearn.neighbors import KDTree
from tqdm import tqdm

from .config import YAMLConfig
from .data import Data


class Dataset(torch_geometric.data.Dataset):
    data_list: list[Data]
    furthest_dist_between_points: float
    valid_idx: list[int]
    histogram: torch.Tensor

    def __getitem__(self, index: int) -> dict[str, Data]:
        # get real index
        idx = self.valid_idx[index]

        # get query Data
        query = self.data_list[idx]

        # get positive index and sample positive
        positive_idx = random.choice(query.positives)
        positive = self.data_list[positive_idx]

        return {'query': query, 'positive': positive}

    def __len__(self):
        return len(self.valid_idx)

    def __init__(
        self,
        data_list: list[Data],
        furthest_dist_between_points: float,
        valid_idx: list[int],
        histogram: torch.Tensor,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_list = data_list
        self.furthest_dist_between_points = furthest_dist_between_points
        self.valid_idx = valid_idx
        self.histogram = histogram

    @classmethod
    def from_pickle(cls, pickle_path: Path, **kwargs) -> 'Dataset':
        click.echo(
            click.style(f"Loading dataset from {pickle_path}...",
                        fg='yellow',
                        bold=True))
        # load data
        with open(pickle_path, 'rb') as f:
            data_dict = pickle.load(f)
        click.echo(
            click.style(
                f">> Dataset loaded with {len(data_dict['data_list'])} samples and {len(data_dict['data_list']) - len(data_dict['valid_idx'])} ignored samples.",
                fg='green'))
        # print ignored samples
        if len(data_dict['valid_idx']) < len(data_dict['data_list']):
            click.echo(
                click.style(
                    f"Ignored samples: {set(range(len(data_dict['data_list']))).difference(set(data_dict['valid_idx']))}",
                    fg='red'))
        # print histogram
        click.echo(
            click.style(
                f"Maximum cluster cardinality: {data_dict['histogram'].argmax()}",
                fg='green'))
        return cls(data_list=data_dict['data_list'],
                   furthest_dist_between_points=data_dict[
                       'furthest_dist_between_points'],
                   valid_idx=data_dict['valid_idx'],
                   histogram=data_dict['histogram'],
                   **kwargs)

    @staticmethod
    def _load_folders_to_df(
        list_of_folders: list[Path],
        n_jobs: int,
        normalization_constant: float | None,
        preprocessing_output_path: Path,
        n_points_to_sample: int,
        n_points_in_local_neighborhood: int,
    ) -> pd.DataFrame:
        df = pd.DataFrame(columns=[
            'submap_pickle_file', 'position', 'furthest_dist_between_points'
        ])

        # iterate over kitti folders and output paths
        for folder in list_of_folders:
            # echo folder
            click.echo(
                click.style(f"Processing folder {folder}...",
                            fg='yellow',
                            bold=True))

            # load config
            path_to_submap_dir = Path(folder)
            assert path_to_submap_dir.exists(
            ), f"Folder {path_to_submap_dir} not found."

            # find sequence name by the fourth parent folder (XX)
            sequence_number = int(path_to_submap_dir.parts[-3][-2:])

            # load data
            list_of_pickle_files = sorted(path_to_submap_dir.glob('*.pkl'))

            # read all data, save to pickle and get valid indexes and max bbox side
            df = pd.concat([
                df if not df.empty else None,
                Dataset._load_files_to_df(
                    list_of_pickle_files,
                    sequence_number,
                    preprocessing_output_path,
                    n_jobs,
                    n_points_to_sample,
                    n_points_in_local_neighborhood,
                )
            ],
                           ignore_index=True)

        # if normalization constant is a float, use it as furthest_dist_between_points
        if normalization_constant is not None:
            assert isinstance(
                normalization_constant, float
            ), "Normalization constant must be a float for test split."
            assert normalization_constant > 0, "Normalization constant must be positive."
            df['furthest_dist_between_points'] = normalization_constant
        return df

    @staticmethod
    def _load_files_to_df(list_of_pickle_files: list[Path],
                          sequence_number: int,
                          preprocessing_output_path: Path, n_jobs: int,
                          n_points_to_sample: int,
                          n_points_in_local_neighborhood: int) -> pd.DataFrame:
        # create dataframe
        df = pd.DataFrame(columns=[
            'submap_pickle_file', 'position', 'furthest_dist_between_points'
        ])
        distance_offset = np.array(
            [1, 1, 1]
        ) * sequence_number * 1e5  # avoid overlapping sequences when computing positives
        click.echo(
            click.style(
                f"Offsetting position by {distance_offset} for sequence {sequence_number}.",
                fg='yellow'))

        # create directories if not exist
        output_folder = preprocessing_output_path / f'{sequence_number:02d}'
        output_folder.mkdir(parents=True, exist_ok=True)
        (output_folder / 'parquet').mkdir(parents=True, exist_ok=True)
        (output_folder / 'pickle').mkdir(parents=True, exist_ok=True)

        # iterate over submaps
        processed_tuple = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(Data.pre_process_submap)(
                submap_file,
                preprocessing_output_path / f'{sequence_number:02d}',
                n_points_to_sample,
                n_points_in_local_neighborhood,
            ) for submap_file in tqdm(
                list_of_pickle_files, colour='yellow', dynamic_ncols=True))

        # add to dataframe
        for (
                furthest_dist_between_points,  # type: ignore
                position,
                data_pickle_file) in processed_tuple:
            # append to dataframe
            if furthest_dist_between_points > 0:
                df.loc[len(df)] = [
                    data_pickle_file, position + distance_offset,
                    furthest_dist_between_points
                ]

        # echo how many samples were read or ignored
        click.echo(
            click.style(
                f"Read {len(df)} samples, ignored {len(list_of_pickle_files) - len(df)} samples with one or no bounding box.",
                fg='yellow'))
        click.echo(
            click.style(
                f">> All samples within X: [{df['position'].apply(lambda x: x[0]).min()},{df['position'].apply(lambda x: x[0]).max()}] and Y: [{df['position'].apply(lambda x: x[1]).min()}, {df['position'].apply(lambda x: x[1]).max()}]",
                fg='yellow'))
        return df

    @staticmethod
    def _save_data_dict_to_pickle(data_filename: Path, position: np.ndarray,
                                  positives: list[int],
                                  non_negatives: list[int]) -> Data:
        return Data(data_filename, position, positives, non_negatives)

    @staticmethod
    def _save_df_to_data_dict(df: pd.DataFrame, n_jobs: int,
                              max_dist_2_positive: float,
                              min_dist_2_negative: float,
                              output_path: Path) -> None:
        # get normalization constant
        furthest_distances_between_points = df[
            'furthest_dist_between_points'].to_numpy()
        furthest_dist_between_points = np.quantile(
            furthest_distances_between_points, 0.95)
        click.echo(
            click.style(
                f"95% largest furthest distance from within all clusters: {furthest_dist_between_points}",
                fg='cyan'))

        # construct tree
        positions_XY = np.concatenate(
            df['position'].to_numpy(),
            dtype=np.float64,
        ).reshape(-1, 3)[:, :2].astype(np.float64)  # assert no precision loss
        tree = KDTree(positions_XY)
        idx_positives = tree.query_radius(positions_XY, r=max_dist_2_positive)
        idx_non_negatives = tree.query_radius(positions_XY,
                                              r=min_dist_2_negative)

        # create data dictionary
        if output_path.exists():
            click.confirm(click.style(
                f"Output folder {output_path} already exists. Continue?",
                fg='red'),
                          default=True,
                          abort=True)
        assert output_path.suffix == '.pkl', "The output path must be a pickle file."
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # save data to pickle
        click.echo(
            click.style("Saving data to pickle...", fg='cyan', bold=True))
        data_list = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(Dataset._save_data_dict_to_pickle)
            (data_filename, position,
             [idx
              for idx in idx_positives[i] if idx != i], idx_non_negatives[i])
            for i, data_filename, position, _ in tqdm(df.itertuples(),
                                                      total=len(df),
                                                      colour='cyan',
                                                      dynamic_ncols=True))
        assert isinstance(data_list, list), 'The data_list must be a list'
        assert isinstance(data_list[0],
                          Data), 'The data_list must be a list of Data'
        assert len(data_list) == len(
            df), 'The data_list must have the same length as the dataframe'

        # filter data with no positives
        valid_idx = [
            i for i, data in enumerate(data_list)
            if len(data.positives) > 0  # type: ignore
        ]

        # save data to pickle
        final_dict = {
            'valid_idx': valid_idx,
            'data_list': data_list,
            'furthest_dist_between_points': furthest_dist_between_points,
            'histogram':
            Dataset.histogram_of_cardinality(data_list, n_jobs)  # type: ignore
        }

        # echo how many samples were read or ignored
        click.echo(
            click.style(
                f"Read {len(valid_idx)} samples, ignored {len(data_list) - len(valid_idx)} samples with no positives.",
                fg='cyan'))

        # save to pickle
        with open(output_path, 'wb') as f:
            pickle.dump(final_dict, f)

    @classmethod
    def from_config(
        cls,
        config: YAMLConfig,
        split: str,
        normalization_constant: float | None = None,
        **kwargs,
    ) -> 'Dataset':

        # check overlap between folders
        assert len(
            set(config.train_folders).intersection(set(config.test_folders))
        ) == 0, "Training and testing folders overlap."

        # check if split is valid
        assert split in [
            'train', 'test'
        ], f"Split {split} is not valid. Choose between 'train' and 'test'."
        if split == 'train':
            list_of_folders = config.train_folders
        else:
            list_of_folders = config.test_folders

        # if loading test, use normalization constant from train
        if split == 'test':
            assert normalization_constant is not None, "Normalization constant must be provided for test split."

        # iterate over input folders
        df = cls._load_folders_to_df(list_of_folders, config.n_jobs,
                                     normalization_constant,
                                     Path(config.preprocessing_folder),
                                     config.n_points_to_sample,
                                     config.n_points_in_local_neighborhood)

        # save data to pickle
        cls._save_df_to_data_dict(
            df, config.n_jobs, config.max_dist2positive,
            config.min_dist2negative, config.output_path /
            f"{split}_maxPos{int(config.max_dist2positive)}_minNeg{int(config.min_dist2negative)}.pkl"
        )

        # return dataset
        return cls.from_pickle(
            config.output_path /
            f"{split}_maxPos{int(config.max_dist2positive)}_minNeg{int(config.min_dist2negative)}.pkl",
            **kwargs)

    @staticmethod
    def histogram_of_cardinality(data_list: list[Data],
                                 n_jobs: int) -> torch.Tensor:
        import warnings
        warnings.warn("This method assumes that all nodes are connected.")
        histogram = torch.zeros(500,
                                dtype=torch.int)  # guess the max cardinality

        def get_number_of_clusters(data: Data) -> int:
            return data.number_of_clusters

        # get number of clusters per data
        n_clusters_in_data_list = joblib.Parallel(
            n_jobs=n_jobs,
            return_as='list')(joblib.delayed(get_number_of_clusters)(data)
                              for data in tqdm(
                                  data_list,
                                  colour='blue',
                                  desc='Counting clusters...',
                                  dynamic_ncols=True,
                              ))

        # drop numbers in histogram
        unique_number_of_clusters, counts = torch.unique(
            torch.tensor(n_clusters_in_data_list),
            return_counts=True,
        )
        histogram[unique_number_of_clusters - 1] = counts.int()

        # cut the histogram
        return histogram[:torch.nonzero(histogram).max().item() + 1]

    @property
    def positions(self) -> torch.Tensor:
        return torch.tensor(np.array([data["query"].position
                                      for data in self]))

    @property
    def timestamps(self) -> torch.Tensor:
        return torch.tensor([data["query"].timestamp for data in self]).float()
