import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pydantic
import torch
from scipy.linalg import expm

from ...data_generation import Submap
from ..utils.pointnet2_utils import furthest_point_sample
from ..utils.riconv2_utils import compute_LRA


def _M(theta):
    return expm(
        np.cross(
            np.eye(3),
            np.array([0, 0, 1]) / np.linalg.norm(np.array([0, 0, 1])) * theta))


class Data(pydantic.BaseModel):
    parquet_file: Path = pydantic.Field(Path())
    position: np.ndarray = pydantic.Field(np.empty(3))
    pose: np.ndarray = pydantic.Field(np.empty(4))
    positives: list[int] = pydantic.Field([])
    non_negatives: list[int] = pydantic.Field([])
    timestamp: float = pydantic.Field(-100.0)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, data_path: Path, position: np.ndarray,
                 positives: list[int], non_negatives: list[int]):
        super().__init__()
        data_item = pickle.load(open(data_path, "rb"))
        self.parquet_file = data_item["parquet_file"]
        self.timestamp = data_item["timestamp"]
        self.position = position
        self.positives = positives
        self.non_negatives = non_negatives
        self.pose = data_item["pose"]

    @property
    def df(self) -> pd.DataFrame:
        # load parquet file
        return pd.read_parquet(self.parquet_file)

    def to_collate_format(
        self, normalization_constant: float, transform: bool
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        # get the cluster points, centers and label probabilities
        df = self.df.groupby("cluster_id")
        cluster_points = df[[
            'x', 'y', 'z', 'normal_x', 'normal_y', 'normal_z'
        ]].apply(lambda x: torch.from_numpy(
            x.to_numpy(  # type: ignore
            ) / normalization_constant)).to_list()
        cluster_centers = torch.from_numpy(df.mean()[[
            'center_x', 'center_y', 'center_z'
        ]].to_numpy()) / normalization_constant
        label_probabilities = torch.from_numpy(
            df.mean()[[f'label_prob_{i}' for i in range(20)]].to_numpy())

        # add data augmentation (random rotation)
        if transform:
            R = _M((np.pi) * 2 * (np.random.rand(1) - 0.5))
            for i, points in enumerate(cluster_points):
                points[:, :3] = points[:, :3] @ R
                points[:, 3:] = points[:, 3:] @ R
                cluster_centers[i] = cluster_centers[i] @ R

        return cluster_points, cluster_centers, label_probabilities

    @property
    def number_of_clusters(self) -> int:
        return len(self.df.groupby('cluster_id'))

    @classmethod
    def from_pickle(cls, path: str) -> 'Data':
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def pre_process_submap(
            submap_pickle_file: Path, output_folder: Path,
            n_points_to_sample: int, n_points_in_local_neighborhood: int
    ) -> tuple[float, np.ndarray, str]:
        # load submap
        submap = Submap.from_pickle(str(submap_pickle_file))

        # get the normalized constant as the biggest distance between two points in the submap
        if submap.points.shape[1] < 2:
            return -1, submap.position.astype(
                np.float32), str(submap_pickle_file)

        # check output folder
        sequence_output_folder = output_folder
        assert sequence_output_folder.exists(
        ), f"Output folder {sequence_output_folder} not found. Since this code is parallelized, please create the directory manually to avoid deadlock."
        assert Path(sequence_output_folder / 'parquet').exists(
        ), f"Output folder {sequence_output_folder / 'parquet'} not found. Since this code is parallelized, please create the directory manually to avoid deadlock."
        assert Path(sequence_output_folder / 'pickle').exists(
        ), f"Output folder {sequence_output_folder / 'pickle'} not found. Since this code is parallelized, please create the directory manually to avoid deadlock."

        # get the parquet filename and data from submap
        parquet_filename = sequence_output_folder / 'parquet' / f"{submap.submap_number:06d}.parquet"

        # construct dataframe
        column_types = {
            'x': np.float32,
            'y': np.float32,
            'z': np.float32,
            'normal_x': np.float32,
            'normal_y': np.float32,
            'normal_z': np.float32,
            'center_x': np.float32,
            'center_y': np.float32,
            'center_z': np.float32,
            'cluster_id': np.int32,
        }
        for i in range(20):
            column_types[f'label_prob_{i}'] = np.float32
        df = pd.DataFrame({
            col: pd.Series(dtype=typ)
            for col, typ in column_types.items()
        })

        # get data from submap
        points = submap.points
        cluster_ids = submap.cluster_ids
        unique_cluster_ids = np.unique(cluster_ids)
        if len(unique_cluster_ids) < 2:  # avoid submaps with only one cluster
            return -1, submap.position.astype(
                np.float32), str(submap_pickle_file)

        # sample points for each cluster and add to dataframe
        furthest_distance = -1.0
        for cluster_id in unique_cluster_ids:
            # get the index of the points in the cluster
            index_of_points_in_cluster = (cluster_ids == cluster_id)
            # sample points
            resampled_points_index = furthest_point_sample(
                torch.Tensor(
                    points.T[index_of_points_in_cluster]).contiguous().to(
                        'cuda').unsqueeze(0),
                n_points_to_sample * 5).detach().cpu(  # type: ignore
                ).squeeze().numpy()

            # compute the furthest distance
            furthest_distance_in_cluster = np.linalg.norm(
                points[:, index_of_points_in_cluster].T[
                    resampled_points_index[0]] -
                points[:, index_of_points_in_cluster].T[
                    resampled_points_index[1]])
            if furthest_distance_in_cluster > furthest_distance:
                furthest_distance = furthest_distance_in_cluster
            # compute cluster center
            center = points[:, index_of_points_in_cluster].T.mean(axis=0)
            centered_points = points[:, index_of_points_in_cluster].T[
                resampled_points_index] - center
            # compute normals
            normals = compute_LRA(
                torch.tensor(np.expand_dims(centered_points, 0)).to('cuda'),
                True,
                nsample=n_points_in_local_neighborhood).squeeze().detach().cpu(
                ).numpy()

            # accumulate to dataframe
            df = pd.concat([
                df,
                pd.DataFrame({
                    'x':
                    centered_points[:n_points_to_sample, 0].astype(np.float32),
                    'y':
                    centered_points[:n_points_to_sample, 1].astype(np.float32),
                    'z':
                    centered_points[:n_points_to_sample, 2].astype(np.float32),
                    'normal_x':
                    normals[:n_points_to_sample, 0].astype(np.float32),
                    'normal_y':
                    normals[:n_points_to_sample, 1].astype(np.float32),
                    'normal_z':
                    normals[:n_points_to_sample, 2].astype(np.float32),
                    'center_x':
                    center[0].astype(np.float32) *
                    np.ones(n_points_to_sample, dtype=np.float32),
                    'center_y':
                    center[1].astype(np.float32) *
                    np.ones(n_points_to_sample, dtype=np.float32),
                    'center_z':
                    center[2].astype(np.float32) *
                    np.ones(n_points_to_sample, dtype=np.float32),
                    'cluster_id':
                    cluster_id * np.ones(n_points_to_sample, dtype=np.int32),
                    **{
                        f'label_prob_{i}':
                        submap.probability_labels[index_of_points_in_cluster][resampled_points_index[:n_points_to_sample], i].astype(np.float32)
                        for i in range(submap.probability_labels.shape[1])
                    }
                })
            ],
                           ignore_index=True)

        # sanity check
        assert furthest_distance > 0, "Furthest distance is zero, no valid clusters were found."

        # save to parquet
        df.to_parquet(parquet_filename, index=False, engine='fastparquet')

        # save to pickle
        pkl_filename = sequence_output_folder / 'pickle' / f"{submap.submap_number:06d}.pkl"
        dict_data = {
            "parquet_file": parquet_filename,
            "pose": submap.pose.astype(np.float32),
            "timestamp": submap.timestamp,
        }
        pickle.dump(dict_data, open(pkl_filename, "wb"))
        return float(furthest_distance), submap.position.astype(
            np.float32), str(pkl_filename)
