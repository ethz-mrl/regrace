import pickle
from pathlib import Path
from typing import Annotated, Literal, Optional

import numpy as np
import numpy.typing as npt
import pydantic
from sklearn.cluster import DBSCAN

from ..utils.map_utils import (get_valid_clusters_labels, label_names_map,
                               voxelize_pointcloud)
from ..utils.pointcloud_utils import (get_accumulated_labels_probabilities,
                                      get_accumulated_points)
from .config import YAMLConfig
from .pointcloud import PointCloud


class Submap(pydantic.BaseModel):
    points: Annotated[npt.NDArray[np.float32],
                      Literal[3, "N"]] = pydantic.Field(..., min_length=3)
    semantic_labels: Annotated[npt.NDArray[np.uint32],
                               Literal["N"]] = pydantic.Field(
                                   ..., min_length=1)
    probability_labels: Annotated[npt.NDArray[np.float32],
                                  Literal["N", 20]] = pydantic.Field(
                                      ..., min_length=20)
    pose: Annotated[npt.NDArray[np.float32],
                    Literal[4, 4]] = pydantic.Field(...,
                                                    min_length=4,
                                                    max_length=4)
    timestamp: float
    config: YAMLConfig
    submap_number: int
    cluster_ids: Optional[Annotated[npt.NDArray[np.int32],
                                    Literal["N"]]] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def position(self) -> Annotated[npt.NDArray[np.float32], Literal["N", 3]]:
        return self.pose[:3, 3]

    @classmethod
    def from_pickle(cls, filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def __init__(self, pointclouds: list[PointCloud], config: YAMLConfig,
                 timestamp: float, pose: Annotated[npt.NDArray[np.float32],
                                                   Literal[4, 4]],
                 submap_number: int):

        # check input parameters
        points = get_accumulated_points(pointclouds, pose)
        probability_labels = get_accumulated_labels_probabilities(pointclouds)

        # generate pointcloud for voxelization
        voxelized_points, voxelized_semantic_labels, voxelized_probability_labels = voxelize_pointcloud(
            points=points.T,
            voxel_size=config.voxel_size,
            probability_labels=probability_labels)

        # write voxelized pointcloud to class variables
        points = np.vstack(
            (voxelized_points.T, np.ones(
                (1, voxelized_points.shape[0])))).astype(np.float32)
        semantic_labels = voxelized_semantic_labels.astype(np.uint8)
        probability_labels = voxelized_probability_labels.astype(np.float32)

        # call super init
        super().__init__(points=points,
                         semantic_labels=semantic_labels,
                         pose=pose,
                         timestamp=timestamp,
                         config=config,
                         submap_number=submap_number,
                         probability_labels=probability_labels)

    def save_pickle(self) -> None:
        # define filename for pickle depending on if clustering was done
        if self.cluster_ids is None:
            filename = f"{self.config.submap.pickle_dir}/{self.submap_number:06d}.pkl"
        else:
            filename = f"{self.config.cluster_folder}/{self.submap_number:06d}.pkl"
        # create output directory
        if not Path(filename).parent.is_dir():
            Path(filename).parent.mkdir(parents=True)
        # save pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_kitti_files(cls, config: YAMLConfig,
                         list_of_scan_indexes: list[int], submap_number: int,
                         timestamps: list[float],
                         poses: Annotated[npt.NDArray[np.float32],
                                          Literal["N", 4, 4]]):
        pointclouds: list[PointCloud] = []
        # get middle index
        middle_index = len(list_of_scan_indexes) // 2
        for scan_number in list_of_scan_indexes:
            # define paths
            points_file = f"{config.single_scans.scans_dir}/{scan_number:06d}.bin"
            probability_labels_file = f"{config.single_scans.predicted_labels_prob_array_dir}/{scan_number:06d}.label"

            # check if files exist
            assert Path(points_file).is_file(), \
                f"{points_file} does not exist"
            assert Path(probability_labels_file).is_file(), \
                f"{probability_labels_file} does not exist"

            # try to read files for five times in case of error
            error_counter = 0
            while error_counter < 5:
                try:
                    points = np.fromfile(points_file,
                                         dtype=np.float32).reshape(
                                             (-1, 4))[:, 0:3]
                    probability_labels = np.fromfile(probability_labels_file,
                                                     dtype=np.float32).reshape(
                                                         (-1, 20))
                    break
                except IOError:
                    print(
                        f"Error reading {points_file} or {probability_labels_file}, trying again..."
                    )
                    error_counter += 1
            if error_counter == 5:
                raise IOError(
                    f"Error reading {points_file} or {probability_labels_file}"
                )

            # create pointcloud list
            pointclouds.append(
                PointCloud(points=points,
                           pose=poses[scan_number - min(list_of_scan_indexes)],
                           timestamp=timestamps[scan_number -
                                                min(list_of_scan_indexes)],
                           probability_labels=probability_labels))

        return cls(pointclouds=pointclouds,
                   config=config,
                   pose=poses[middle_index],
                   timestamp=timestamps[middle_index],
                   submap_number=submap_number)

    def cluster(self, dbscan: DBSCAN) -> None:

        # create tmp points, semantic labels and cluster_id arrays
        points = np.empty((0, 3), dtype=np.float32)
        semantic_labels = np.empty((0, ), dtype=np.uint32)
        probability_labels = np.empty((0, 20), dtype=np.float32)
        cluster_ids = np.empty((0, ), dtype=np.int32)

        # iterate over labels
        for label in label_names_map.keys():
            # get indexes of points with each label
            indexes_of_points_with_label = np.where(
                self.semantic_labels == label)[0]

            # check if label should be ignored
            force_invalid_label = False
            if len(indexes_of_points_with_label) == 0:
                continue  # skip if no points with label
            if (label_names_map[label] in self.config.ignore_labels) or (
                    len(indexes_of_points_with_label)
                    < self.config.min_points_per_cluster):
                force_invalid_label = True

            # get points with label
            points_3D = self.points[:, indexes_of_points_with_label][:3, :].T

            # cluster and check if there are enough points in the cluster
            cluster_labels = dbscan.fit_predict(points_3D)
            valid_clusters_labels = get_valid_clusters_labels(
                cluster_labels, self.config.min_points_per_cluster,
                force_invalid_label)
            valid_mask = np.ones_like(valid_clusters_labels, dtype=bool)
            sum_mask = (valid_clusters_labels
                        != -1) * (max(cluster_ids, default=-1) + 1)
            valid_clusters_labels += sum_mask
            del sum_mask

            # add points to cluster object
            points = np.vstack((points, points_3D[valid_mask]))
            semantic_labels = np.hstack(
                (semantic_labels, np.full(points_3D[valid_mask].shape[0],
                                          label)))
            probability_labels = np.vstack(
                (probability_labels,
                 self.probability_labels[indexes_of_points_with_label]
                 [valid_mask]))
            cluster_ids = np.hstack(
                (cluster_ids, valid_clusters_labels.astype(np.int32)))

        # check if cluster is empty
        if len(points) == 0:
            raise ValueError("No clusters found")
        # save cluster ids
        self.cluster_ids = cluster_ids
        self.points = points.T
        self.semantic_labels = semantic_labels
        self.probability_labels = probability_labels

    @property
    def cluster_centers(
            self) -> Annotated[npt.NDArray[np.float32], Literal["N", 3]]:
        assert self.cluster_ids is not None, \
                "Cluster ids not found. Run clustering first."
        cluster_centers = np.empty((0, 3), dtype=np.float32)
        for cluster_id in np.unique(self.cluster_ids):
            cluster_centers = np.vstack(
                (cluster_centers,
                 np.mean(self.points.T[self.cluster_ids == cluster_id],
                         axis=0)))
        return cluster_centers
