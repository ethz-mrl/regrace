import pickle

import numpy as np
import numpy.typing as npt
import open3d as o3d
from numba import njit
from tailwind_colors import TAILWIND_COLORS_HEX

from .yaml_utils import (get_label_inverse_learning_map, get_label_names)

cyl2kitti_label_map = get_label_inverse_learning_map()
label_names_map = get_label_names()
colors_per_label: dict[str, str] = {
    "building": TAILWIND_COLORS_HEX.CYAN_400,
    "car": TAILWIND_COLORS_HEX.RED_400,
    "truck": TAILWIND_COLORS_HEX.GREEN_400,
    "bicycle": TAILWIND_COLORS_HEX.PINK_400,
    "bus": TAILWIND_COLORS_HEX.YELLOW_400,
    "motorcycle": TAILWIND_COLORS_HEX.AMBER_400,
    "on-rails": TAILWIND_COLORS_HEX.ORANGE_400,
    "trunk": TAILWIND_COLORS_HEX.GREEN_400,
    "other-vehicle": TAILWIND_COLORS_HEX.EMERALD_400,
    "person": TAILWIND_COLORS_HEX.ROSE_400,
    "bicyclist": TAILWIND_COLORS_HEX.PINK_500,
    "motorcyclist": TAILWIND_COLORS_HEX.AMBER_500,
    "fence": TAILWIND_COLORS_HEX.BLUE_500,
    "other-structure": TAILWIND_COLORS_HEX.INDIGO_500,
    "pole": TAILWIND_COLORS_HEX.PURPLE_500,
    "traffic-sign": TAILWIND_COLORS_HEX.PINK_600,
    "moving-car": TAILWIND_COLORS_HEX.RED_500,
    "moving-bicyclist": TAILWIND_COLORS_HEX.PINK_600,
    "moving-person": TAILWIND_COLORS_HEX.ROSE_500,
    "moving-motorcyclist": TAILWIND_COLORS_HEX.AMBER_600,
    "moving-on-rails": TAILWIND_COLORS_HEX.ORANGE_500,
    "moving-bus": TAILWIND_COLORS_HEX.YELLOW_500,
    "moving-truck": TAILWIND_COLORS_HEX.GREEN_500,
    "moving-other-vehicle": TAILWIND_COLORS_HEX.EMERALD_500,
    "road": TAILWIND_COLORS_HEX.ORANGE_500,
    "sidewalk": TAILWIND_COLORS_HEX.PINK_500,
    "vegetation": TAILWIND_COLORS_HEX.GREEN_500,
}

# define numba functions for speedup
@njit
def jit_argmax(arr: npt.NDArray[np.float32]) -> int:
    if arr.size == 0:
        return -1
    return int(arr.argmax())


@njit
def jit_mean(arr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    res = []
    for i in range(arr.shape[1]):
        res.append(arr[:, i].mean())
    return np.array(res).reshape(-1)


@njit
def jit_sum(arr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    res = []
    for i in range(arr.shape[1]):
        res.append(arr[:, i].sum())
    return np.array(res).reshape(-1)


def voxelize_pointcloud(
    points: npt.NDArray[np.float32],
    voxel_size: float,
    probability_labels: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int8],
           npt.NDArray[np.float64]]:
    # create open3d pointcloud
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(points)

    # voxelization
    o3d_pc_voxelized, _, points_idx_per_voxel = o3d_pc.voxel_down_sample_and_trace(
        voxel_size=voxel_size,
        min_bound=o3d_pc.get_min_bound(),
        max_bound=o3d_pc.get_max_bound())
    assert o3d_pc_voxelized.has_points(), \
        "Voxelized pointcloud is empty"

    # init vectors for voxelized labels
    voxelized_points = np.asarray(o3d_pc_voxelized.points)
    voxelized_semantic_labels = np.zeros(
        (np.asarray(voxelized_points).shape[0], 1)).astype(np.int8)
    voxelized_probability_labels = np.zeros(
        (np.asarray(voxelized_points).shape[0], 20)).astype(np.float64)

    # iterate over voxels
    for (voxel, points) in enumerate(points_idx_per_voxel):
        # get average label probabilities for voxel
        labels_prob = jit_mean(probability_labels[points])  # type: ignore
        voxelized_probability_labels[voxel] = labels_prob
        # get most frequent semantic label for voxel
        voxelized_semantic_labels[voxel] = cyl2kitti_label_map[jit_argmax(
            labels_prob)]

    return voxelized_points, voxelized_semantic_labels.astype(
        np.int64), voxelized_probability_labels  # type: ignore


def get_valid_clusters_labels(
        cluster_labels: npt.NDArray[np.int64],
        min_points_per_cluster: int,
        force_invalid_label: bool = False) -> npt.NDArray[np.int64]:
    # check if force_invalid_label is set
    if force_invalid_label:
        return np.full_like(cluster_labels, -1)
    # get unique clusters and their counts
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    invalid_clusters = unique_clusters[(counts < min_points_per_cluster)]
    # set invalid clusters to -1
    cluster_labels[np.isin(cluster_labels, invalid_clusters)] = -1

    return cluster_labels
