from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt

from ..src.pointcloud import PointCloud


def get_accumulated_points(
    pointclouds: list[PointCloud], ref_pose: Annotated[npt.NDArray[np.float32],
                                                       Literal[4, 4]]
) -> Annotated[npt.NDArray[np.float32], Literal[3, "N"]]:
    points = np.empty((4, 0), dtype=np.float32)
    ref_pose_inv = np.linalg.inv(ref_pose)
    for pointcloud in pointclouds:
        points = np.hstack(
            (points, (ref_pose_inv @ pointcloud.pose @ np.vstack(
                (pointcloud.points, np.ones(
                    (1, pointcloud.points.shape[1])))))))
    return points[:3].astype(np.float32)


def get_accumulated_labels_probabilities(
    pointclouds: list[PointCloud]
) -> Annotated[npt.NDArray[np.float32], Literal["N", 20]]:
    labels_probabilities = np.empty((0, 20), dtype=np.float32)
    for pointcloud in pointclouds:
        labels_probabilities = np.vstack(
            (labels_probabilities, pointcloud.probability_labels))
    return labels_probabilities.astype(np.float32)
