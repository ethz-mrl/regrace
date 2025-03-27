from pathlib import Path
from typing import Annotated, Literal

import click
import joblib
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from ..src.config import YAMLConfig
from ..src.map import Submap


def read_camera2velodyne_matrix(
        calib_file: Path) -> Annotated[npt.NDArray[np.float32], Literal[4, 4]]:
    assert calib_file.is_file(), f"{calib_file} does not exist"
    with open(calib_file, 'r') as f:
        calib = np.array([[float(x) for x in line.split()[1:]]
                          for line in f.readlines()]).astype(np.float32)
    camera2velodyne = np.vstack((calib[-1].reshape(
        (-1, 4)), np.array([0, 0, 0, 1]))).astype(np.float32)
    return camera2velodyne


def read_scan_poses(
    poses_file: str,
    calib_file: str,
) -> list[Annotated[npt.NDArray[np.float32], Literal[4, 4]]]:
    assert Path(poses_file).is_file(), f"{poses_file} does not exist"
    # read calibration file
    camera2velodyne = read_camera2velodyne_matrix(calib_file=Path(calib_file))

    # read poses file
    poses = np.loadtxt(poses_file).reshape((-1, 12)).astype(np.float32)

    # transform poses
    calibrated_poses = [(np.linalg.inv(camera2velodyne) @ np.vstack(
        (pose.reshape((-1, 4)), np.array([0, 0, 0, 1]))).astype(np.float32)
                         @ camera2velodyne) for pose in poses]

    return calibrated_poses


def read_timestamps(timestamps_file: str, ) -> list[float]:
    assert Path(timestamps_file).is_file(), f"{timestamps_file} does not exist"
    # read timestamps file
    timestamps = np.loadtxt(timestamps_file).astype(np.float32)
    return list(timestamps)


def get_scan_indexes_per_submap(
    dist_between_submaps: float,
    single_scan_poses: list[Annotated[npt.NDArray[np.float32], Literal[4, 4]]],
    index_of_first_scan: int = 0,
    interval_between_scans: int = 1,
) -> list[npt.NDArray[np.int32]]:

    # initialize variables
    list_of_scan_indexes: list[npt.NDArray[np.int32]] = []
    submap_poses: list[Annotated[npt.NDArray[np.float32], Literal[4, 4]]] = []

    # iterate over poses
    for k in range(index_of_first_scan, len(single_scan_poses),
                   interval_between_scans):
        last_scan = None
        submap_poses.append(single_scan_poses[k])
        for (i, pose) in enumerate(single_scan_poses[k + 1:]):
            # get distance to last pose
            distance = np.linalg.norm(pose[:3, 3] -
                                      single_scan_poses[k][:3, 3])
            # update last scan
            last_scan = i + k + 1
            # if distance is bigger than threshold, create new submap
            if distance > dist_between_submaps:
                break
        if last_scan is None: continue
        distance_bewteen_first_and_last = np.linalg.norm(
            single_scan_poses[last_scan][:3, 3] - single_scan_poses[k][:3, 3])
        if distance_bewteen_first_and_last < dist_between_submaps:
            continue
        list_of_scan_indexes.append(np.array(range(k, last_scan)))

    return list_of_scan_indexes


def create_submap(
    config: YAMLConfig,
    scan_index: list[int],
    poses: Annotated[npt.NDArray[np.float32], Literal["N", 4, 4]],
    timestamps: list[float],
    submap_number: int,
) -> None:
    # create submap
    submap = Submap.from_kitti_files(config=config,
                                     list_of_scan_indexes=scan_index,
                                     poses=poses,
                                     timestamps=timestamps,
                                     submap_number=submap_number)
    # save submap
    submap.save_pickle()


def generate_submaps(config: YAMLConfig, first_scan_index: int = 0) -> None:

    click.echo(
        click.style(
            f"Generating submaps for folder: {config.single_scans.scans_dir}",
            bold=True,
            fg='yellow'))

    # read poses and timestamps
    poses = read_scan_poses(poses_file=config.single_scans.poses_file,
                            calib_file=config.single_scans.calib_file)
    timestamps = read_timestamps(
        timestamps_file=config.single_scans.timestamp_file)

    # get scan indexes per submap
    list_of_scan_indexes = get_scan_indexes_per_submap(
        dist_between_submaps=config.dist_between_submaps,
        single_scan_poses=poses,
        index_of_first_scan=first_scan_index,
        interval_between_scans=config.interval_between_scans)

    # create submaps
    joblib.Parallel(n_jobs=config.n_workers)(joblib.delayed(create_submap)(
        config=config,
        scan_index=list(scan_index),
        poses=np.take(poses, scan_index, axis=0),
        timestamps=np.take(timestamps, scan_index),
        submap_number=i,
    ) for (i, scan_index) in enumerate(
        tqdm(list_of_scan_indexes, colour='YELLOW', dynamic_ncols=True)))
