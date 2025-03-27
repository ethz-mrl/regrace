from datetime import datetime
from pathlib import Path

import pydantic
import yaml


class KITTIFolder(pydantic.BaseModel):
    scans_dir: str
    poses_file: str
    _timestamp_file: str = pydantic.PrivateAttr('')
    calib_file: str
    _pickle_dir: str = pydantic.PrivateAttr('')
    predicted_labels_dir: str
    predicted_labels_prob_array_dir: str

    @property
    def pickle_dir(self) -> str:
        assert self._pickle_dir != '', "Pickle directory not set"
        return self._pickle_dir

    @pickle_dir.setter
    def pickle_dir(self, value: str):
        self._pickle_dir = value

    @property
    def timestamp_file(self) -> str:
        assert self._timestamp_file != '', "Timestamp file not set"
        return self._timestamp_file

    @timestamp_file.setter
    def timestamp_file(self, value: str):
        self._timestamp_file = value

    def __init__(self, **data):
        super().__init__(scans_dir=data['scans_dir'],
                         poses_file=data['poses_file'],
                         calib_file=data['calib_file'],
                         predicted_labels_dir=data['predicted_labels_dir'],
                         predicted_labels_prob_array_dir=data[
                             'predicted_labels_prob_array_dir'])
        if 'pickle_dir' in data:
            self._pickle_dir = data['pickle_dir']
        if 'timestamp_file' in data:
            self._timestamp_file = data['timestamp_file']


class YAMLConfig(pydantic.BaseModel):

    # folder data
    sequence: int
    sequence_str: str
    single_scans: KITTIFolder
    submap: KITTIFolder
    output_dir: str

    # accumulation
    dist_between_submaps: float
    voxel_size: float
    interval_between_scans: int

    # clustering
    min_points_per_cluster: int
    max_dist_between_points_in_cluster: float
    min_number_of_neighbours: int
    ignore_labels: list[str]
    cluster_folder: str

    # flags
    force_accumulation: bool
    force_clustering: bool

    # unique identifier
    date_time: datetime

    # runtime variables
    n_workers: int = 1

    @classmethod
    def from_kitti_yaml(cls, yaml_path: str):
        # read yaml file
        assert yaml_path.endswith('.yaml'), f"{yaml_path} is not a yaml file"
        assert Path(yaml_path).is_file(), f"{yaml_path} does not exist"
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        if 'date_time' not in config:
            config['date_time'] = datetime.now()

        # read config
        sequence = config['sequence']
        dist_between_submaps = config['dist_between_submaps']
        voxel_size = config['voxel_size']
        kitti_dir = config['kitti_dir']

        # sequence string
        sequence_str = str(sequence).zfill(2)

        # output directory
        output_dir = f'{config["output_folder"]}/seq{sequence_str}/submap/'
        cluster_folder = f'{output_dir}/cluster/'

        # generate paths
        assert Path(kitti_dir).is_dir(), f"{kitti_dir} does not exist"
        single_scans = KITTIFolder(
            scans_dir=f'{kitti_dir}/sequences/{sequence_str}/velodyne/',
            poses_file=f'{kitti_dir}/sequences/{sequence_str}/poses.txt',
            calib_file=f'{kitti_dir}/sequences/{sequence_str}/calib.txt',
            predicted_labels_dir=
            f'{config["output_folder"]}/seq{sequence_str}/single-scan/predicted-label/',
            predicted_labels_prob_array_dir=
            f'{config["output_folder"]}/seq{sequence_str}/single-scan/predicted-label-probability/',
            timestamp_file=f'{kitti_dir}/sequences/{sequence_str}/times.txt')
        submap = KITTIFolder(scans_dir=f'{output_dir}/velodyne/',
                             poses_file='',
                             calib_file='',
                             predicted_labels_dir='',
                             predicted_labels_prob_array_dir='',
                             pickle_dir=f'{output_dir}/all-points/',
                             timestamp_file=f'{output_dir}/timestamps.txt')

        # return instance
        return cls(sequence=sequence,
                   sequence_str=sequence_str,
                   single_scans=single_scans,
                   submap=submap,
                   dist_between_submaps=dist_between_submaps,
                   voxel_size=voxel_size,
                   min_points_per_cluster=config['min_points_per_cluster'],
                   max_dist_between_points_in_cluster=config[
                       'max_dist_between_points_in_cluster'],
                   min_number_of_neighbours=config['min_number_of_neighbours'],
                   force_accumulation=config['force_accumulation'],
                   force_clustering=config['force_clustering'],
                   date_time=config['date_time'],
                   output_dir=output_dir,
                   ignore_labels=config['ignore_labels'],
                   cluster_folder=cluster_folder,
                   interval_between_scans=config['interval_between_scans'],
                   n_workers=config['num_workers'])
