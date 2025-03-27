import os
from pathlib import Path

import click

from .src.config import YAMLConfig
from .utils.generate_cluster import generate_clusters
from .utils.generate_labels import generate_labels
from .utils.generate_submaps import generate_submaps


def main(config_path: str) -> None:
    click.echo(
        click.style(f">> Generating data using config: {config_path}",
                    bold=True,
                    fg='blue'))

    config = YAMLConfig.from_kitti_yaml(config_path)
    # generate labels if not already generated
    if not Path(config.single_scans.predicted_labels_dir).exists(
    ) or not os.listdir(config.single_scans.predicted_labels_dir):
        generate_labels(config=config)
    # generate submaps
    if not Path(
            config.submap.pickle_dir).exists() or config.force_accumulation:
        if Path(config.output_dir).exists():  # empty output_dir
            os.system(f"rm -r {config.output_dir}")
        os.system(f"mkdir {config.output_dir}")
        generate_submaps(config=config)
    # cluster
    if not Path(config.cluster_folder).exists() or config.force_clustering:
        if Path(config.cluster_folder).exists():  # empty cluster_folder
            os.system(f"rm -r {config.cluster_folder}")
        os.system(f"mkdir {config.cluster_folder}")
        generate_clusters(config=config)
