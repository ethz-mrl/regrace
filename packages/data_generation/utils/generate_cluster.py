from pathlib import Path

import click
import joblib
import sklearn.cluster
import sklearn.metrics.cluster
from tqdm import tqdm

from ..src.config import YAMLConfig
from ..src.map import Submap


def cluster_submap(
    pickle_file: Path,
    dbscan: sklearn.cluster.DBSCAN,
    cluster_folder: str,
    min_points_per_cluster: int,
) -> None:
    # load submap
    submap = Submap.from_pickle(filename=str(pickle_file))
    # get clusters
    try:
        submap.config.cluster_folder = cluster_folder  # overwrite the cluster folder
        submap.config.min_points_per_cluster = min_points_per_cluster
        submap.cluster(dbscan)
        submap.save_pickle()
        # remove the file in the all-points folder
        # import os
        # os.system(f"rm {submap.config.pickle_dir}/{submap.submap_number:06d}.pkl")
    except ValueError:
        click.echo(
            click.style(
                f"Submap {submap.submap_number} has no points to cluster. Skipping...",
                fg='yellow'))


def generate_clusters(config: YAMLConfig) -> None:

    click.echo(
        click.style(f"Clustering points to {config.cluster_folder}",
                    fg='blue',
                    bold=True))

    # define DBSCAN
    dbscan = sklearn.cluster.DBSCAN(
        eps=config.max_dist_between_points_in_cluster,
        min_samples=config.min_number_of_neighbours,
        metric='euclidean',
        n_jobs=-1)

    # list all pickle files
    pickle_files = sorted(Path(config.submap.pickle_dir).rglob("*.pkl"))

    # cluster
    joblib.Parallel(n_jobs=config.n_workers)(
        joblib.delayed(cluster_submap)(
            pickle_file=pickle_file,
            dbscan=dbscan,
            cluster_folder=config.cluster_folder,
            min_points_per_cluster=config.min_points_per_cluster)
        for pickle_file in tqdm(pickle_files, colour='BLUE'))
