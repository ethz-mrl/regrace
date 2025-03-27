from ..src.config import YAMLConfig
from ..src.dataset import Dataset


def generate_data(config: YAMLConfig) -> None:
    # generate the dataset from the config
    train_dataset = Dataset.from_config(config, "train")
    Dataset.from_config(config, "test",
                        float(train_dataset.furthest_dist_between_points))


def load_triplets(config: YAMLConfig, split: str = "train") -> Dataset:
    transform = None
    return Dataset.from_pickle(
        pickle_path=config.output_path /
        f"{split}_maxPos{int(config.max_dist2positive)}_minNeg{int(config.min_dist2negative)}.pkl",
        transform=transform)
