# -------------------
# set seeds
# -------------------
# yapf: disable
import torch
import torch.backends.cudnn

torch.use_deterministic_algorithms(True,warn_only=True)
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)
import random

random.seed(0)
import numpy as np

np.random.seed(0)
from pathlib import Path

# yapf: enable
# -------------------
# start script
# -------------------
import click

from .src.config import YAMLConfig
from .utils.generate_data import generate_data
from .utils.utils import test, train


def main(config_path: str) -> None:
    click.echo(
        click.style(f">> Generating data using config: {config_path}",
                    bold=True,
                    fg='blue'))

    config_yaml = Path(config_path)
    assert config_yaml.exists(), f"File {config_yaml} not found."
    config = YAMLConfig.from_yaml(config_yaml)

    if config.generate_triplets:
        generate_data(config)

    if config.train:
        train(config, config_yaml)

    if config.test:
        test(config)
