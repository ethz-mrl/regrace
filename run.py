import packages.data_generation as data_generation
import packages.regrace as regrace
import torch
import torch.backends.cudnn
import click
import pathlib

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)
import random

random.seed(0)
import numpy as np

np.random.seed(0)


@click.command()
@click.option('--config_file',
              '-c',
              default='config/default.yaml',
              help='Path to the configuration file.')
@click.option('--generate_submaps',
              is_flag=True,
              default=False,
              help='Generate submaps from the data.')
def main(
    config_file: str,
    generate_submaps: bool,
):
    if generate_submaps:
        data_generation.main(config_file)
    else:
        click.echo(
            click.style(f">> Generating data using config: {config_file}",
                        bold=True,
                        fg='blue'))

        config_yaml = pathlib.Path(config_file)
        assert config_yaml.exists(), f"File {config_yaml} not found."
        config = regrace.YAMLConfig.from_yaml(config_yaml)

        if config.generate_triplets:
            regrace.generate_data(config)

        if config.train:
            regrace.train(config, config_yaml)

        if config.test:
            regrace.test(config)


if __name__ == '__main__':
    main()
