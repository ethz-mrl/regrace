import os
from pathlib import Path

import click
import numpy as np
from torch import argmax, device, from_numpy, nn, no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm

from .. import cylinder3d
from ..src.config import YAMLConfig
from .yaml_utils import LABEL_MAPPING_FILE, get_label_inverse_learning_map

CYLINDER3D_PATH = os.path.dirname(cylinder3d.__file__)


def generate_labels(config: YAMLConfig) -> None:

    # set parameters
    data_folder_path = Path(config.single_scans.scans_dir)
    assert data_folder_path.is_dir(
    ), f"Data folder {data_folder_path} does not exist"
    config_yaml_path = Path(f"{CYLINDER3D_PATH}/config/semantickitti.yaml")
    assert config_yaml_path.is_file(
    ), f"Config file {config_yaml_path} does not exist"

    # create output folder
    predicted_label_folder_path = Path(
        config.single_scans.predicted_labels_dir)
    predicted_label_folder_path.mkdir(parents=True, exist_ok=True)
    predicted_prob_label_folder_path = Path(
        config.single_scans.predicted_labels_prob_array_dir)
    predicted_prob_label_folder_path.mkdir(parents=True, exist_ok=True)

    # verbose configurations
    click.echo(
        click.style(f"Generating labels for {data_folder_path}",
                    fg='yellow',
                    bold=True))

    # load config file
    config_dict = cylinder3d.config.config.load_config_data(
        path=str(config_yaml_path))
    dataset_parameters = config_dict['dataset_params']
    model_parameters = config_dict['model_params']
    train_parameters = config_dict['train_params']
    batch_size = 1  # DO NOT CHANGE THIS
    assert batch_size == 1, "[ERROR] Batch size must be 1 for inference"

    # get label names
    cyl2kitti_label_map = get_label_inverse_learning_map()

    # build model
    cylinder3d_model = cylinder3d.builder.model_builder.build(
        model_config=model_parameters)
    assert os.path.exists(
        train_parameters['model_load_path']
    ), f"Cannot find model load path {train_parameters['model_load_path']}"
    cylinder3d_model = cylinder3d.utils.load_save_util.load_checkpoint(
        model_load_path=train_parameters['model_load_path'],
        model=cylinder3d_model,
        device=0)
    cylinder3d_model.to(device('cuda:0'))

    # load dataset
    dataset_pointclouds = cylinder3d.dataloader.pc_dataset.get_pc_model_class(
        'SemKITTI_demo')(
            data_path=data_folder_path,
            imageset='demo',  # set to 'demo' if no labels are available
            return_ref=True,
            label_mapping=LABEL_MAPPING_FILE)
    dataset_loader = DataLoader(
        dataset=cylinder3d.dataloader.dataset_semantickitti.get_model_class(
            name=dataset_parameters['dataset_type'])(
                dataset_pointclouds,
                grid_size=model_parameters['output_shape'],
                fixed_volume_space=dataset_parameters['fixed_volume_space'],
                max_volume_space=dataset_parameters['max_volume_space'],
                min_volume_space=dataset_parameters['min_volume_space'],
                ignore_label=dataset_parameters['ignore_label']),
        batch_size=batch_size,
        collate_fn=cylinder3d.dataloader.dataset_semantickitti.collate_fn_BEV,
        shuffle=False,
        num_workers=int(config.n_workers / 4))

    # inference
    cylinder3d_model.eval()
    with no_grad():
        with tqdm(total=len(dataset_loader), dynamic_ncols=True) as pbar:
            for i, (_, _, grid, _, pt_features) in enumerate(dataset_loader):
                # get data
                pt_features_tensor = [
                    from_numpy(i).float().cuda() for i in pt_features
                ]
                grid_tensor = [from_numpy(i).cuda() for i in grid]
                # predict
                prediction = cylinder3d_model(pt_features_tensor, grid_tensor,
                                              batch_size)
                label_prediction = argmax(prediction,
                                          dim=1).cpu().detach().numpy()
                # get probability labels
                softmax = nn.Softmax(dim=1)
                labels_probabilities = softmax(prediction).cpu().detach(
                ).numpy()[0, :, grid[0][:, 0], grid[0][:, 1],
                          grid[0][:, 2]].astype('float32')
                # save
                label_prediction_array = np.vectorize(
                    cyl2kitti_label_map.__getitem__)(
                        label_prediction[0, grid[0][:, 0], grid[0][:, 1],
                                         grid[0][:, 2]]).astype('uint32')
                label_prediction_array.tofile(
                    f"{config.single_scans.predicted_labels_dir}/{str(i).zfill(6)}.label"
                )
                labels_probabilities.tofile(
                    f"{config.single_scans.predicted_labels_prob_array_dir}/{str(i).zfill(6)}.label"
                )
                # update progress bar
                pbar.update(1)
