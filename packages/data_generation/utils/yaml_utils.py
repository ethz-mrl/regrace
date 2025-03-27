from pathlib import Path

import yaml

LABEL_MAPPING_FILE = Path(
    __file__).resolve().parent / '../cylinder3d/config/label-mapping.yaml'


def get_label_inverse_learning_map() -> dict[int, int]:
    assert Path(LABEL_MAPPING_FILE).exists(
    ), f"Label mapping file {LABEL_MAPPING_FILE} does not exist"
    with open(LABEL_MAPPING_FILE, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    return semkittiyaml['learning_map_inv']


def get_label_color_mapping() -> dict[int, list[int]]:
    assert Path(LABEL_MAPPING_FILE).exists(
    ), f"Label mapping file {LABEL_MAPPING_FILE} does not exist"
    with open(LABEL_MAPPING_FILE, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    return semkittiyaml['color_map']


def get_label_names() -> dict[int, str]:
    assert Path(LABEL_MAPPING_FILE).exists(
    ), f"Label mapping file {LABEL_MAPPING_FILE} does not exist"
    with open(LABEL_MAPPING_FILE, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    return semkittiyaml['labels']


def get_label_learning_map() -> dict[int, int]:
    assert Path(LABEL_MAPPING_FILE).exists(
    ), f"Label mapping file {LABEL_MAPPING_FILE} does not exist"
    with open(LABEL_MAPPING_FILE, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    return semkittiyaml['learning_map']


color_per_label_map = get_label_color_mapping()


def get_rgb_of_label(label: int) -> tuple[float, float, float]:
    # generate color from label id
    if label not in color_per_label_map:
        raise ValueError(f"[ERROR] Unknown label {label}")
    c = color_per_label_map[label]
    return (c[0] / 255, c[1] / 255, c[2] / 255)
