import time
import os
from os import listdir
from pathlib import Path

import numpy as np
from loguru import logger


def check_sorting_steady_non_steady(folder_1: str | Path, folder_2: str | Path, **kwargs) -> None:
    if not kwargs.get('folder_sorting', '') == 'steady_non_steady':
        raise NotImplementedError('Folder organization other than steady_non_steady is not implemented')




def load_segments_data(folder_1: str | Path, folder_2: str | Path) -> tuple[dict, dict]:
    folder_1 = Path(folder_1)
    folder_2 = Path(folder_2)

    segments_filepaths_1 = folder_1.glob('*.npy')
    segments_filepaths_2 = folder_2.glob('*.npy')

    segments_1 = {}
    for segments_filepath in segments_filepaths_1:
        segments_1[segments_filepath.name] = np.load(segments_filepath)

    segments_2 = {}
    for segments_filepath in segments_filepaths_2:
        segments_2[segments_filepath.name] = np.load(segments_filepath)

    return segments_1, segments_2


def segments_compare_visualization(folder_1: str | Path, folder_2: str | Path, **kwargs):
    check_sorting_steady_non_steady(folder_1, folder_2, kwargs)

    steady_segments_1, steady_segments_2 = load_segments_data(folder_1, folder_2)


def segments_compare_statistics(folder_1: str | Path, folder_2: str | Path, sort_by_folders: str):
    if not sort_by_folders == 'steady_non_steady':
        raise NotImplementedError('Folder organization other than steady_non_steady is not implemented')

    folder_1 = Path(folder_1)
    folder_2 = Path(folder_2)


videos_segments_folder_1 = ''
videos_segments_folder_2 = ''

folder_organization_parameters: dict[str, str] = {'folder_sorting': 'steady_non_steady', 'steady_folder': 'steady', 'non_steady_folder': 'non_steady'}

segments_compare_visualization(videos_segments_folder_1, videos_segments_folder_2, **folder_organization_parameters)
segments_compare_statistics(videos_segments_folder_1, videos_segments_folder_2, **folder_organization_parameters)
