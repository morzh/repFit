import time
import os
from os import listdir
from pathlib import Path

import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib import collections as mc


def check_sorting_steady_non_steady(folder_1: str | Path, folder_2: str | Path, **kwargs) -> None:
    if not kwargs.get('folder_sorting', '') == 'steady_non_steady':
        raise NotImplementedError('Folder organization other than steady_non_steady is not implemented')

    steady_subfolder = kwargs['steady_folder']
    non_steady_subfolder = kwargs['non_steady_folder']

    steady_path_1 = os.path.join(folder_1, steady_subfolder)
    non_steady_path_1 = os.path.join(folder_2, non_steady_subfolder)

    if not os.path.exists(steady_path_1):
        pass


def load_videos_segments_data(folder_path: str | Path) -> dict:
    folder_path = Path(folder_path)
    segments_filepaths = folder_path.glob('*.npy')
    segments = {}
    for segments_filepath in segments_filepaths:
        segments[segments_filepath.name] = np.load(segments_filepath)
    return segments


def overlap_segments_data(segments_1: dict, segments_2: dict) -> tuple[dict, dict]:
    common_keys = set(segments_1).intersection(segments_2)  # O(N * log N) operation

    segments_new_1 = {}
    segments_new_2 = {}
    for key in common_keys:
        segments_new_1[key] = segments_1[key]
        segments_new_2[key] = segments_2[key]

    return segments_new_1, segments_new_2


def segments_compare_visualization(folder_path_1: str | Path, folder_path_2: str | Path):
    # check_sorting_steady_non_steady(folder_1, folder_2, kwargs)
    segments_1 = load_videos_segments_data(folder_path_1)
    segments_2 = load_videos_segments_data(folder_path_2)

    segments_1, segments_2 = overlap_segments_data(segments_1, segments_2)
    assert len(segments_1) == len(segments_2)

    segments_color_1 = 'g'
    segments_color_2 = 'b'

    plt.figure(figsize=(20, 25))
    for segment_index, segment in enumerate(segments_1):
        plt.plot(segment, [segment_index, segment_index], c=segments_color_1, thickness=7)
        plt.scatter(segment, [segment_index, segment_index], color=segments_color_1, thickness=7)
    for segment_index, segment in enumerate(segments_2):
        plt.plot(segment, [segment_index, segment_index], c=segments_color_2, thickness=3)
        plt.scatter(segment, [segment_index, segment_index], color=segments_color_2, thickness=7)
    plt.show()


def segments_compare_statistics(folder_1: str | Path, folder_2: str | Path):
    folder_1 = Path(folder_1)
    folder_2 = Path(folder_2)


root_folder = '/media/anton/4c95a564-35ea-40b5-b747-58d854a622d0/home/anton/work/fitMate/datasets/'
filter_subfolder = 'steady'
videos_segments_folder_1 = os.path.join(root_folder, 'squats_2022_coarse_steady_camera_yolo_segmentation-m', filter_subfolder)
videos_segments_folder_2 = os.path.join(root_folder, 'squats_2022_coarse_steady_camera_yolo_detector-m', filter_subfolder)

segments_compare_visualization(videos_segments_folder_1, videos_segments_folder_1)
# segments_compare_statistics(videos_segments_folder_1, videos_segments_folder_2)
