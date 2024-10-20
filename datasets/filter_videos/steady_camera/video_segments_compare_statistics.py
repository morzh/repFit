import os
from pathlib import Path
import pprint
import numpy as np
import matplotlib.pyplot as plt


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


def get_segments(folder_path_1: str | Path, folder_path_2: str | Path) -> tuple[dict, dict]:
    segments_1 = load_videos_segments_data(folder_path_1)
    segments_2 = load_videos_segments_data(folder_path_2)
    segments_1, segments_2 = overlap_segments_data(segments_1, segments_2)
    assert len(segments_1) == len(segments_2)
    return segments_1, segments_2


def segments_compare_visualization(path_1: str | Path, path_2: str | Path) -> None:
    segments_1, segments_2 = get_segments(path_1, path_2)
    assert len(segments_1) == len(segments_2)

    segments_color_1 = 'r'
    segments_color_2 = 'g'

    for (video_filename, video_segments_1), video_segments_2 in zip(segments_1.items(), segments_2.values()):

        plt.figure(figsize=(20, 25))
        plt.title(os.path.splitext(video_filename)[0])
        for segment_index, segment in enumerate(video_segments_1):
            plt.plot(segment, (0, 0), c=segments_color_1, linewidth=7)
        for segment_index, segment in enumerate(video_segments_2):
            plt.plot(segment, (0, 0), c=segments_color_2, linewidth=3)
        plt.show()


def segments_compare_statistics(path_1: str | Path, path_2: str | Path) -> dict[str, list]:
    segments_1, segments_2 = get_segments(path_1, path_2)
    segments_statistics = {}

    for (video_filename, video_segments_1), video_segments_2 in zip(segments_1.items(), segments_2.values()):
        video_filename_base = os.path.splitext(video_filename)[0]
        if video_segments_1.shape == video_segments_2.shape:
            segments_distances = np.abs(video_segments_1 - video_segments_2)
            segments_distances_mean = np.mean(segments_distances)
            segments_distances_std = np.std(segments_distances)
            segments_statistics[video_filename_base] = [segments_distances_mean, segments_distances_std]
        else:
            segments_statistics[video_filename_base] = [np.nan, np.nan]

    return segments_statistics


if __name__ == '__main__':
    root_folder = '/media/anton/4c95a564-35ea-40b5-b747-58d854a622d0/home/anton/work/fitMate/datasets/'
    filter_subfolder = 'steady'
    videos_segments_folder_1 = os.path.join(root_folder, 'squats_2022_coarse_steady_camera_yolo_segmentation-m', filter_subfolder)
    videos_segments_folder_2 = os.path.join(root_folder, 'squats_2022_coarse_steady_camera_yolo_segmentation-s', filter_subfolder)
    # videos_segments_folder_2 = os.path.join(root_folder, 'squats_2022_coarse_steady_camera_yolo_detector-m', filter_subfolder)

    # segments_compare_visualization(videos_segments_folder_1, videos_segments_folder_1)
    compare_statistics_results = segments_compare_statistics(videos_segments_folder_1, videos_segments_folder_2)

    non_zero_statistics = {key: value for key, value in compare_statistics_results.items() if value[0] != 0 and not np.isnan(value[0])}
    nan_statistics = {key: value for key, value in compare_statistics_results.items() if np.isnan(value[0])}

    print(len(non_zero_statistics))
    pprint.pprint(non_zero_statistics, width=161)

    print(len(nan_statistics))
    pprint.pprint(nan_statistics, width=161)
