import time
import os
from os import listdir
from utils.multiprocess import run_pool_steady_camera_filter
from filters.steady_camera_filter.extract_video_segmens import yaml_parameters
from filters.steady_camera_filter.extract_video_segmens import extract_and_write_steady_camera_segments
from filters.steady_camera_filter.extract_video_segmens import move_steady_non_steady_videos_to_subfolders


videos_root_folder = '/media/anton/4c95a564-35ea-40b5-b747-58d854a622d0/home/anton/work/fitMate/datasets'
videos_source_folder = os.path.join(videos_root_folder, 'squats_2022')
videos_target_folder = os.path.join(videos_root_folder, 'squats_2022_coarse_steady_camera')
videos_steady_subfolder = 'steady'
videos_non_steady_subfolder = 'non_steady'
videos_extensions = ['.mp4', '.MP4', '.mkv', '.webm']

use_multiprocessing = True
number_processes = 4

video_source_filepaths = [os.path.join(videos_source_folder, f) for f in listdir(videos_source_folder)
                          if os.path.isfile(os.path.join(videos_source_folder, f)) and os.path.splitext(f)[-1] in videos_extensions]
os.makedirs(videos_target_folder, exist_ok=True)

parameters = yaml_parameters('steady_camera_filter_parameters.yaml')
time_start = time.time()
if use_multiprocessing:
    run_pool_steady_camera_filter(extract_and_write_steady_camera_segments,
                                  video_source_filepaths,
                                  videos_target_folder,
                                  parameters,
                                  n_process=number_processes)
else:
    for video_source_filepath in video_source_filepaths:
        extract_and_write_steady_camera_segments(video_source_filepath, videos_target_folder, parameters)
time_end = time.time()
print(f'Filtering time for {len(video_source_filepaths)} videos took {time_end - time_start} seconds')

move_steady_non_steady_videos_to_subfolders(videos_target_folder,
                                            'steady',
                                            'nonsteady',
                                            videos_steady_subfolder,
                                            videos_non_steady_subfolder)
