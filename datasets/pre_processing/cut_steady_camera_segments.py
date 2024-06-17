import time
import os
from os import listdir
from os.path import isfile, join, splitext
from pathlib import Path
from utils.multiprocess import run_pool_steady_camera_filter
from filters.steady_camera_filter.extract_video_segmens import yaml_parameters
from filters.steady_camera_filter.extract_video_segmens import extract_write_steady_camera_segments


videos_source_folder = '/media/anton/4c95a564-35ea-40b5-b747-58d854a622d0/home/anton/work/fitMate/datasets/squats_2022'
videos_target_folder = '/media/anton/4c95a564-35ea-40b5-b747-58d854a622d0/home/anton/work/fitMate/datasets/squats_2022_coarse_steady_camera'

videos_extensions = ['.mp4', 'MP4', '.mkv', 'webm']
video_source_filepaths = [join(videos_source_folder, f) for f in listdir(videos_source_folder)
                          if isfile(join(videos_source_folder, f)) and splitext(f)[-1] in videos_extensions]
os.makedirs(videos_target_folder, exist_ok=True)
parameters = yaml_parameters('steady_camera_filter_parameters.yaml')
if parameters is None:
    exit(1)

'''
for video_source_filepath in video_source_filepaths:
    video_source_filename = os.path.basename(video_source_filepath)
    if video_source_filename != 'Squats onthe Cable Core-Xwocbi-tp2o.mkv':
        continue
    extract_write_steady_camera_segments(video_source_filepath)

# mean_empty_slice_videos = []
'''
time_start = time.time()
run_pool_steady_camera_filter(extract_write_steady_camera_segments, video_source_filepaths, videos_target_folder, parameters, n_process=5)
time_end = time.time()
filtering_time = time_end - time_start
print(f'Filtering time for {len(video_source_filepaths)} took {filtering_time} seconds')
