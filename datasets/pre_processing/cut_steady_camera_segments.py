import os
from os import listdir
from os.path import isfile, join, splitext
from pathlib import Path

from filters.steady_camera_filter.extract_video_segmens import extract_coarse_steady_camera_video_segments, write_video_segments

videos_source_folder = '/media/anton/4c95a564-35ea-40b5-b747-58d854a622d0/home/anton/work/fitMate/datasets/squats_2022'
videos_target_folder = '/media/anton/4c95a564-35ea-40b5-b747-58d854a622d0/home/anton/work/fitMate/datasets/squats_2022_coarse_steady_camera'

videos_extensions = ['.mp4', 'MP4', '.mkv', 'webm']
video_source_filenames = [f for f in listdir(videos_source_folder) if isfile(join(videos_source_folder, f)) and splitext(f)[-1] in videos_extensions]

os.makedirs(videos_target_folder, exist_ok=True)

for video_source_filename in video_source_filenames:
    video_target_filename = os.path.splitext(video_source_filename)[0] + '.mp4'

    video_source_filepath = os.path.join(videos_source_folder, video_source_filename)
    video_target_filepath = os.path.join(videos_target_folder, video_target_filename)

    video_segments = extract_coarse_steady_camera_video_segments(video_source_filepath, number_frames_to_average=20)
    write_video_segments(video_target_filepath, video_segments)
