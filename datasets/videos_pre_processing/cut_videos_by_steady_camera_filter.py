import time
import os
from os import listdir
from loguru import logger

from utils.multiprocess import run_pool_steady_camera_filter
from filters.steady_camera.extract_video_segmens import read_yaml, extract_and_write_steady_camera_segments, sort_videos_by_criteria


@logger.catch
def cut_videos(**kwargs):
    videos_source_folder = kwargs['videos_source_folder']
    videos_target_folder = kwargs['videos_target_folder']
    videos_extensions = kwargs['videos_extensions']
    use_multiprocessing = kwargs.get('use_multiprocessing', False)
    number_processes = kwargs.get('number_processes', 2)
    move_to_folders_strategy = kwargs.get('move_to_folders_strategy', 'none')

    video_source_filepaths = [os.path.join(videos_source_folder, f) for f in listdir(videos_source_folder)
                              if os.path.isfile(os.path.join(videos_source_folder, f)) and os.path.splitext(f)[-1] in videos_extensions]
    os.makedirs(videos_target_folder, exist_ok=True)

    steady_camera_filter_kwargs = read_yaml('steady_camera_filter_parameters.yaml')
    time_start = time.time()
    if use_multiprocessing:
        run_pool_steady_camera_filter(extract_and_write_steady_camera_segments,
                                      video_source_filepaths,
                                      videos_target_folder,
                                      number_processes=number_processes,
                                      **steady_camera_filter_kwargs
                                      )
    else:
        for video_source_filepath in video_source_filepaths:
            extract_and_write_steady_camera_segments(video_source_filepath, videos_target_folder, **steady_camera_filter_kwargs)
    time_end = time.time()
    logger.info(f'Filtering time for {len(video_source_filepaths)} videos took {(time_end - time_start):.2f} seconds')
    sort_videos_by_criteria(move_to_folders_strategy, videos_source_folder, videos_target_folder)


if __name__ == '__main__':
    videos_root_folder = '/media/anton/4c95a564-35ea-40b5-b747-58d854a622d0/home/anton/work/fitMate/datasets'
    processing_kwargs = {
        'videos_source_folder': os.path.join(videos_root_folder, 'squats_2022'),
        'videos_target_folder': os.path.join(videos_root_folder, 'squats_2022_coarse_steady_camera_yolo_segmentation_yolov9-c'),
        'move_to_folders_strategy': 'steady_non_steady',   # 'by_source_filename'
        'videos_extensions': ['.mp4', '.MP4', '.mkv', '.webm'],
        'use_multiprocessing': True,
        'number_processes': 2
    }
    logger.add('cut_videos_by_steady_camera_filter.log', format="{time} {message}", level="DEBUG", retention="5 days", compression='zip')
    cut_videos(**processing_kwargs)
