import time
import os
from os import listdir
from loguru import logger

from utils.multiprocess import run_pool_steady_camera_filter
from filters.steady_camera_filter.extract_video_segmens import read_yaml
from filters.steady_camera_filter.extract_video_segmens import extract_and_write_steady_camera_segments
from filters.steady_camera_filter.extract_video_segmens import move_steady_non_steady_videos_to_subfolders, move_videos_by_filename


@logger.catch
def cut_videos(**kwargs):
    videos_source_folder = kwargs['videos_source_folder']
    videos_target_folder = kwargs['videos_target_folder']
    videos_extensions = kwargs['videos_extensions']
    use_multiprocessing = kwargs.get('use_multiprocessing', False)
    number_processes = kwargs.get('number_processes', 4)
    move_to_folders_strategy = kwargs.get('move_to_folders_strategy', 'none')

    video_source_filepaths = [os.path.join(videos_source_folder, f) for f in listdir(videos_source_folder)
                              if os.path.isfile(os.path.join(videos_source_folder, f)) and os.path.splitext(f)[-1] in videos_extensions]
    os.makedirs(videos_target_folder, exist_ok=True)

    steady_camera_filter_parameters = read_yaml('steady_camera_filter_parameters.yaml')
    time_start = time.time()
    if use_multiprocessing:
        run_pool_steady_camera_filter(extract_and_write_steady_camera_segments,
                                      video_source_filepaths,
                                      videos_target_folder,
                                      steady_camera_filter_parameters,
                                      number_processes=number_processes)
    else:
        for video_source_filepath in video_source_filepaths:
            extract_and_write_steady_camera_segments(video_source_filepath, videos_target_folder, steady_camera_filter_parameters)
    time_end = time.time()
    logger.info(f'Filtering time for {len(video_source_filepaths)} videos took {(time_end - time_start):.2f} seconds')

    match move_to_folders_strategy:
        case 'steady_non_steady':
            move_steady_non_steady_videos_to_subfolders(videos_target_folder,
                                                        'steady',
                                                        'nonsteady')
        case 'by_source_filename':
            move_videos_by_filename(videos_source_folder, videos_target_folder)


if __name__ == '__main__':
    videos_root_folder = '/media/anton/4c95a564-35ea-40b5-b747-58d854a622d0/home/anton/work/fitMate/datasets'

    processing_parameters = dict()
    processing_parameters['videos_source_folder'] = os.path.join(videos_root_folder, 'squats_2022_abriged')
    processing_parameters['videos_target_folder'] = os.path.join(videos_root_folder, 'squats_2022_coarse_steady_camera__')
    processing_parameters['move_to_folders_strategy'] = 'steady_non_steady'   # 'by_source_filename'
    processing_parameters['videos_steady_subfolder'] = 'steady'
    processing_parameters['videos_non_steady_subfolder'] = 'non_steady'
    processing_parameters['videos_extensions'] = ['.mp4', '.MP4', '.mkv', '.webm']
    processing_parameters['use_multiprocessing'] = True
    processing_parameters['number_processes'] = 2

    logger.add('cut_videos_by_steady_camera_filter.log', format="{time} {message}", level="DEBUG", retention="11 days", compression='zip')
    cut_videos(**processing_parameters)
