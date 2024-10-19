from loguru import logger
import os
import time

from utils.cv.video_writer import VideoWriter
from utils.multiprocess import run_pool_single_persons_filter


def  extract_and_write_single_person_segments(video_source_filepath: os.PathLike | str,
                                              videos_target_folder: os.PathLike | str,
                                              **filter_parameters):
    """

    """

@logger.catch
def process_videos_by_single_persons_filter(input_output_config: dict, filter_parameters: dict) -> None:
    """
    Description:
        Filter video by single persons filter. Output of this filter is the set of video segments, each of which with a bounding box.

    :param input_output_config: input output folders configuration
    :param filter_parameters: steady camera filter parameters
    """

    videos_root_folder = str(input_output_config.get('videos_root_folder', None))
    videos_source_subfolder = str(input_output_config.get('videos_source_subfolder', None))
    videos_target_subfolder = str(input_output_config.get('videos_target_subfolder', None))

    if videos_root_folder is None:
        raise ValueError('input_output_config dictionary should contain videos_root_folder key argument')
    if videos_source_subfolder is None:
        raise ValueError('input_output_config dictionary should contain videos_source_subfolder key argument')
    if videos_target_subfolder is None:
        raise ValueError('input_output_config dictionary should contain videos_target_subfolder key argument')

    videos_source_folder = str(os.path.join(videos_root_folder, videos_source_subfolder))
    videos_target_folder = str(os.path.join(videos_root_folder, videos_target_subfolder))
    videos_extensions = input_output_config.get('videos_extensions', ['.mp4', '.webm', '.mkv'])
    use_multiprocessing = input_output_config.get('use_multiprocessing', False)
    number_processes = input_output_config.get('number_processes', 2)

    video_source_filepaths = [os.path.join(videos_source_folder, f) for f in os.listdir(videos_source_folder)
                              if os.path.isfile(os.path.join(videos_source_folder, f)) and os.path.splitext(f)[-1] in videos_extensions]
    os.makedirs(videos_target_folder, exist_ok=True)

    time_start = time.time()
    if use_multiprocessing:
        run_pool_single_persons_filter(extract_and_write_single_person_segments,
                                      video_source_filepaths,
                                      videos_target_folder,
                                      number_processes=number_processes,
                                      **filter_parameters)
    else:
        for video_source_filepath in video_source_filepaths:
            extract_and_write_single_person_segments(video_source_filepath, videos_target_folder, **filter_parameters)
    time_end = time.time()

    logger.info(f'Filtering time for {len(video_source_filepaths)} videos took {(time_end - time_start):.2f} seconds')
    # sort_videos_by_criteria(move_to_folders_strategy, videos_source_folder, videos_target_folder)