import cv2
from loguru import logger
import os
import os.path
import shutil
import time

import filters.steady_camera.core.ocr.ocr_factory as ocr_factory
import filters.steady_camera.core.persons_mask.persons_mask_factory as persons_mask_factory
from filters.steady_camera.core.steady_camera_coarse_filter import SteadyCameraCoarseFilter
from filters.steady_camera.core.video_file_segments import VideoFileSegments
from utils.cv.video_writer import VideoWriter
# from utils.cv.video_segments_writer import VideoSegmentsWriter
from utils.multiprocess import run_pool_steady_camera_filter
from utils.io.files_operations import  check_filename_entry_in_folder
from utils.cv.video_tools import video_resolution_check


class PrintColors:
    """
    Description:
        This class helps with colored text printing. Example of usage: print(f'{PrintColors.BOLD}some text{PrintColors.END}')
    """
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def extract_coarse_steady_camera_filter_video_segments(video_filepath: str, **options) -> VideoFileSegments:
    """
    Description:
        Extract segments from video in frames, where camera is steady (meets steadiness criteria of coarse steady camera filter).

    :param video_filepath: filepath of the video

    :keyword verbose_filename: log video_filepath's filename.
    :keyword steady_camera_coarse_filter: coarse steady camera filter key word arguments.
    :keyword minimum_steady_camera_time_segment: minimum duration of steady camera segment.
    :keyword combine_adjacent_segments: combine two or more adjacent video segments, e.g. adjacent segments are [0, 199] and [200, 599]
    :keyword poc_registration_verbose: log phase only correlation registration results.
    :keyword verbose_steady_segments: log steady segments values.

    :return: video segments for given video
    """
    if options['verbose_filename']:
        video_filename = os.path.basename(video_filepath)
        logger.info(f'{video_filename} :: calculating segments.')

    filter_parameters = options['steady_camera_coarse_filter']
    number_frames_to_average = filter_parameters['number_frames_to_average']
    if number_frames_to_average < 5:
        logger.warning(f'Value {number_frames_to_average} of number_frames_to_average is low, results could be non applicable')

    ocr_model = ocr_factory.factory.create(filter_parameters['text_mask_model'], **filter_parameters['text_mask_models'])
    persons_mask_model = persons_mask_factory.factory.create(filter_parameters['persons_mask_model'], **filter_parameters['persons_mask_models'])

    steady_camera_filter = SteadyCameraCoarseFilter(video_filepath, ocr_model, persons_mask_model, **filter_parameters)
    steady_camera_filter.process(filter_parameters['poc_show_averaged_frames_pair'])

    steady_segments = steady_camera_filter.steady_camera_video_segments()
    steady_segments.filter_by_time(options['minimum_steady_camera_time_segment'])

    if options['combine_adjacent_segments']:
        steady_segments.frames_segments.combine_adjacent_segments()

    if filter_parameters['poc_registration_verbose']:
        steady_camera_filter.log_registration_results()
    if filter_parameters['verbose_steady_segments']:
        logger.info(steady_segments)

    return steady_segments


def write_video_segments(video_filepath, output_folder, video_segments: VideoFileSegments, **options) -> None:
    """
    Description:
        Cuts input video according video_segments information.

    :param video_filepath: input video filepath
    :param output_folder: output folder for trimmed videos
    :param video_segments information about video segments to trim

    :keyword verbose_filename: log video_filepath's filename
    :keyword save_steady_camera_segments_values: save .npy file with steady segments values (for further statistics).
    :keyword write_segments_complement: write segments complements, in other words, non-steady video segments.
    :keyword minimum_non_steady_camera_time_segment: minimum duration of non-steady segment.
    :keyword save_non_steady_camera_segments_values: save .npy file with non-steady segments values (for further statistics).

    :raises FileNotFoundError: when video_filepath does not exist

    :return: None
    """
    if not os.path.exists(video_filepath):
        raise FileNotFoundError(f'File {video_filepath} does not exist')

    if options['verbose_filename']:
        video_filename = os.path.basename(video_filepath)
        logger.info(f'{video_filename} :: writing video segment(s).')

    video_segments_writer = VideoWriter(input_filepath=video_filepath,
                                        output_folder=output_folder,
                                        fps=video_segments.metadata.video_fps)

    video_segments_writer.write_segments(video_segments, filter_name='steady')
    if options['save_steady_camera_segments_values'] and video_segments.frames_segments.size > 0:
        video_segments_writer.write_segments_values(video_segments, filter_name='steady')

    if options['write_segments_complement']:
        time_threshold = options['minimum_non_steady_camera_time_segment']
        video_segments_complement = video_segments.complement()
        video_segments_complement.filter_by_time(time_threshold)
        video_segments_writer.write_segments(video_segments_complement, filter_name='nonsteady')

        if options['save_non_steady_camera_segments_values'] and video_segments_complement.frames_segments.size > 0:
            video_segments_writer.write_segments_values(video_segments_complement, filter_name='nonsteady')


def extract_and_write_steady_camera_segments(video_source_filepath, videos_target_folder, **options) -> None:
    """
    Description:
        Convenient function for multiprocessing. It violates single responsibility principle, but who cares.

    :param video_source_filepath: source video filepath
    :param videos_target_folder: output folder for segmented videos

    :return: None
    """
    video_source_filename = os.path.basename(video_source_filepath)
    video_source_filename_base = video_source_filename.split('.')[0]
    if check_filename_entry_in_folder(videos_target_folder, video_source_filename_base):
        return

    video_processing_start_time = time.time()
    minimum_resolution = options['video_segments_extraction']['resolution_filter']['minimum_dimension_resolution']
    video_filename = os.path.basename(video_source_filepath)
    if not video_resolution_check(video_source_filepath, minimum_dimension_size=minimum_resolution):
        logger.info(f"{video_filename} :: one of the resolution dimension has size less than {minimum_resolution} pixels")
        return

    video_segments = extract_coarse_steady_camera_filter_video_segments(video_source_filepath, **options['video_segments_extraction'])
    write_video_segments(video_source_filepath, videos_target_folder, video_segments, **options['video_segments_writer'])
    video_processing_end_time = time.time()
    logger.info(f'{video_filename} :: processing took {(video_processing_end_time - video_processing_start_time):.2f} seconds, '
                f'video duration is {(video_segments.metadata.frames_number / video_segments.metadata.video_fps):.2f} seconds.')


def sort_videos_by_criteria(move_to_folders_strategy: str, raw_videos_folder: str, filtered_videos_folder: str) -> None:
    """
    Description:
        Move videos to different folders using some strategy.

    :param move_to_folders_strategy: sorting strategy
    :param raw_videos_folder:  folder with raw (unprocessed) videos
    :param filtered_videos_folder: folder with processed (filtered) videos

    :raises ValueError: in case of unknown strategy

    :return: None
    """
    match move_to_folders_strategy:
        case 'steady_non_steady':
            move_steady_non_steady_videos_to_subfolders(filtered_videos_folder, 'steady', 'nonsteady')
        case 'by_source_filename':
            move_videos_by_filename(raw_videos_folder, filtered_videos_folder)
        case 'do_not_sort':
            return
        case _:
            raise ValueError("Only 'steady_non_steady' and 'by_source_filename' sorting strategies are supported.")


def move_videos_by_filename(videos_source_folder: str, processed_videos_folder: str) -> None:
    """
    Description:
        Move processed steady and non-steady videos and (probably) their segments to different folders.

    :param videos_source_folder: folder with source video files;
    :param processed_videos_folder: folder with processed steady and non-steady video files;

    :return: None
    """
    source_filenames = [os.path.basename(f) for f in os.listdir(videos_source_folder) if os.path.isfile(os.path.join(videos_source_folder, f))]
    source_files_basename = [os.path.splitext(f)[0] for f in source_filenames]
    if not len(source_files_basename):
        return

    processed_filenames = [f for f in os.listdir(processed_videos_folder) if os.path.isfile(os.path.join(processed_videos_folder, f))]
    if not len(processed_filenames):
        return

    for source_file_basename in source_files_basename:
        processed_videos_basename_entry = [f for f in processed_filenames if source_file_basename in f]
        for processed_video in processed_videos_basename_entry:
            source_filepath = os.path.join(processed_videos_folder, processed_video)
            target_folder = os.path.join(processed_videos_folder, source_file_basename)
            target_filepath = os.path.join(target_folder, processed_video)
            os.makedirs(target_folder, exist_ok=True)
            shutil.move(source_filepath, target_filepath)


def move_steady_non_steady_videos_to_subfolders(videos_source_folder: str, steady_suffix: str, non_steady_suffix: str) -> None:
    """
    Description:
        Move processed steady and non-steady videos and (probably) their segments to different folders. If filename has steady_entry,
        it will bw moved to subfolder_steady subfolder of the root_folder.
        If filename has non_steady_entry, it will bw moved to subfolder_non_steady subfolder of the root_folder.

    :param videos_source_folder: folder with source video files;
    :param steady_suffix: filename entry which  indicates that video is (considered as) steady. Also, subfolder to move steady file to;
    :param non_steady_suffix: filename entry which that indicates video is (considered as) non-steady. Also, subfolder to move non-steady file to.

    :return: None
    """
    source_filepaths = [os.path.join(videos_source_folder, f) for f in os.listdir(videos_source_folder)
                        if os.path.isfile(os.path.join(videos_source_folder, f))]
    target_steady_folder = os.path.join(videos_source_folder, steady_suffix)
    target_non_steady_folder = os.path.join(videos_source_folder, non_steady_suffix)
    os.makedirs(target_steady_folder, exist_ok=True)
    os.makedirs(target_non_steady_folder, exist_ok=True)

    for source_filepath in source_filepaths:
        filename = os.path.basename(source_filepath)
        if non_steady_suffix in filename:
            target_filepath = str(os.path.join(videos_source_folder, non_steady_suffix, filename))
            shutil.move(source_filepath, target_filepath)
        elif steady_suffix in filename:
            target_filepath = str(os.path.join(videos_source_folder, steady_suffix, filename))
            shutil.move(source_filepath, target_filepath)


@logger.catch
def process_videos_by_steady_camera_filter(config_io: dict, filter_parameters: dict) -> None:
    """
    Description:
        Filter video by steady camera filter. Output of this filter is set of video segments at which camera is steady (within some threshold).

    :param config_io:
    :param filter_parameters:

    :raises ValueError:
    """

    videos_root_folder = str(config_io.get('videos_root_folder', None))
    videos_source_subfolder = str(config_io.get('videos_source_subfolder', None))
    videos_target_subfolder = str(config_io.get('videos_target_subfolder', None))

    if videos_root_folder is None:
        raise ValueError('config_io dictionary should contain videos_root_folder key argument')
    if videos_source_subfolder is None:
        raise ValueError('config_io dictionary should contain videos_source_subfolder key argument')
    if videos_target_subfolder is None:
        raise ValueError('config_io dictionary should contain videos_target_subfolder key argument')

    videos_source_folder = str(os.path.join(videos_root_folder, videos_source_subfolder))
    videos_target_folder = str(os.path.join(videos_root_folder, videos_target_subfolder))

    videos_extensions = config_io.get('videos_extensions', ['.mp4', '.webm', '.mkv'])
    use_multiprocessing = config_io.get('use_multiprocessing', False)
    number_processes = config_io.get('number_processes', 2)
    move_to_folders_strategy = config_io.get('move_to_folders_strategy', 'none')

    video_source_filepaths = [os.path.join(videos_source_folder, f) for f in os.listdir(videos_source_folder)
                              if os.path.isfile(os.path.join(videos_source_folder, f)) and os.path.splitext(f)[-1] in videos_extensions]
    os.makedirs(videos_target_folder, exist_ok=True)


    time_start = time.time()
    if use_multiprocessing:
        run_pool_steady_camera_filter(extract_and_write_steady_camera_segments,
                                      video_source_filepaths,
                                      videos_target_folder,
                                      number_processes=number_processes,
                                      **filter_parameters)
    else:
        for video_source_filepath in video_source_filepaths:
            extract_and_write_steady_camera_segments(video_source_filepath, videos_target_folder, **filter_parameters)
    time_end = time.time()

    logger.info(f'Filtering time for {len(video_source_filepaths)} videos took {(time_end - time_start):.2f} seconds')
    sort_videos_by_criteria(move_to_folders_strategy, videos_source_folder, videos_target_folder)