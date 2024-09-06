import cv2
from loguru import logger
import os.path
import shutil
import time
import yaml

import filters.steady_camera.core.ocr.ocr_factory as ocr_factory
import filters.steady_camera.core.persons_mask.persons_mask_factory as persons_mask_factory
from filters.steady_camera.core.steady_camera_coarse_filter import SteadyCameraCoarseFilter
from filters.steady_camera.core.video_segments import VideoSegments
from utils.cv.video_segments_writer import VideoSegmentsWriter


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


def read_yaml(yaml_filepath: str) -> dict:
    """
    Description:
        Read yaml file

    :param yaml_filepath: filepath to .yaml file

    :return: dictionary with yaml data
    """
    parameters = None
    if not os.path.exists(yaml_filepath):
        raise FileNotFoundError('Parameters YAML file does not exist')

    with open(yaml_filepath) as f:
        try:
            parameters = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    if parameters is None:
        raise ValueError('Something wrong with the YAML file')

    return parameters


def video_resolution_check(video_filepath: str, minimum_dimension_size: int = 360) -> bool:
    """
    Description:
        Check if video size is greater than a given threshold.

    :param video_filepath: filepath of the video
    :param minimum_dimension_size: minimum(video width, video height) threshold

    :return: True if (width, height) >  minimum_dimension_size, False otherwise
    """
    video_capture = cv2.VideoCapture(video_filepath)
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    maximum_dimension = max(video_width, video_height)

    if maximum_dimension > minimum_dimension_size:
        return True
    return False


def extract_coarse_steady_camera_filter_video_segments(video_filepath: str, **kwargs) -> VideoSegments:
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
    if kwargs['verbose_filename']:
        video_filename = os.path.basename(video_filepath)
        logger.info(f'{video_filename} :: calculating segments.')

    filter_parameters = kwargs['steady_camera_coarse_filter']
    number_frames_to_average = filter_parameters['number_frames_to_average']
    if number_frames_to_average < 5:
        logger.warning(f'Value {number_frames_to_average} of number_frames_to_average is low, results could be non applicable')

    ocr_model = ocr_factory.factory.create(filter_parameters['text_mask_model'], **filter_parameters['text_mask_models'])
    persons_mask_model = persons_mask_factory.factory.create(filter_parameters['persons_mask_model'], **filter_parameters['persons_mask_models'])

    steady_camera_filter = SteadyCameraCoarseFilter(video_filepath, ocr_model, persons_mask_model, **filter_parameters)
    steady_camera_filter.process(filter_parameters['poc_show_averaged_frames_pair'])

    steady_segments = steady_camera_filter.steady_camera_video_segments()
    steady_segments.filter_by_time_duration(kwargs['minimum_steady_camera_time_segment'])

    if kwargs['combine_adjacent_segments']:
        steady_segments.combine_adjacent_segments()
    if filter_parameters['poc_registration_verbose']:
        steady_camera_filter.log_registration_results()
    if filter_parameters['verbose_steady_segments']:
        logger.info(steady_segments)

    return steady_segments


def write_video_segments(video_filepath, output_folder, video_segments: VideoSegments, **kwargs) -> None:
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

    if kwargs['verbose_filename']:
        video_filename = os.path.basename(video_filepath)
        logger.info(f'{video_filename} :: writing video segment(s).')

    video_segments_writer = VideoSegmentsWriter(input_filepath=video_filepath,
                                                output_folder=output_folder,
                                                fps=video_segments.video_fps)

    video_segments_writer.write(video_segments, filter_name='steady')
    if kwargs['save_steady_camera_segments_values'] and video_segments.segments.size > 0:
        video_segments_writer.write_segments_values(video_segments, filter_name='steady')

    if kwargs['write_segments_complement']:
        time_threshold = kwargs['minimum_non_steady_camera_time_segment']
        video_segments_complement = video_segments.complement()
        video_segments_complement.filter_by_time_duration(time_threshold)
        video_segments_writer.write(video_segments_complement, filter_name='nonsteady')

        if kwargs['save_non_steady_camera_segments_values'] and video_segments_complement.segments.size > 0:
            video_segments_writer.write_segments_values(video_segments_complement, filter_name='nonsteady')


def extract_and_write_steady_camera_segments(video_source_filepath, videos_target_folder, **kwargs) -> None:
    """
    Description:
        Convenient function for multiprocessing. It violates single responsibility principle, but who cares.

    :param video_source_filepath: source video filepath
    :param videos_target_folder: output folder for segmented videos

    :return: None
    """
    video_processing_start_time = time.time()
    minimum_resolution = kwargs['video_segments_extraction']['resolution_filter']['minimum_dimension_resolution']
    video_filename = os.path.basename(video_source_filepath)
    if not video_resolution_check(video_source_filepath, minimum_dimension_size=minimum_resolution):
        logger.info(f"{video_filename} :: one of the resolution dimension has size less than {minimum_resolution} pixels")
        return

    video_segments = extract_coarse_steady_camera_filter_video_segments(video_source_filepath, **kwargs['video_segments_extraction'])
    write_video_segments(video_source_filepath, videos_target_folder, video_segments, **kwargs['video_segments_writer'])
    video_processing_end_time = time.time()
    logger.info(f'{video_filename} :: processing took {(video_processing_end_time - video_processing_start_time):.2f} seconds, '
                f'video duration is {(video_segments.frames_number / video_segments.video_fps):.2f} seconds.')


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
        case _:
            raise ValueError("Only 'steady_non_steady' and 'by_source_filename' strategies are supported.")


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
