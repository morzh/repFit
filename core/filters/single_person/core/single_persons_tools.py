from loguru import logger
import os
import time

from core.filters.single_person.core.filters.area_filter_addon import AreaFilterAddon
from core.filters.single_person.core.filters.area_ratio_filter_addon import AreaRatioFilterAddon
from core.filters.single_person.core.filters.beidge_gaps_filter_addon import BridgeGapsFilterAddon
from core.filters.single_person.core.filters.segments_duration_filter_addon import SegmentsDurationFilterAddon
from core.filters.single_person.core.multiple_persons_tracker import PersonsTracker
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks

from core.utils.parallel.multiprocess import run_pool_single_persons_filter
from core.utils.io.files_operations import check_filename_entry_in_folder
from core.utils.cv.video_tools import video_resolution_check


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
    if number_processes > 1:
        run_pool_single_persons_filter(process_single_persons,
                                       video_source_filepaths,
                                       videos_target_folder,
                                       number_processes=number_processes,
                                       **filter_parameters)
    else:
        for video_source_filepath in video_source_filepaths:
            process_single_persons(video_source_filepath, videos_target_folder, **filter_parameters)
    time_end = time.time()

    logger.info(f'Filtering time for {len(video_source_filepaths)} videos took {(time_end - time_start):.2f} seconds')
    # sort_videos_by_criteria(move_to_folders_strategy, videos_source_folder, videos_target_folder)


def  process_single_persons(video_source_filepath: os.PathLike | str, videos_target_folder: os.PathLike | str, **parameters) -> None:
    """
    Description:
        Convenient function for multiprocessing. It violates single responsibility principle, but who cares.

    :param video_source_filepath: source video filepath
    :param videos_target_folder: output folder for segmented videos
    """
    video_source_filename = os.path.basename(video_source_filepath)
    video_source_filename_base = video_source_filename.split('.')[0]
    if check_filename_entry_in_folder(videos_target_folder, video_source_filename_base):
        return

    video_processing_start_time = time.time()
    minimum_resolution = parameters['video_input']['minimal_resolution']
    video_filename = os.path.basename(video_source_filepath)
    if not video_resolution_check(video_source_filepath, minimum_dimension_size=minimum_resolution):
        logger.info(f"{video_filename} :: one of the resolution dimension has size less than {minimum_resolution} pixels")
        return

    video_bbox_segments = extract_single_persons_from_video(video_source_filepath, **parameters['bounding_boxes_segments_extraction'])
    write_video_bbox_segments(video_source_filepath, videos_target_folder, video_bbox_segments, **parameters['video_segments_writer'])
    video_processing_end_time = time.time()
    logger.info(f'{video_filename} :: processing took {(video_processing_end_time - video_processing_start_time):.2f} seconds, '
                f'video duration is {(video_bbox_segments.video_properties.approximate_frames_number / video_bbox_segments.video_properties.fps):.2f} seconds.')


def extract_single_persons_from_video(video_source_filepath, **parameters) -> MultiplePersonsTracks:
    """
     Description:
    """
    yolo_weights_filepath = os.path.join(parameters['yolo_weights_path'], parameters['yolo_model'])
    persons_tracker = PersonsTracker(str(yolo_weights_filepath))

    persons_tracks = persons_tracker.track(video_source_filepath, **parameters['frames_stride'])

    area_filter_addon = AreaFilterAddon(**parameters['person_minimum_area'])
    area_ratio_filter_addon = AreaRatioFilterAddon(**parameters['person_area_ratio'])
    duration_filter_addon = SegmentsDurationFilterAddon(**parameters['person_minimal_time'])
    bridge_gaps_filter_addon = BridgeGapsFilterAddon(**parameters['maximum_gap_time'])

    persons_tracks.apply_filter(area_filter_addon)
    persons_tracks.apply_filter(area_ratio_filter_addon)
    persons_tracks.apply_filter(duration_filter_addon)
    persons_tracks.apply_filter(bridge_gaps_filter_addon)

    return persons_tracks


def write_video_bbox_segments(source_filepath, target_folder, video_bbox_segments, cut_out_strategy, **parameters) -> None:
    """
        Description:
            Write video segments with bounding boxes

    :param source_filepath:
    :param target_folder:
    :param video_bbox_segments:
    :param cut_out_strategy:

    :key key1: asdasdas
    """