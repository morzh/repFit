import shutil

import numpy as np
from loguru import logger
import os
import pickle
import time

from core.filters.single_person.core.filter_addons.confidence_filter_addon import ConfidenceFilterAddon
from core.filters.single_person.core.filter_addons.partial_person_filter_addon import PartialPersonFilterAddon
from core.filters.single_person.core.filter_addons.whole_person_filter_addon import WholePersonFilterAddon
from core.filters.single_person.core.filter_addons.absolute_area_filter_addon import AbsoluteAreaFilterAddon
from core.filters.single_person.core.filter_addons.area_ratio_filter_addon import AreaRatioFilterAddon
from core.filters.single_person.core.filter_addons.bridge_gaps_filter_addon import BridgeGapsFilterAddon
from core.filters.single_person.core.filter_addons.segments_duration_filter_addon import SegmentsDurationFilterAddon

from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks
from core.filters.single_person.core.multiple_persons_tracker import PersonsTracker

from core.utils.geometry.bounding_boxes.bounding_box_2d import BoundingBox2D
from core.utils.parallel.multiprocess import run_pool_single_persons_filter
from core.utils.io.files_operations import check_filename_entry_in_folder, extract_name_extension_from_filepath
from core.utils.cv.video_tools import video_resolution_check,  VideoWriter


@logger.catch
def process_videos_by_single_persons_filter(input_output_config: dict, filter_parameters: dict) -> None:
    """
    Description:
        Filter video by single persons filter. Output of this filter is the set of video segments, each of which with a bounding box.

    :param input_output_config: input output folders configuration
    :param filter_parameters: steady camera filter parameters

    :raises ValueError:
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
        run_pool_single_persons_filter(process_video,
                                       video_source_filepaths,
                                       videos_target_folder,
                                       number_processes=number_processes,
                                       **filter_parameters)
    else:
        for video_source_filepath in video_source_filepaths:
            process_video(video_source_filepath, videos_target_folder, **filter_parameters)
    time_end = time.time()

    logger.info(f'Filtering time for {len(video_source_filepaths)} videos took {(time_end - time_start):.2f} seconds')


def  process_video(video_source_filepath: os.PathLike | str, videos_target_folder: os.PathLike | str, **parameters) -> None:
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

    video_input_parameters = parameters['video_input']
    tracking_parameters = parameters['tracking']
    filtering_parameters = parameters['filtering']
    visualization_parameters = parameters['visualization']
    video_segments_writer_parameters = parameters['video_segments_writer']

    do_filtering = filtering_parameters.get('do_filtering', False)
    do_visualization = visualization_parameters['do_visualization']
    write_tracks_to_videos = video_segments_writer_parameters.get('write_persons_tracks', False)

    video_processing_start_time = time.time()
    minimum_resolution = video_input_parameters.get('minimal_resolution', 200)
    video_filename = os.path.basename(video_source_filepath)
    if not video_resolution_check(video_source_filepath, minimum_dimension_size=minimum_resolution):
        logger.info(f"{video_filename} :: one of the resolution dimension has size less than {minimum_resolution} pixels")
        return

    tracks = obtain_multiple_persons_tracks(video_source_filepath, **tracking_parameters)

    if do_filtering:
        filter_multiple_persons_tracks(tracks, **filtering_parameters)
    if write_tracks_to_videos:
        write_multiple_persons_tracks(video_source_filepath, videos_target_folder, tracks, **parameters['video_segments_writer'])

    video_processing_end_time = time.time()
    logger.info(f'{video_filename} :: processing took {(video_processing_end_time - video_processing_start_time):.2f} seconds, '
                f'video duration is {(tracks.video_properties.approximate_frames_number / tracks.video_properties.fps):.2f} seconds.')

    if do_visualization:
        tracks.visualize(**visualization_parameters)


def filter_multiple_persons_tracks(tracks: MultiplePersonsTracks, **parameters) -> MultiplePersonsTracks:
    """
    Description:
        Filter persons track by predefined set of filters.

    :param tracks: persons tracks

    :keyword confidence:
    :keyword absolute_area:
    :keyword area_ratio:
    :keyword partial_person:
    :keyword segments_duration:
    :keyword bridging_gaps:

    :return: filtered tracks
    """
    if parameters['confidence']['apply']:
        confidence_filter_addon = ConfidenceFilterAddon(parameters['confidence']['confidence_threshold'])
        tracks.apply_filter(confidence_filter_addon)

    if parameters['absolute_area']['apply']:
        area_filter_addon = AbsoluteAreaFilterAddon(parameters['absolute_area']['area_threshold'])
        tracks.apply_filter(area_filter_addon)

    if parameters['area_ratio']['apply']:
        area_ratio_filter_addon = AreaRatioFilterAddon(parameters['area_ratio']['ratio_threshold'])
        tracks.apply_filter(area_ratio_filter_addon)

    if parameters['partial_person']['apply']:
        partial_person_filter_addon = PartialPersonFilterAddon(**parameters['partial_person'])
        tracks.apply_filter(partial_person_filter_addon)

    if parameters['segments_duration']['apply']:
        duration_filter_addon = SegmentsDurationFilterAddon(parameters['segments_duration']['duration_threshold'])
        tracks.apply_filter(duration_filter_addon)

    if parameters['bridging_gaps']['apply']:
        bridge_gaps_filter_addon = BridgeGapsFilterAddon(parameters['bridging_gaps']['gap_threshold'])
        tracks.apply_filter(bridge_gaps_filter_addon)

    if parameters['whole_person']['apply']:
        whole_person_filter_addon = WholePersonFilterAddon(**parameters['whole_person'])
        tracks.apply_filter(whole_person_filter_addon)

    return tracks


def obtain_multiple_persons_tracks(video_source_filepath, **parameters) -> MultiplePersonsTracks:
    """
     Description:

     :param video_source_filepath:

     :keyword yolo_weights_path:
     :keyword yolo_weights_path:

     :raises ValueError:

     :return: multiple persons track
    """
    yolo_weights_folder = parameters.get('yolo_weights_path', None)
    yolo_model = parameters.get('yolo_model', 'yolov10l.pt')
    write_tracked_data = parameters.get('write_tracked_data', False)
    tracked_data_suffix = parameters.get('tracked_data_suffix', True)
    use_saved_data = parameters.get('use_saved_data', True)

    yolo_weights_filepath = os.path.join(str(yolo_weights_folder), str(yolo_model))
    persons_track_data_filepath = f'{video_source_filepath}.{tracked_data_suffix}.pickle'
    persons_tracker = PersonsTracker(weights_pathname=yolo_weights_filepath)
    track_persons = False

    if not use_saved_data:
        track_persons = True
    elif use_saved_data and os.path.exists(persons_track_data_filepath):
        try:
            with open(persons_track_data_filepath, "rb") as input_file:
                tracks = pickle.load(input_file)
            track_persons = False
        except ModuleNotFoundError:
            track_persons = True

    if track_persons:
        tracks = persons_tracker.track(video_source_filepath, **parameters)
        if write_tracked_data:
            tracks.serialize(persons_track_data_filepath)

    return tracks


def write_multiple_persons_tracks(source_filepath: os.PathLike | str, target_folder: os.PathLike | str, tracks: MultiplePersonsTracks, **parameters) -> None:
    """
    Description:
        Write video segments with bounding boxes.

    :param source_filepath: source video filepath
    :param target_folder: target folder to write videos to
    :param tracks: multiple persons tracks

    :key key1: asdasdas
    """
    output_video_suffix = parameters.get('video_suffix', 'single_person')
    segment_duration_threshold = parameters.get('whole_person_duration_threshold', 1)
    segments_gap_threshold = parameters.get('whole_person_gap_duration', 3)

    # input_video_bounding_box = BoundingBox2D(0, 0, tracks.video_properties.width - 1, tracks.video_properties.height - 1)

    for person_id, person_track in tracks.persons.items():
        # current_other_persons_ids = [p_id for p_id in tracks.persons.keys() if p_id != person_id]
        current_whole_person_segments = person_track.calculate_whole_person_segments(tracks.frames_stride)

        current_gap_length_threshold = round(segments_gap_threshold * tracks.video_properties.fps)
        current_whole_person_segments.bridge_gaps(current_gap_length_threshold)

        current_segment_length_threshold = round(segment_duration_threshold * tracks.video_properties.fps)
        current_whole_person_segments.filter_by_length(current_segment_length_threshold)

        if not current_whole_person_segments.size:
            continue

        current_whole_person_boxes = person_track.bounding_boxes_per_segment(current_whole_person_segments)
        video_filename_base, video_filename_extension = extract_name_extension_from_filepath(tracks.video_properties.filepath)
        current_output_video_file_basename = f'{video_filename_base}__{output_video_suffix}-id{person_id}'

        if person_track.is_track_equals_video(tracks.video_properties, tracks.frames_number):
            output_filepath = os.path.join(target_folder, current_output_video_file_basename.join(['.', video_filename_extension]))
            shutil.copy(source_filepath, output_filepath)
            continue

        video_writer = VideoWriter(source_filepath, target_folder, fps=tracks.video_properties.fps)
        video_writer.write_segments_with_bounding_boxes(current_whole_person_segments, current_whole_person_boxes, current_output_video_file_basename)