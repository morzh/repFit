import copy
from loguru import logger
import numpy as np
import os
import pickle
import shutil
import time

from core.filters.single_person.source.filter_addons.bounding_boxes_iou_filter_addon import BoundingBoxesIouFilterAddon
from core.filters.single_person.source.filter_addons.confidence_filter_addon import ConfidenceFilterAddon
from core.filters.single_person.source.filter_addons.copy_symmetric_occluded_joints_yolo_filter_addon import CopySymmetricOccludedJointsYoloFilterAddon
# from core.filters.single_person.source.filter_addons.partial_person_filter_addon import PartialPersonFilterAddon
from core.filters.single_person.source.filter_addons.joints_filter_addon import JointsFilterAddon
from core.filters.single_person.source.filter_addons.absolute_area_filter_addon import AbsoluteAreaFilterAddon
from core.filters.single_person.source.filter_addons.area_ratio_filter_addon import AreaRatioFilterAddon
from core.filters.single_person.source.filter_addons.bridge_gaps_filter_addon import BridgeGapsFilterAddon
from core.filters.single_person.source.filter_addons.mean_person_confidence_filter_addon import MeanPersonConfidenceFilterAddon
from core.filters.single_person.source.filter_addons.segments_duration_filter_addon import SegmentsDurationFilterAddon

from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks
from core.filters.single_person.source.multiple_persons_tracker import PersonsTracker

from core.utils.geometry.bounding_boxes.bounding_box_2d import BoundingBox2D
from core.utils.parallel.multiprocess import run_pool_single_persons_filter
from core.utils.io.files_operations import extract_name_extension_from_filepath
from core.utils.cv.video_tools import video_resolution_check,  VideoWriter


@logger.catch
def process_videos_by_single_persons_filter(input_output_config: dict, filter_parameters: dict) -> None:
    """
    Description:
        Filter video by single persons filter. Output of this filter is the set of video segments, each of which with a bounding box.

    :param input_output_config: input output folders configuration
    :param filter_parameters: steady camera filter parameters

    :raises ValueError: if ``filter_parameters`` are incorrect.
    """

    videos_root_folder = str(input_output_config.get('videos_root_folder', None))
    videos_source_subfolder = str(input_output_config.get('videos_source_subfolder', None))
    videos_target_subfolder = str(input_output_config.get('videos_target_subfolder', None))

    if videos_root_folder is None: raise ValueError('input_output_config dictionary should contain videos_root_folder key argument')
    if videos_source_subfolder is None: raise ValueError('input_output_config dictionary should contain videos_source_subfolder key argument')
    if videos_target_subfolder is None: raise ValueError('input_output_config dictionary should contain videos_target_subfolder key argument')

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
        run_pool_single_persons_filter(process_video, video_source_filepaths, videos_target_folder, number_processes=number_processes, **filter_parameters)
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
    video_input_parameters = parameters['video_input']
    tracking_parameters = parameters['tracking']
    visualization_parameters = parameters['visualization']
    video_segments_writer_parameters = parameters['video_segments_writer']
    do_visualization = visualization_parameters['do_visualization']

    video_processing_start_time = time.time()
    minimum_resolution = video_input_parameters.get('minimal_resolution', 200)
    video_filename = os.path.basename(video_source_filepath)
    if not video_resolution_check(video_source_filepath, minimum_dimension_size=minimum_resolution):
        logger.info(f"{video_filename} :: one of the resolution dimension has size less than {minimum_resolution} pixels")
        return

    tracks = obtain_multiple_persons_tracks(video_source_filepath, **tracking_parameters)
    filter_tracks(tracks, **parameters)

    if video_segments_writer_parameters.get('write_persons_tracks', False):
        write_multiple_persons_tracks(video_source_filepath, videos_target_folder, tracks, **parameters['video_segments_writer'])

    video_processing_end_time = time.time()
    logger.info(f'{video_filename} :: processing took {(video_processing_end_time - video_processing_start_time):.2f} seconds, '
                f'video duration is {(tracks.video_properties.approximate_frames_number / tracks.video_properties.fps):.2f} seconds.')

    if do_visualization:
        tracks.visualize_tracked_data(**visualization_parameters)


def obtain_multiple_persons_tracks(video_source_filepath, **parameters) -> MultiplePersonsTracks:
    """
     Description:
        Get multiple persons tracks from video file or from serialized data file.

     :param video_source_filepath: video filepath

     :keyword yolo_weights_path: YOLO weights filepath;
     :keyword yolo_model: YOLO model filename (e.g. yolov10l.pt);
     :keyword write_tracked_data: serialize tracked data;
     :keyword tracked_data_suffix: tracked data suffix;
     :keyword use_saved_data: do not track data and use serialized data instead.

     :raises ValueError:

     :return: multiple persons track
    """
    yolo_weights_folder = parameters.get('yolo_weights_path', None)
    yolo_model = parameters.get('yolo_model', 'yolov10l.pt')
    write_tracked_data = parameters.get('write_tracked_data', False)
    tracked_data_suffix = parameters.get('tracked_data_suffix', True)
    use_saved_data = parameters.get('use_saved_data', True)

    yolo_weights_filepath = os.path.join(str(yolo_weights_folder), str(yolo_model))
    multiple_persons_tracks_pickle_filepath = f'{video_source_filepath}.{tracked_data_suffix}.pickle'
    persons_tracker = PersonsTracker(weights_pathname=yolo_weights_filepath)
    track_persons = True

    if use_saved_data and os.path.exists(multiple_persons_tracks_pickle_filepath):
        try:
            with open(multiple_persons_tracks_pickle_filepath, "rb") as input_file:
                tracks = pickle.load(input_file)
            track_persons = False
            logger.info(f'Data has been read from {input_file}')
        except ModuleNotFoundError as e:
            logger.warning(e.msg)

    if track_persons:
        tracks = persons_tracker.track(video_source_filepath, **parameters)
        if write_tracked_data:
            tracks.serialize(multiple_persons_tracks_pickle_filepath)

    return tracks


def filter_tracks(tracks: MultiplePersonsTracks, **parameters) -> None:
    """
    Description:
        Filter in place person's tracks with predefined set of filters.

    :param tracks: multiple persons tracks;

    :keyword tracked_data_filtering: parameters for tracked data filtering
    :keyword partial_body_data_filtering: parameters for partial body filtering
    :keyword full_body_data_filtering: parameters for full body filtering
    :keyword inter_persons_filtering: parameters for inter persons filtering
    :keyword segments_filtering: parameters for partial and full body time segments filtering
    """
    if parameters['do_filtering']:
        tracked_data_filtering_parameters = parameters['tracked_data_filtering']
        partial_body_filtering_parameters = parameters['partial_body_filtering']
        full_body_filtering_parameters = parameters['full_body_filtering']
        inter_persons_filtering_parameters = parameters['inter_persons_filtering']
        segments_filtering_parameters = parameters['segments_filtering']

        tracks.clear_filtering_chain()

        filter_tracked_data(tracks, **tracked_data_filtering_parameters)
        filter_partial_body_data(tracks, **partial_body_filtering_parameters)
        filter_full_body_data(tracks, **full_body_filtering_parameters)
        filter_inter_persons_data(tracks, **inter_persons_filtering_parameters)
        filter_segments(tracks, **segments_filtering_parameters)


def filter_tracked_data(tracks: MultiplePersonsTracks, **parameters) -> None:
    """
    Description:
        Filter tracked person's data in multiple persons track by predefined set of filters.

    :param tracks: multiple persons tracks;

    :keyword absolute_area: absolute area filter parameters
    :keyword area_ratio:  area ratio filter parameters
    """

    if parameters['mean_confidence']['apply']:
        mean_confidence_threshold = parameters['mean_confidence']['mean_confidence_threshold']
        mean_confidence_filter_addon = MeanPersonConfidenceFilterAddon(mean_confidence_threshold)
        tracks.apply_filter(mean_confidence_filter_addon)

    if parameters['absolute_area']['apply']:
        area_filter_addon = AbsoluteAreaFilterAddon(parameters['absolute_area']['area_threshold'])
        tracks.apply_filter(area_filter_addon)

    if parameters['area_ratio']['apply']:
        area_ratio_threshold = parameters['area_ratio']['ratio_threshold']
        mean_confidence_threshold = parameters['area_ratio']['mean_confidence_threshold']
        area_ratio_filter_addon = AreaRatioFilterAddon(area_ratio_threshold, mean_confidence_threshold)
        tracks.apply_filter(area_ratio_filter_addon)

    if parameters['copy_symmetric_occluded_joints']['apply']:
        add_noise = parameters['copy_symmetric_occluded_joints']['add_noise']
        copy_symmetric_joints = CopySymmetricOccludedJointsYoloFilterAddon(add_noise)
        tracks.apply_filter(copy_symmetric_joints)


def filter_partial_body_data(tracks: MultiplePersonsTracks, **parameters) -> None:
    """
    Description:
        Filter in place partial persons data in multiple persons track by predefined set of filters.

    :param tracks: multiple persons tracks;

    :keyword confidence: confidence filter parameters;
    :keyword joints: partial person filter parameters;
    """
    if parameters['joints']['apply']:
        joints_filter_addon = JointsFilterAddon(**parameters['joints'])
        tracks.apply_filter(joints_filter_addon)

    if parameters['confidence']['apply']:
        confidence_filter_addon = ConfidenceFilterAddon(parameters['confidence']['confidence_threshold'])
        tracks.apply_filter(confidence_filter_addon)


def filter_full_body_data(tracks: MultiplePersonsTracks, **parameters) -> None:
    """
    Description:
        Filter full in place body persons data in multiple persons track by predefined set of filters.

    :param tracks: multiple persons tracks;

    :keyword confidence: partial person confidence filter parameters;
    :keyword joints: partial person filter parameters.
    """
    if parameters['joints']['apply']:
        whole_person_filter_addon = JointsFilterAddon(**parameters['joints'])
        tracks.apply_filter(whole_person_filter_addon)

    if parameters['confidence']['apply']:
        confidence_filter_addon = ConfidenceFilterAddon(parameters['confidence']['confidence_threshold'])
        tracks.apply_filter(confidence_filter_addon)


def filter_inter_persons_data(tracks, **parameters) -> None:
    """
    Description:
        Filter in place persons full or partial body data if it meets some criteria.

    :param tracks: person's tracks.
    """
    if parameters['bounding_boxes_iou']['apply']:
        bounding_box_iou_filter_addon = BoundingBoxesIouFilterAddon(**parameters['bounding_box_iou'])
        tracks.apply_filter(bounding_box_iou_filter_addon)


def filter_segments(tracks, **parameters) -> None:
    """
    Description:
        Filter in place persons full or partial body segments.

    :param tracks: person's tracks.

    :keyword bridging_gaps: bridging gap filter parameters.
    :keyword segments_duration: segments duration filter parameters;
    """
    if parameters['bridging_gaps']['apply']:
        bridge_gaps_filter_addon = BridgeGapsFilterAddon(**parameters['bridging_gaps'], filter_full_body_person=True)
        tracks.apply_filter(bridge_gaps_filter_addon)
        bridge_gaps_filter_addon = BridgeGapsFilterAddon(**parameters['bridging_gaps'], filter_full_body_person=False)
        tracks.apply_filter(bridge_gaps_filter_addon)
    if parameters['segments_duration']['apply']:
        segments_duration_filter_addon = SegmentsDurationFilterAddon(**parameters['segments_duration'], filter_full_body_person=True)
        tracks.apply_filter(segments_duration_filter_addon)
        segments_duration_filter_addon = SegmentsDurationFilterAddon(**parameters['segments_duration'], filter_full_body_person=False)
        tracks.apply_filter(segments_duration_filter_addon)


def write_multiple_persons_tracks(source_filepath: os.PathLike | str, target_folder: os.PathLike | str, tracks: MultiplePersonsTracks, **parameters) -> None:
    """
    Description:
        Write video segments with bounding boxes.

    :param source_filepath: source video filepath
    :param target_folder: target folder to write videos to
    :param tracks: multiple persons tracks

    :key input_video_bounding_box: output video suffix
    """
    output_video_suffix = parameters.get('video_suffix', 'single_person')
    input_video_bounding_box = BoundingBox2D(0, 0, tracks.video_properties.width - 1, tracks.video_properties.height - 1)

    for person_id, person_track in tracks.persons.items():
        if not len(person_track.full_body_data.frames_segments):
            continue

        current_full_body_person_segments = person_track.full_body_data.frames_segments
        current_full_body_person_boxes_per_segment = person_track.bounding_boxes_per_segment(current_full_body_person_segments)
        current_other_persons_segments = tracks.clip_persons_segments(current_full_body_person_segments, person_id)
        current_other_persons_boxes = tracks.persons_bounding_boxes(current_other_persons_segments, person_id)

        current_full_body_person_boxes_per_segment = MultiplePersonsTracks.enlarge_bounding_boxes(current_full_body_person_boxes_per_segment, current_other_persons_boxes, input_video_bounding_box)

        video_filename_base, video_filename_extension = extract_name_extension_from_filepath(tracks.video_properties.filepath)
        current_output_video_file_basename = f'{video_filename_base}__{output_video_suffix}-id{person_id}'

        if person_track.is_track_equals_video(tracks.video_properties, tracks.exact_frames_number):
            output_filepath = os.path.join(target_folder, current_output_video_file_basename.join(['.', video_filename_extension]))
            shutil.copy(source_filepath, output_filepath)
            continue

        video_writer = VideoWriter(source_filepath, target_folder, fps=tracks.video_properties.fps)
        video_writer.write_segments_with_bounding_boxes(current_full_body_person_segments, current_full_body_person_boxes_per_segment, current_output_video_file_basename)
