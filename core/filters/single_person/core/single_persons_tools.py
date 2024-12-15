from loguru import logger
import os
import pickle
import time

from core.filters.single_person.core.filter_addons.confidence_filter_addon import ConfidenceFilterAddon
from core.filters.single_person.core.filter_addons.partial_person_filter_addon import PartialPersonFilterAddon
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks
from core.filters.single_person.core.multiple_persons_tracker import PersonsTracker

from core.filters.single_person.core.filter_addons.absolute_area_filter_addon import AbsoluteAreaFilterAddon
from core.filters.single_person.core.filter_addons.area_ratio_filter_addon import AreaRatioFilterAddon
from core.filters.single_person.core.filter_addons.bridge_gaps_filter_addon import BridgeGapsFilterAddon
from core.filters.single_person.core.filter_addons.segments_duration_filter_addon import SegmentsDurationFilterAddon
from core.utils.cv.video_writer import VideoWriter

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

    do_filtering = filtering_parameters.get('do_filtering', False)
    do_visualization = visualization_parameters['do_visualization']
    write_tracks_to_videos = tracking_parameters.get('write_tracks', False)

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

    if use_saved_data and os.path.exists(persons_track_data_filepath):
        with open(persons_track_data_filepath, "rb") as input_file:
            tracks = pickle.load(input_file)
        return tracks
    else:
        tracks = persons_tracker.track(video_source_filepath, **parameters)
        if write_tracked_data:
            tracks.serialize(persons_track_data_filepath)

    return tracks


def write_multiple_persons_tracks(source_filepath: os.PathLike | str, target_folder: os.PathLike | str, multiple_persons_tracks: MultiplePersonsTracks, **parameters) -> None:
    """
    Description:
        Write video segments with bounding boxes

    :param source_filepath:
    :param target_folder:
    :param multiple_persons_tracks:

    :key key1: asdasdas
    """

    video_writer = VideoWriter(source_filepath, target_folder)
    video_writer.write_multiple_persons_tracks(multiple_persons_tracks)