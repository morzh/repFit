import os.path
import warnings
import numpy as np
import cv2
import yaml

from typing import Annotated, Literal, TypeVar, Optional
from numpy.typing import NDArray

from filters.steady_camera_filter.core.ocr.craft import Craft
from filters.steady_camera_filter.core.steady_camera_coarse_filter import SteadyCameraCoarseFilter
from cv_utils.video_segments_writer import VideoSegmentsWriter
from filters.steady_camera_filter.core.video_segments import VideoSegments

segments_list = Annotated[NDArray[np.int32], Literal["N", 2]]


def yaml_parameters(filepath: str) -> Optional[dict]:
    parameters = None
    with open('steady_camera_filter_parameters.yaml') as f:
        try:
            parameters = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    return parameters


def video_resolution_check(video_filepath: str, minimum_dimension_size: int = 360):
    video_capture = cv2.VideoCapture(video_filepath)
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    maximum_dimension = max(video_width, video_height)

    if maximum_dimension > minimum_dimension_size:
        return True
    return False


def extract_coarse_steady_camera_video_segments(video_filepath: str, parameters: dict) -> VideoSegments:
    if parameters['verbose_filename']:
        video_filename = os.path.basename(video_filepath)
        print(video_filename)

    number_frames_to_average = parameters['number_frames_to_average']
    if number_frames_to_average < 5:
        warnings.warn(f'Value {number_frames_to_average} of number_frames_to_average is low, results could be non applicable')

    ocr_engine = Craft()
    camera_filter = SteadyCameraCoarseFilter(video_filepath,
                                             ocr_engine,
                                             number_frames_to_average=number_frames_to_average,
                                             maximum_shift_length=parameters['maximum_shift_length'],
                                             poc_maximum_image_dimension=parameters['poc_maximum_dimension'],
                                             poc_minimum_confidence=parameters['poc_minimum_confidence'])
    camera_filter.process(parameters['poc_show_averaged_frames_pair'])
    steady_segments = camera_filter.calculate_steady_camera_ranges()
    steady_segments = camera_filter.filter_segments_by_time(steady_segments, parameters['minimum_steady_camera_time_segment'])

    if parameters['poc_registration_verbose']:
        camera_filter.print_registration_results()
    if parameters['verbose_segments']:
        print(steady_segments)

    del camera_filter
    return steady_segments


def write_video_segments(video_filepath, output_folder, video_segments: VideoSegments, parameters: dict):
    if not os.path.exists(video_filepath):
        raise FileNotFoundError(f'File {video_filepath} does not exist')

    video_segments_writer = VideoSegmentsWriter(input_filepath=video_filepath,
                                                output_folder=output_folder,
                                                fps=video_segments.video_fps,
                                                width=video_segments.video_width,
                                                height=video_segments.video_height,
                                                scale=parameters['scale_factor'])
    video_segments_writer.write(video_segments, write_method='cv2', use_gaps=parameters['use_segments_gaps'])


def extract_write_steady_camera_segments(video_source_filepath, videos_target_folder, parameters):
    minimum_resolution = parameters['video_segments_extraction']['minimum_dimension_resolution']
    if not video_resolution_check(video_source_filepath, minimum_dimension_size=minimum_resolution):
        return

    video_segments = extract_coarse_steady_camera_video_segments(video_source_filepath, parameters['video_segments_extraction'])
    write_video_segments(video_source_filepath, videos_target_folder, video_segments, parameters['video_segments_output'])
