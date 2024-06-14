import os.path
import warnings
import numpy as np
import cv2
import yaml

from typing import Annotated, Literal, TypeVar, Optional
from numpy.typing import NDArray

from filters.steady_camera_filter.core.steady_camera_coarse_filter import SteadyCameraCoarseFilter
from cv_utils.video_segments_writer import VideoSegmentsWriter

segments_list = Annotated[NDArray[np.int32], Literal["N", 2]]


def yaml_parameters(filepath: str) -> Optional[dict]:
    parameters = None
    with open('steady_camera_filter_parameters.yaml') as f:
        try:
            parameters = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    return parameters


def extract_coarse_steady_camera_video_segments(video_filepath: str, parameters: dict) -> segments_list:
    if parameters['verbose_filename']:
        video_filename = os.path.basename(video_filepath)
        print(video_filename)

    number_frames_to_average = parameters['number_frames_to_average']
    if number_frames_to_average < 5:
        warnings.warn(f'Value {number_frames_to_average} of number_frames_to_average is low, results could be non applicable')

    camera_filter = SteadyCameraCoarseFilter(video_filepath,
                                             number_frames_to_average=number_frames_to_average,
                                             poc_maximum_dimension=parameters['poc_maximum_dimension'],
                                             minimum_ocr_confidence=parameters['minimum_ocr_confidence'],
                                             maximum_shift_length=parameters['maximum_shift_length'],
                                             poc_minimum_confidence=parameters['poc_minimum_confidence'])
    camera_filter.process(parameters['poc_show_averaged_frames_couple'])
    steady_segments = camera_filter.calculate_steady_camera_ranges()
    steady_segments = camera_filter.filter_segments_by_time(steady_segments, parameters['minimum_steady_camera_time_segment'])

    if parameters['poc_registration_verbose']:
        camera_filter.print_registration_results()
    if parameters['verbose_segments']:
        print(steady_segments)

    return steady_segments


def write_video_segments(video_filepath, output_folder, segments, parameters: dict):
    if not os.path.exists(video_filepath):
        raise FileNotFoundError(f'File {video_filepath} does not exist')

    video_capture = cv2.VideoCapture(str(video_filepath))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_segments_writer = VideoSegmentsWriter(input_filepath=video_filepath,
                                                output_folder=output_folder,
                                                fps=fps,
                                                width=width,
                                                height=height,
                                                scale=parameters['scale_factor'])
    video_segments_writer.write(segments, use_gaps=parameters['use_segments_gaps'])
