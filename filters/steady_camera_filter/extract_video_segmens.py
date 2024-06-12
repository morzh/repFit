import os.path
import warnings
import numpy as np
import cv2

from typing import Annotated, Literal, TypeVar, Optional
from numpy.typing import NDArray

from filters.steady_camera_filter.core.steady_camera_coarse_filter import SteadyCameraCoarseFilter
from cv_utils.video_segments_writer import VideoSegmentsWriter

segments_list = Annotated[NDArray[np.int32], Literal["N", 2]]


def extract_coarse_steady_camera_video_segments(video_filepath: str, parameters: dict) -> segments_list:
    number_frames_to_average = parameters['number_frames_to_average']
    if number_frames_to_average < 5:
        warnings.warn(f'Value {number_frames_to_average} of number_frames_to_average is low, results could be non applicable')

    camera_filter = SteadyCameraCoarseFilter(video_filepath,
                                             number_frames_to_average=number_frames_to_average,
                                             poc_maximum_dimension=parameters['poc_maximum_dimension'],
                                             minimum_ocr_confidence=parameters['minimum_ocr_confidence'],
                                             maximum_shift_length=parameters['maximum_shift_length'],
                                             minimum_poc_confidence=parameters['minimum_poc_confidence'])
    camera_filter.process()
    return camera_filter.calculate_steady_camera_ranges()


def write_video_segments(video_filepath, output_folder, segments, scale_factor=0.5, use_segments_gaps=False):
    if not os.path.exists(video_filepath):
        return

    video_capture = cv2.VideoCapture(str(video_filepath))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_segments_writer = VideoSegmentsWriter(input_filepath=video_filepath,
                                                output_folder=output_folder,
                                                fps=fps,
                                                width=width,
                                                height=height,
                                                scale=scale_factor)
    video_segments_writer.write(segments, write_segments_inbetween=use_segments_gaps)
