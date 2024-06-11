import warnings
import numpy as np

from typing import Annotated, Literal, TypeVar, Optional
from numpy.typing import NDArray

from filters.steady_camera_filter.core.steady_camera_coarse_filter import SteadyCameraCoarseFilter
from cv_utils.video_segments_writer import VideoSegmentsWriter

segments_list = Annotated[NDArray[np.int32], Literal["N", 2]]


def extract_coarse_steady_camera_video_segments(video_filepath, number_frames_to_average=15) -> segments_list:
    if number_frames_to_average < 5:
        warnings.warn(f'Value {number_frames_to_average} of number_frames_to_average is low, results could be non applicable')
    camera_filter = SteadyCameraCoarseFilter(video_filepath, number_frames_to_average=number_frames_to_average)
    camera_filter.process()
    return camera_filter.calculate_steady_camera_ranges()


def write_video_segments(video_filepath, segments):
    video_segments_writer = VideoSegmentsWriter(video_filepath)
    video_segments_writer.write(segments)