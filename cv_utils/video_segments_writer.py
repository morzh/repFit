import copy
import os
import shutil
from pathlib import Path
from typing import Annotated, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from cv_utils.video_reader import VideoReader
from filters.steady_camera_filter.core.video_segments import VideoSegments

segments_list = Annotated[NDArray[np.int32], Literal["N", 2]]


class VideoSegmentsWriter:
    """
    Class for writing video segments to a different video files to a given output folder.
    """
    def __init__(self, input_filepath: str | Path, output_folder: str | Path, fps: float, scale_factor: float = 0.5):
        """
        :param input_filepath: input filepath
        :param output_folder: folder for output videos
        :param fps: FPS for output videos
        :param scale_factor: scale factor for output videos
        """
        self.input_filepath = input_filepath
        self.output_folder = output_folder
        self.fps = fps

    def write(self, video_segments: VideoSegments, filter_name: str = 'steady') -> None:
        """
        Description:
            Write video segments as separate video files.
        :param video_segments: video segments
        :param filter_name: name of the filter (prefix to frames range)
        """

        if video_segments.segments.size == 0:
            return

        if video_segments.whole_video_segments_check():
            video_filename_base, _ = self.extract_filename_base_extension()
            output_filepath = os.path.join(os.path.join(self.output_folder, video_filename_base + '__' + filter_name + '__' + '.mp4'))
            shutil.copy(self.input_filepath, output_filepath)
            return

        video_reader = VideoReader(self.input_filepath, use_tqdm=False)
        resolution = (video_segments.video_width, video_segments.video_height)
        index_segment = 0
        current_segment = video_segments.segments[index_segment]

        for index_frame, frame in enumerate(video_reader):
            if index_frame == current_segment[0]:
                current_output_filepath = self.current_filepath_segment(current_segment, filter_name)
                current_video_writer = cv2.VideoWriter(current_output_filepath, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, resolution)

            if current_segment[0] <= index_frame <= current_segment[1]:
                current_video_writer.write(frame)

            if index_frame == current_segment[1]:
                current_video_writer.release()
                index_segment += 1
                if index_segment == video_segments.segments.shape[0]:
                    return
                current_segment = video_segments.segments[index_segment]

    def extract_filename_base_extension(self) -> tuple[str, str]:
        """
        Description:
            Extract file name without extension and file extension from file pathname.
        :return: file name and file extension
        """
        video_filename = os.path.basename(self.input_filepath)
        return os.path.splitext(video_filename)

    def current_filepath_segment(self, segment: np.ndarray, frames_range_prefix='steady') -> str:
        """
        Description:
            Get video file name for a given segment
        :param segment: video segment (just start and end frame)
        :param frames_range_prefix: frames range prefix
        :return: filename
        """
        video_filename_base, _ = self.extract_filename_base_extension()
        start_frame = str(segment[0])
        end_frame = str(segment[1])
        filename_frames_range = '_' + start_frame + '-' + end_frame + '__'
        video_filename = video_filename_base + '__' + frames_range_prefix + filename_frames_range + '.mp4'
        output_filepath = os.path.join(self.output_folder, video_filename)
        return output_filepath
