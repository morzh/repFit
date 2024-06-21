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
        self.scale = scale_factor

    def write_segments(self, video_segments: VideoSegments, filter_name: str = 'steady') -> None:
        """
        Description:
            Write video segments as separate video files.
        :param video_segments: video segments
        :param filter_name: name of the filter (prefix to frames range)
        """

        if video_segments.segments.size == 0:
            return

        if self.whole_video_segments_check(video_segments):
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

    def write_segments_gaps(self, video_segments: VideoSegments, filter_name='nonsteady') -> None:
        """
        Description:
            This method calculates segments between given video segments and video frames range.
            It could be used for debugging purposes to write video segments, where camera is not steady.
        :param video_segments: video segments
        :param filter_name: name of the filter of filtering stage (in other words prefix to frames range)
        """
        video_segments_gaps = self.calculate_segments_gaps(video_segments)
        if video_segments_gaps.segments.size == 0:
            return
        self.write_segments(video_segments_gaps, filter_name)

    def write(self, video_segments: VideoSegments, write_gaps: bool = False) -> None:
        """
        Description:
            Write video segments.
        :param video_segments: video segments
        :param write_gaps: write video segments and gaps between segments
        """

        if write_gaps:
            self.write_segments_gaps(video_segments)

        self.write_segments(video_segments)

    @staticmethod
    def whole_video_segments_check(video_segments: VideoSegments) -> bool:
        """
        Description:
            Checks if video_segments has only one segment with frame start equals zero and frame end equals frames number - 1
        :return: True if there is only one whole range video segment, False otherwise
        """
        return (video_segments.segments.shape[0] == 1 and
                video_segments.segments[0, 0] == 0 and
                video_segments.segments[-1, -1] == video_segments.frames_number - 1)

    @staticmethod
    def calculate_segments_gaps(video_segments: VideoSegments) -> VideoSegments:
        r"""
        Description:
            Video segments complement set closure, where set is a  :math:`[0, N_{f} - 1]` segment. Formula:

            .. math::
                \mathbf{C} \Big \{ [0, N_f - 1]  \ \backslash  \  \left ( \cup_{n=1}^{N_s} s_n \right) \Big \}

            where :math:`N_f` -- number of frames, :math:`N_s` -- number of segments, :math:`\{s_n\}` -- segments,
            :math:`\mathbf{C}` -- set closure.
        :param video_segments: video segments information
        :return: inverted video segments
        """
        segments = video_segments.segments.flatten()
        segments = np.insert(segments, 0, 0)
        segments = np.append(segments, video_segments.frames_number - 1)
        segments = segments.reshape(-1, 2)

        if segments[0, 0] == segments[0, 1]:
            segments = np.delete(segments, 0, axis=0)
        if segments[-1, 0] == segments[-1, 1]:
            segments = np.delete(segments, -1, axis=0)

        video_segments_gaps = copy.copy(video_segments)
        video_segments_gaps.segments = segments

        return video_segments_gaps

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
        filename_frames_range = '_' + str(segment[0]) + '-' + str(segment[1]) + '__'
        video_filename = video_filename_base + '__' + frames_range_prefix + filename_frames_range + '.mp4'
        output_filepath = os.path.join(self.output_folder, video_filename)
        return output_filepath
