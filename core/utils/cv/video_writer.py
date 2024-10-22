import os
import shutil
from pathlib import Path
import cv2
import numpy as np

from typing import Annotated, Literal
from numpy.typing import NDArray

from core.utils.cv.video_reader import VideoReader
from core.utils.cv.video_file_segments import VideoFileSegments
# from filters.single_person.core.multiple_persons_tracks import SinglePersonTrack

segments_list = Annotated[NDArray[np.int32], Literal["N", 2]]


class VideoWriter:
    """
    Class for writing video segments to a different video files to a given output folder.
    """
    def __init__(self, input_filepath: str | Path, output_folder: str | Path, fps: float):
        """
        Description:
            VideoWriter class constructor.

        :param input_filepath: input filepath
        :param output_folder: folder for output videos
        :param fps: FPS for output videos
        """
        self.input_filepath = input_filepath
        self.output_folder = output_folder
        self.fps = fps

    def write_segments(self, video_file_segments: VideoFileSegments, filter_name: str = 'steady') -> None:
        """
        Description:
            Write video segments as separate video files.

        :param video_file_segments: video segments
        :param filter_name: name of the filter (prefix to frames range)
        """

        if video_file_segments.frames_segments.size == 0:
            return

        if video_file_segments.whole_video_segments_check():
            video_filename_base, _ = self.extract_extension_from_filepath(self.input_filepath)
            video_filename = f'{video_filename_base}__{filter_name}__.mp4'
            output_filepath = os.path.join(self.output_folder, video_filename)
            shutil.copy(self.input_filepath, output_filepath)
            return

        # logger.info(f'Video segments: \n {video_segments.segments}')
        video_reader = VideoReader(self.input_filepath, use_tqdm=False)
        resolution = (video_file_segments.metadata.video_width, video_file_segments.metadata.video_height)
        index_segment = 0
        current_segment = video_file_segments.frames_segments.segments[index_segment]
        current_segment_start = current_segment[0]
        current_segment_end = current_segment[1]

        for index_frame, frame in enumerate(video_reader):
            current_video_writer = None
            if index_frame == current_segment_start:
                current_output_filepath = self.current_filepath_segment(current_segment, filter_name)
                current_video_writer = cv2.VideoWriter(current_output_filepath, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, resolution)
                # logger.info(f'Opened video for writing with segment {video_segments.segments[index_segment]}, {index_segment=}')
                # logger.info(f'Video segments \n: {video_segments.segments}')

            if current_segment_start <= index_frame <= current_segment_end:
                current_video_writer.write(frame)

            if index_frame == current_segment_end:
                current_video_writer.release()
                index_segment += 1
                if index_segment == video_file_segments.frames_segments.shape[0]:
                    return
                current_segment = video_file_segments.frames_segments.segments[index_segment]
                current_segment_start = current_segment[0]
                current_segment_end = current_segment[1]

    @staticmethod
    def extract_extension_from_filepath(input_filepath) -> tuple[str, str]:
        """
        Description:
            Extract file name without extension and file extension from file pathname.

        :return: file name and file extension
        """
        video_filename = os.path.basename(input_filepath)
        return os.path.splitext(video_filename)

    def current_filepath_segment(self, segment: np.ndarray, frames_range_prefix='steady') -> str:
        """
        Description:
            Get video file name for a given segment

        :param segment: video segment (just start and end frame)
        :param frames_range_prefix: frames range prefix
        :return: filename
        """
        video_filename_base, _ = self.extract_extension_from_filepath(self.input_filepath)
        start_frame = str(segment[0]).zfill(5)
        end_frame = str(segment[1]).zfill(5)
        video_filename = f'{video_filename_base}__{frames_range_prefix}_{start_frame}-{end_frame}__.mp4'
        output_filepath = os.path.join(self.output_folder, video_filename)
        return output_filepath

    def write_segments_values(self, video_segments: VideoFileSegments, filter_name: str = 'steady') -> None:
        """
        Description:
            Write segments values. This feature is for debug purposes

        :param video_segments: video segments
        :param filter_name: filter name (e.g. steady or non-steady)
        :return: None
        """
        video_filename_base, _ = self.extract_extension_from_filepath(self.input_filepath)
        segments_values_filename = f'{video_filename_base}__{filter_name}__.npy'
        segments_values_filepath = os.path.join(self.output_folder, segments_values_filename)

        video_segments.frames_segments.write(segments_values_filepath)
