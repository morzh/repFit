import os
import shutil
import ffmpeg
import cv2
import numpy as np

from typing import Annotated, Literal
from numpy.typing import NDArray
from sqlalchemy.testing.plugin.plugin_base import warnings

from core.filters.single_person.core.single_person_track import SinglePersonTrack
from core.utils.cv.video_stride_reader import VideoReader
from core.utils.cv.video_file_segments import VideoFileSegments
from core.utils.io.files_operations import extract_extension_from_filepath, filter_filepath_segment


segments_list = Annotated[NDArray[np.int32], Literal["N", 2]]


class VideoWriter:
    """
    Class for writing video segments to a different video files to a given output folder.
    """
    def __init__(self, input_filepath: os.PathLike, output_folder: os.PathLike, fps: float):
        """
        Description:
            VideoWriter class constructor.

        :param input_filepath: input filepath
        :param output_folder: folder for output videos
        :param fps: FPS for output videos
        """
        self._input_filepath = input_filepath
        self._output_folder = output_folder
        self._fps = fps


    def write_segments(self, video_file_segments: VideoFileSegments, filter_name: str = 'steady', method='cv2'):
        """
        Description:
            Write video segments as separate video files.

        :param video_file_segments: video segments
        :param filter_name: name of the filter (prefix to frames range)
        :param method: write method (cv2 or ffmpeg )

        :raise ValueError: if ``method`` is different from cv2 or ffmpeg.
        """
        if video_file_segments.size == 0:
            warnings.warn('Segments are empty. No video will bw written.')
            return

        if video_file_segments.whole_video_segments_check():
            video_filename_base, _ = extract_extension_from_filepath(self._input_filepath)
            video_filename = f'{video_filename_base}__{filter_name}__.mp4'
            output_filepath = os.path.join(self._output_folder, video_filename)
            shutil.copy(self._input_filepath, output_filepath)
            return

        if method == 'cv2':
            self._write_segments_cv2(video_file_segments, filter_name)
        elif method == 'ffmpeg':
            self._write_segments_ffmpeg(video_file_segments, filter_name)
        else:
            raise ValueError('Method could be only cv2 or ffmpeg')


    def write_person_track(self, track: SinglePersonTrack, person_id: int, suffix_person='ch', suffix_segment='fr') -> None:
        """
        Description:
            Write person's track to a wet of video files.

        :param track: person's track
        """
        for segment in track.frame_segments:
            source_video_filename = os.path.basename(self._input_filepath)
            source_video_filename_base = os.path.split(source_video_filename, '.')[0]
            current_video_filename = f'{source_video_filename_base}__{suffix_person}{person_id}_fr{suffix_segment[0]}-{suffix_segment[1]}__.mp4'
            video_filepath = os.path.join(self._output_folder, current_video_filename)
            self._write_single_segment_ffmpeg(segment, video_filepath)


    def _write_segments_cv2(self, video_file_segments: VideoFileSegments, filter_name: str = 'steady') -> None:
        """
        Description:
            Write video segments as separate video files.

        :param video_file_segments: video segments
        :param filter_name: name of the filter (prefix to frames range)
        """
        # logger.info(f'Video segments: \n {video_segments.segments}')
        video_reader = VideoReader(self._input_filepath)
        resolution = (video_file_segments.video_properties.video_width, video_file_segments.video_properties.video_height)
        index_segment = 0
        current_segment = video_file_segments.values[index_segment]
        current_segment_start = current_segment[0]
        current_segment_end = current_segment[1]

        for index_frame, frame in enumerate(video_reader):
            current_video_writer = None
            if index_frame == current_segment_start:
                current_output_filepath = filter_filepath_segment(self._input_filepath, self._output_folder, current_segment, filter_name)
                current_video_writer = cv2.VideoWriter(current_output_filepath, cv2.VideoWriter_fourcc(*'mp4v'), self._fps, resolution)
                # logger.info(f'Opened video for writing with segment {video_segments.segments[index_segment]}, {index_segment=}')

            if current_segment_start <= index_frame < current_segment_end:
                current_video_writer.write(frame)

            if index_frame == (current_segment_end - 1):
                current_video_writer.release()
                index_segment += 1
                if index_segment == video_file_segments.shape[0]:
                    return
                current_segment = video_file_segments.values[index_segment]
                current_segment_start = current_segment[0]
                current_segment_end = current_segment[1]


    def _write_segments_ffmpeg(self, video_file_segments: VideoFileSegments, postfix: str = 'steady') -> None:
        for segment in video_file_segments:
            current_output_filepath = filter_filepath_segment(self._input_filepath, self._output_folder, segment, postfix)
            self._write_single_segment_ffmpeg(segment, current_output_filepath)


    def _write_single_segment_ffmpeg(self, segment: np.ndarray, output_filepath: str) -> None:
        """
        Description:
            Write video segment to video file. In other words trim video  from start to end video frame.

        :param segment: input segment  (numpy array of size 2)
        :param output_filepath: output video filepath.
        """
        pts = 'PTS-STARTPTS'
        input_stream = ffmpeg.input(self._input_filepath)
        video_cut = input_stream.trim(start=segment[0], end=segment[1]).setpts(pts)
        output_video = ffmpeg.output(video_cut, output_filepath, format='mp4')
        output_video.run()



    @property
    def input_filepath(self) -> os.PathLike:
        return self._input_filepath


    @property
    def output_folder(self) -> os.PathLike:
        return self._output_folder


    @property
    def fps(self) -> float:
        return self._fps
