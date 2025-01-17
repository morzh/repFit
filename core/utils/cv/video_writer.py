import os
import shutil
import ffmpeg
import cv2
import numpy as np

from sqlalchemy.testing.plugin.plugin_base import warnings

from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks
from core.filters.single_person.source.single_person_track import SinglePersonTrack
from core.utils.cv.frames_segments import FramesSegments
from core.utils.cv.video_properties import VideoProperties
from core.utils.cv.video_stride_reader import VideoReader
from core.utils.cv.video_file_segments import VideoFileSegments
from core.utils.geometry.bounding_boxes.bounding_boxes_2d_array import BoundingBoxes2DArray
from core.utils.io.files_operations import extract_name_extension_from_filepath, filter_filepath_segment


class VideoWriter:
    """
    Class for writing video segments or multiple persons tracks to separate video files using given output folder.
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


    def write_segments(self, video_file_segments: VideoFileSegments, output_filename_suffix: str = 'steady', method='cv2') -> None:
        """
        Description:
            Write video segments as separate video files.

        :param video_file_segments: video segments
        :param output_filename_suffix: (short) string to represent some additional information in filename.
        This string will be in front of the frames range and after file basename.
        :param method: write method (cv2 or ffmpeg)

        :raise ValueError: if ``method`` is different from cv2 or ffmpeg.
        """
        if video_file_segments.segments.size == 0:
            warnings.warn('Segments are empty. No video will bw written.')
            return

        if video_file_segments.is_whole_video_single_segment():
            video_filename_base, _ = extract_name_extension_from_filepath(self._input_filepath)
            video_filename = f'{video_filename_base}__{output_filename_suffix}__.mp4'
            output_filepath = os.path.join(self._output_folder, video_filename)
            shutil.copy(self._input_filepath, output_filepath)
            return

        if method == 'cv2':
            self._write_segments_cv2(video_file_segments.video_properties, video_file_segments.segments, output_filename_suffix)
        elif method == 'ffmpeg':
            self._write_segments_ffmpeg(video_file_segments, output_filename_suffix)
        else:
            raise ValueError('Method could be only cv2 or ffmpeg')


    def write_segments_with_bounding_boxes(self, segments: FramesSegments, boxes: BoundingBoxes2DArray, output_video_filename_base: str) -> None:
        """
        Description:
            Write frames segments with corresponding bounding boxes to separate video files.

        :param segments: video frames segments
        :param boxes: bounding boxes. Each bounding box corresponds to each segment.
        :param output_video_filename_base: filename base of output video file.
        """

        video_reader = VideoReader(self._input_filepath)
        # source_video_filename = os.path.basename(self._input_filepath)
        # source_video_filename_base = os.path.split(source_video_filename, '.')[0]

        index = 0
        current_segment = segments[index]
        current_segment_start = current_segment[0]
        current_segment_end = current_segment[1]
        current_bounding_box = boxes[index]
        current_resolution = (current_bounding_box[2], current_bounding_box[3])  # width, height
        current_video_writer = None

        for index_frame, frame in enumerate(video_reader):

            if index_frame == current_segment_start:
                current_output_filename = f'{output_video_filename_base}_{current_segment[0]}-{current_segment[1]}__.mp4'
                current_output_filepath = os.path.join(self._output_folder, current_output_filename)
                current_video_writer = cv2.VideoWriter(current_output_filepath, cv2.VideoWriter_fourcc(*'mp4v'), self._fps, current_resolution)

            if current_segment_start <= index_frame < current_segment_end:
                frame_bounding_box = frame[current_bounding_box[1]: current_bounding_box[1] + current_bounding_box[3], current_bounding_box[0]: current_bounding_box[0] + current_bounding_box[2]]
                current_video_writer.write(frame_bounding_box)

            if index_frame == (current_segment_end - 1):
                current_video_writer.release()
                index += 1
                if index == segments.shape[0]:
                    return
                current_segment = segments.values[index]
                current_segment_start = current_segment[0]
                current_segment_end = current_segment[1]



    def _write_segments_cv2(self, video_properties: VideoProperties, segments: FramesSegments, suffix: str = 'steady') -> None:
        """
        Description:
            Write video segments as a separate video files.

        :param video_properties: video properties
        :param segments: video frame segments
        :param suffix: suffix of the output filename (after source filename base and before frames range). E.g. it could be the name of the filter
        """
        # logger.info(f'Video segments: \n {video_segments.segments}')
        video_reader = VideoReader(self._input_filepath)
        resolution = (video_properties.width, video_properties.height)
        index_segment = 0
        current_segment = segments.values[index_segment]
        current_segment_start = current_segment[0]
        current_segment_end = current_segment[1]
        current_video_writer = None

        for index_frame, frame in enumerate(video_reader):
            if index_frame == current_segment_start:
                current_output_filepath = filter_filepath_segment(self._input_filepath, self._output_folder, current_segment, suffix)
                current_video_writer = cv2.VideoWriter(current_output_filepath, cv2.VideoWriter_fourcc(*'mp4v'), self._fps, resolution)
                # logger.info(f'Opened video for writing with segment {video_segments.segments[index_segment]}, {index_segment=}')

            if current_segment_start <= index_frame < current_segment_end:
                current_video_writer.write(frame)

            if index_frame == (current_segment_end - 1):
                current_video_writer.release()
                index_segment += 1
                if index_segment == segments.shape[0]:
                    return
                current_segment = segments.values[index_segment]
                current_segment_start = current_segment[0]
                current_segment_end = current_segment[1]


    def _write_segments_ffmpeg(self, video_file_segments: VideoFileSegments, suffix: str = 'steady') -> None:
        """
        Description:
            Write video file segments to separate video files.

        :param video_file_segments: video file segments;
        :param suffix: (short) string to represent some additional information in filename.
        """
        for segment in video_file_segments.segments:
            current_output_filepath = filter_filepath_segment(self._input_filepath, self._output_folder, segment, suffix)
            self._write_single_segment_ffmpeg(segment, current_output_filepath)


    def _write_single_segment_ffmpeg(self, segment: np.ndarray, output_filepath: str) -> None:
        """
        Description:
            Write single video segment to the video file. In other words trim video from start to end video frame.

        :param segment: input segment (numpy array of size 2)
        :param output_filepath: output video filepath.
        """
        pts = 'PTS-STARTPTS'
        input_stream = ffmpeg.input(self._input_filepath)
        video_cut = input_stream.trim(start=segment[0], end=segment[1]).setpts(pts)
        output_video = ffmpeg.output(video_cut, output_filepath, format='mp4')
        output_video.run()


    @property
    def input_filepath(self) -> os.PathLike:
        """
        Description:
            Input video file path getter.

        :return: input video filepath.
        """
        return self._input_filepath


    @property
    def output_folder(self) -> os.PathLike:
        """
        Description:
            Output folder getter.

        :return: output folder for videos
        """
        return self._output_folder


    @property
    def fps(self) -> float:
        """
        Description:
            Input video file frames per second path getter.

        :return: input video FPS.
        """
        return self._fps
