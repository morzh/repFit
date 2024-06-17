import os
import shutil
from pathlib import Path
from dataclasses import dataclass
import cv2
import numpy as np
import ffmpeg
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.config import FFMPEG_BINARY
import subprocess

from cv_utils.video_reader import VideoReader
from filters.steady_camera_filter.core.video_segments import VideoSegments

from typing import Annotated, Literal, TypeVar, Optional
from numpy.typing import NDArray

segments_list = Annotated[NDArray[np.int32], Literal["N", 2]]


@dataclass(frozen=True, slots=True)
class VideoSegmentsWriter:
    input_filepath: str | Path
    output_folder: str | Path
    fps: int
    width: int
    height: int
    scale: float = 0.5

    def write_segments_ffmpeg(self, video_segments: VideoSegments) -> None:
        for segment in video_segments.segments:
            ffmpeg_input = ffmpeg.input(self.input_filepath)
            current_output_filepath = self.current_filepath_segment(segment)
            ffmpeg_output = ffmpeg.output(ffmpeg_input.trim(start_frame=segment[0], end_frame=segment[1]), current_output_filepath)
            ffmpeg.run(ffmpeg_output)

    def write_segments_moviepy_ffmpeg(self, video_segments: VideoSegments) -> None:
        for segment in video_segments.segments:
            time_start = segment[0] / video_segments.video_fps
            time_end = segment[1] / video_segments.video_fps
            duration = time_end - time_start
            # T1, T2 = [int(1000 * t) for t in [time_start, time_end]]
            current_output_filepath = self.current_filepath_segment(segment)
            # subprocess.run(['ffmpeg', '-y', '-i', self.input_filepath, '-ss', "%0.2f" % time_start, '-t', "%0.2f" % duration, '-codec', 'copy', '-strict', '-2', current_output_filepath])
            ffmpeg_extract_subclip(self.input_filepath, time_start, time_end, targetname=current_output_filepath)

    def write_segments_cv2(self, video_segments: VideoSegments) -> None:
        video_filename_base, video_filename_extension = self.extract_filename_base_extension()
        video_reader = VideoReader(self.input_filepath, use_tqdm=False)
        index_segment = 0
        current_segment = video_segments.segments[index_segment]

        for index_frame, frame in enumerate(video_reader):
            if index_frame == current_segment[0]:
                current_output_filepath = self.current_filepath_segment(current_segment)
                current_video_writer = cv2.VideoWriter(current_output_filepath, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))

            if current_segment[0] <= index_frame <= current_segment[1]:
                current_video_writer.write(frame)

            if index_frame == current_segment[1]:
                current_video_writer.release()
                index_segment += 1
                if index_segment == video_segments.segments.shape[0]:
                    return
                current_segment = video_segments.segments[index_segment]

    def write_segments_gaps(self, video_segments: VideoSegments, write_method: str = 'ffmpeg'):
        segments_gaps = self.calculate_segments_gaps(video_segments)
        raise NotImplementedError('Writing non steady camera video segments in not implemented yet')

    def write(self, video_segments: VideoSegments, write_method: str = 'cv2', use_gaps: bool = False):
        if video_segments.segments.size == 0:
            return

        if use_gaps:
            video_segments_gaps = self.write_segments_gaps(video_segments, write_method)

        if (video_segments.segments.shape[0] == 1 and
                video_segments.segments[0, 0] == 0 and video_segments.segments[-1, -1] == video_segments.frames_number - 1):
            video_filename = os.path.basename(self.input_filepath)
            output_filepath = os.path.join(os.path.join(self.output_folder, video_filename))
            shutil.copy(self.input_filepath, output_filepath)
            return

        if write_method == 'cv2':
            self.write_segments_cv2(video_segments)
        elif write_method == 'ffmpeg':
            self.write_segments_ffmpeg(video_segments)
        elif write_method == 'moviepy_ffmpeg':
            self.write_segments_moviepy_ffmpeg(video_segments)
        else:
            raise NotImplementedError('Methods other than cv2 and ffmpeg are not implemented yet')

    def calculate_segments_gaps(self, video_segments: VideoSegments):
        pass

    def extract_filename_base_extension(self) -> tuple[str, str]:
        video_filename = os.path.basename(self.input_filepath)
        return os.path.splitext(video_filename)

    def current_filepath_segment(self, segment):
        video_filename_base, _ = self.extract_filename_base_extension()
        filename_postfix = '_' + str(segment[0]) + '-' + str(segment[1]) + '__'
        video_filename = video_filename_base + '__steady' + filename_postfix + '.mp4'
        output_filepath = os.path.join(self.output_folder, video_filename)
        return output_filepath
