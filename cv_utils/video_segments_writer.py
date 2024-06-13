import os
from pathlib import Path
from dataclasses import dataclass
import cv2
import numpy as np

from cv_utils.video_reader import VideoReader

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

    def write_segments(self, segments: segments_list):
        video_filename = os.path.basename(self.input_filepath)
        video_filename_base, video_filename_extension = os.path.splitext(video_filename)
        video_reader = VideoReader(self.input_filepath, use_tqdm=False)
        index_segment = 0
        current_segment = segments[index_segment]

        for index_frame, frame in enumerate(video_reader):
            if index_frame == current_segment[0]:
                current_filename_postfix = '_' + str(current_segment[0]) + '-' + str(current_segment[1]) + '__'
                current_video_filename = video_filename_base + '__steady' + current_filename_postfix + video_filename_extension
                current_output_filepath = os.path.join(self.output_folder, current_video_filename)
                current_video_writer = cv2.VideoWriter(current_output_filepath, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))

            if index_frame in current_segment:
                current_video_writer.write(frame)

            if index_frame == current_segment[1]:
                current_video_writer.release()
                index_segment += 1
                if index_segment == segments.shape[0]:
                    return
                current_segment = segments[index_segment]

    def write_segments_gaps(self, segments: segments_list):
        segments_gaps = self.calculate_segments_gaps(segments)
        raise NotImplementedError('Writing non steady camera video segments in not implemented yet')

    def write(self, segments: segments_list, use_gaps=False):
        if use_gaps:
            self.write_segments_gaps(segments)
        else:
            self.write_segments(segments)

    def calculate_segments_gaps(self, segments: segments_list):
        pass
