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

    def write(self, segments: segments_list, use_gaps=False):
        video_filename = os.path.basename(self.input_filepath)
        video_filename_base, video_filename_extension = os.path.splitext(video_filename)
        video_reader = VideoReader(self.input_filepath)
        current_segment = segments[0]
        for index_frame, frame in enumerate(video_reader):
            if index_frame < current_segment[0]:
                current_video_writer = None
            elif index_frame in current_segment:
                current_video_filename = video_filename_base + '__steady_' + str(current_segment[0]) + '-' + str(current_segment[1]) + '__' + video_filename_extension
                current_putput_filepath = os.path.join(self.output_folder, current_video_filename)
                current_video_writer = cv2.VideoWriter(current_putput_filepath, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
                current_video_writer.write(frame)
            else:
                current_segment = segments[1]
                if isinstance(current_video_writer, cv2.VideoWriter):
                    current_video_writer.release()
