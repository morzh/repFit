import os

import cv2
from collections import deque
from pathlib import Path
from typing import Iterator

from core.utils.cv.video_reader import VideoReader


class VideoStrideReader(VideoReader):
    """
    Description:
        Read frames from video file.
    """

    def __init__(self, video_filepath: os.PathLike, **options):
        """
        Description:
            VideoReader class constructor.

        :param video_filepath: video file path

        :keyword stride: frames stride or read frames with frame gaps. If stride equals one frames iteration will be as usual.
        If greater than one, ....

        :raises FileNotFoundError: If video file is not presented at given ``video_filepath``.
        """
        super().__init__(video_filepath, **options)

        self._stride: int = max(options.get('stride', 1), 1)
        self._stride_frames = deque(maxlen=2)


    def __iter__(self) -> Iterator[cv2.typing.MatLike]:
        """
         Description:
            Frames generator without tqdm progress.

        :rtype: video frame
        """
        for frame in super().__iter__():
            if self.current_frame_index % self._stride == 0:
                self._stride_frames.append(frame)
                if len(self._stride_frames) == 2:
                    yield self._stride_frames[0]


    @property
    def stride(self) -> int:
        """
        Description:
            Returns frames stride

        :return: frames stride
        """
        return self._stride
