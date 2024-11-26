from collections import deque
import cv2
import os
from typing import Iterator

from core.utils.cv.video_reader import VideoReader


class VideoStrideReader(VideoReader):
    """
    Description:
        Read frames from video file. Each n-th frame will be read, others -- dropped. If stride equals to one video frames will be yielded as in ordinary case.

    :ivar _stride: frames stride. After current frame read, next (stride - 1) frames will bw skipped.
    :ivar _stride_frames: Double queue with two stride frames. We need  this queue cause last read frame should not be yield.
    The reason of such behaviour is FrameSegments class consideration and further frame segments calculation.
    """

    def __init__(self, video_filepath: os.PathLike, **options):
        """
        Description:
            VideoStrideReader class constructor.

        :param video_filepath: video file path

        :keyword stride: frames stride or read frames with frame gaps.

        :raises FileNotFoundError: If video file is not presented at given ``video_filepath``.
        """
        super().__init__(video_filepath, **options)
        self._stride: int = max(options.get('stride', 1), 1)
        self._stride_frames = deque(maxlen=2)


    def __iter__(self) -> Iterator[cv2.typing.MatLike]:
        """
         Description:
            Stride frames generator.

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
            Frames stride getter.

        :return: video frames stride
        """
        return self._stride
