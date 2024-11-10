import cv2
from pathlib import Path
from typing import Iterator

from core.utils.cv.video_reader import VideoReader


class VideoStrideReader(VideoReader):
    """
    Description:
        Read frames from video file.
    """

    def __init__(self, video_filepath: str | Path, **options):
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
        self._current_stride_frame_index: int = -1

    def __iter__(self) -> Iterator[cv2.typing.MatLike]:
        """
         Description:
            Frames generator without tqdm progress.

        :rtype: video frame
        """
        for frame in super().__iter__():
            if self.current_frame_index % self._stride == 0:
                self._current_stride_frame_index += 1
                yield frame

    @property
    def current_stride_frame_index(self) -> int:
        """
        Description:
            Returns current video frame index, taking stride into account

        :return: current stride frame index
        """
        return self._current_stride_frame_index

    @property
    def stride(self) -> int:
        """
        Description:
            Returns frames stride

        :return: frames stride
        """
        return self._stride
