import cv2
from pathlib import Path
import os.path
from tqdm import tqdm

from typing import Generator

from core.utils.cv.video_properties import VideoProperties


class VideoReader:
    """
    Description:
        Read frames from video with frame_generator() or __iter__.

    Usage example:
        video_reader = VideoReader(video_fpath)
        frame_generator = video_reader.frame_generator()
        for frame in frame_generator:
            pass
    """
    def __init__(self, video_filepath: str | Path, **options):
        """
        Description:
            VideoReader class constructor.

        :param video_filepath: video file path

        :keyword use_tqdm: use console progress indicator
        :keyword stride:

        :raises FileNotFoundError: If video file is not presented at given ``video_filepath``.
        """
        if os.path.exists(str(video_filepath)):
            self.video_capture = cv2.VideoCapture(str(video_filepath))
        else:
            FileNotFoundError(f'Video file {video_filepath} does not exist')

        self.success: bool = False
        self.frame: cv2.typing.MatLike = None
        self.video_properties: VideoProperties = self._init_video_properties(video_filepath)
        self.stride: int = max(options.get('stride', 1), 1)

        self._original_frame_index: int = -1
        self._strided_frame_index: int = -1
        self._use_tqdm = options.get('use_tqdm', False)
        self._init_video_capture()

    def _init_video_properties(self, video_filepath) -> VideoProperties:
        """
            Description:
                Initialize VideoProperties class

        :param video_filepath: video filepath

        :return: VideoProperties class instance
        """
        width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        frames_number = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoProperties(filepath=video_filepath, width=width, height=height, approximate_frames_number=frames_number, fps=fps)

    def _init_video_capture(self) -> None:
        """
        Description:
            Initialize frames capturing process.
        """
        if self.video_capture.isOpened():
            self.frame = self.read_frame()
            if self.success and self._use_tqdm:
                self._progress = tqdm(range(self.video_properties.approximate_frames_number))
                self._progress.update()

    def __iter__(self) -> Generator[cv2.typing.MatLike]:
        """
         Description:
            Frames generator  without tqdm progress.

        :return: generator object
        """
        while self.success:
            current_frame = self.read_frame()
            if self._original_frame_index % self.stride == 0:
                yield_frame = self.frame
                self.frame = current_frame
                self._strided_frame_index += 1
                yield yield_frame

    def __del__(self):
        self.video_capture.release()

    def read_frame(self) -> cv2.typing.MatLike:
        """
        Description:
            Read frame from video capture frame generator

        :return: video frame
        """
        self.success, frame = self.video_capture.read()
        if self.success:
            self._original_frame_index += 1
        return frame


    @property
    def current_frame_index(self) -> int:
        """
        Description:
            Returns current video frame index

        :return: current frame index
        """
        return self._strided_frame_index

    @staticmethod
    def imshow(frame: cv2.typing.MatLike, window_name: str = 'window') -> None:
        """
        Description:
            Shows image.

        :param frame: video frame
        :param window_name:  image window title
        """
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1)
        if key == 27:  # if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            exit(1)

    @property
    def progress(self) -> any:
        """
        Description:

        """
        return self._progress.n
