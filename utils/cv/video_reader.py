import os.path

from pathlib import Path
import cv2
from tqdm import tqdm

from typing import Generator


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
    def __init__(self, filepath: str | Path, **reader_options):
        """
        Description:
            VideoReader class constructor.

        :param filepath: video file path

        :keyword skip_first_frames_number:
        :keyword number_of_frames_to_drop:
        :keyword skip_last_frames_number:

        :raises FileNotFoundError: If video file is not presented at given path.
        """
        if os.path.exists(str(filepath)):
            self.video_capture = cv2.VideoCapture(str(filepath))
        else:
            FileNotFoundError(f'Video file {filepath} does not exist')

        self.approximate_frames_number: int = 0
        self.success: bool = False
        self.frame = None
        self._use_tqdm = reader_options.get('use_tqdm', False)

        self._fps: float = 0.0
        self._video_current_frame_index: int = -1
        self._current_captured_frame_index: int = -1
        self.stride = max(reader_options.get('stride', 1), 1)

        self._init_video_capture()

    def _init_video_capture(self) -> None:
        """
        Description:
            Initialize frames capturing process.
        """
        if self.video_capture.isOpened():
            self.approximate_frames_number = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))

            self.frame = self.read_frame()

            if self.success and self._use_tqdm:
                self._progress = tqdm(range(self.approximate_frames_number))
                self._progress.update()

    def frame_generator(self, skip_first_frames_number=0) -> Generator[cv2.typing.MatLike]:
        """
        Description:
            Frames generator with skipping frames tqdm options.

        :return: generator object
        """
        while self.success:
            current_frame = self.read_frame()
            if self._video_current_frame_index % self.stride == 0:
                yield_frame = self.frame
                self.frame = current_frame
                self._current_captured_frame_index += 1
                yield yield_frame

    def __iter__(self) -> Generator[cv2.typing.MatLike]:
        """
         Description:
            Frames generator  without tqdm progress.

        :return: generator object
        """
        while self.success:
            current_frame = self.read_frame()
            if self._video_current_frame_index % self.stride == 0:
                yield_frame = self.frame
                self.frame = current_frame
                self._current_captured_frame_index += 1
                yield yield_frame

    def __del__(self):
        self.video_capture.release()

    def read_frame(self) -> cv2.typing.MatLike:
        """
        Description:
        """
        self.success, frame = self.video_capture.read()
        if self.success:
            self._video_current_frame_index += 1
        return frame


    @property
    def current_frame_index(self):
        """
        Description:
            Returns current video frame index
        """
        return self._video_current_frame_index

    @staticmethod
    def imshow(frame, window_name: str = 'window') -> None:
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1)
        if key == 27:  # if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            exit(1)

    @property
    def progress(self):
        return self._progress.n

    @property
    def fps(self) -> float:
        """
        Description:
            Get frames per second value

        :return: frames per second
        """
        return self._fps

    @property
    def width(self) -> int:
        """
        Description:
            Get video width.

        :return: video width
        """
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """
        Description:
            Get video height.

        :return: video height
        """
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def resolution(self) -> tuple[int, int]:
        """
        Description:
            Get resolution in (width, height) format.

        :return: video resolution
        """
        return self.width, self.height

    @property
    def video_duration(self) -> float:
        """
        Description:
            Get video duration in seconds.

        :return: video duration
        """
        return self.approximate_frames_number / self._fps
