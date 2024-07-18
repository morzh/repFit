import os.path
from tqdm import tqdm
from pathlib import Path
import cv2


class VideoReader:
    """
    Read frames from video with frame_generator.
    Example:
        video_reader = VideoReader(video_fpath)
        frame_generator = video_reader.frame_generator()
        for frame in frame_generator:
            pass
    """
    def __init__(self, filepath: str | Path, skip_frames_number: int = 0, use_tqdm: bool = True):
        """
        Description:
            VideoReader class constructor.

        :param filepath: video file path
        :param skip_frames_number:
        :param use_tqdm: use tqdm progress bar for frames generator.
        """
        if os.path.exists(str(filepath)):
            self.video_capture = cv2.VideoCapture(str(filepath))
        else:
            FileNotFoundError(f'Video file {filepath} does not exist')

        self.frames_number: int = 0
        self.success: bool = False
        self.frame = None
        self.use_tqdm = use_tqdm

        self._fps: float = 0.0
        self._current_frame_index: int = -1
        self._integer_division_value = max(skip_frames_number + 1, 1)
        self._init_info()

    def _init_info(self):
        if self.video_capture.isOpened():
            self.frames_number = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
            self.success, self.frame = self.video_capture.read()
            if self.success and self.use_tqdm:
                self._progress = tqdm(range(self.frames_number))
                self._progress.update()

    def frame_generator(self):
        """
        :return: generator object
        """
        while self.success:
            self.success, _frame = self.video_capture.read()
            return_frame = self.frame
            self.frame = _frame
            self._current_frame_index += 1
            if self.use_tqdm:
                self._progress.update()
            yield return_frame

    def __iter__(self):
        while self.success:
            self.success, _frame = self.video_capture.read()
            return_frame = self.frame
            self.frame = _frame
            self._current_frame_index += 1
            if not self._current_frame_index % self._integer_division_value:
                yield return_frame

    def __del__(self):
        self.video_capture.release()

    @property
    def current_frame_index(self):
        return self._current_frame_index

    @staticmethod
    def imshow(frame, window_name: str = 'window'):
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1)
        if key == 27:  # if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            exit(1)

    @property
    def progress(self):
        return self._progress.n

    @property
    def fps(self):
        return self._fps

    @property
    def width(self) -> int:
        """
        Description:
            Get video width.

        @return: video width
        """
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """
        Description:
            Get video height.

        @return: video height
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
        return self.frames_number / self._fps
