import os.path
from tqdm import tqdm
from pathlib import Path
import cv2


class VideoReader:
    """ Read frames from video with frame_generator

        example:
        video_reader = VideoReader(video_fpath)
        frame_generator = video_reader.frame_generator()
        for frame in frame_generator:
            pass

    """
    def __init__(self, fpath: str | Path, use_tqdm=True):
        self.fpath = fpath
        if os.path.exists(str(fpath)):
            self.video_capture = cv2.VideoCapture(str(fpath))
        else:
            FileNotFoundError(f'Video file {self.fpath} does not exist')

        self.n_frames = None
        self._fps = None
        self._current_frame_index = 0
        self.success = False
        self.frame = None
        self.use_tqdm = use_tqdm
        self._init_info()

    def _init_info(self):
        if self.video_capture.isOpened():
            self.n_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
            self.success, self.frame = self.video_capture.read()
            if self.success and self.use_tqdm:
                self._progress = tqdm(range(self.n_frames))
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
            self._progress.update()
            yield return_frame

    def __next__(self):
        if self.success:
            self.success, _frame = self.video_capture.read()
            return_frame = self.frame
            self.frame = _frame
            self._current_frame_index += 1
            return return_frame
        else:
            StopIteration

    def __iter__(self):
        while self.success:
            self.success, _frame = self.video_capture.read()
            return_frame = self.frame
            self.frame = _frame
            self._current_frame_index += 1
            yield return_frame

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
        @return: video width
        """
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """
        @return: video height
        """
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def resolution(self) -> tuple[int, int]:
        """
        Resolution in (width, height) format
        """
        return self.width, self.height

    def __del__(self):
        if self.video_capture is not None:
            self.video_capture.release()
