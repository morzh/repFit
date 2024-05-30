from typing import Union
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
    def __init__(self, fpath: Union[str, Path]):
        self.fpath = fpath
        self.video_reader = cv2.VideoCapture(str(fpath))
        self.n_frames = None
        self._fps = None
        self.success = False
        self.frame = None
        self._init_info()

    def _init_info(self):
        if self.video_reader.isOpened():
            self.n_frames = int(self.video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = int(self.video_reader.get(cv2.CAP_PROP_FPS))
            self.success, self.frame = self.video_reader.read()
            if self.success:
                self._progress = tqdm(range(self.n_frames))
                self._progress.update()

    def frame_generator(self):
        """
        :return: generator object
        """
        while self.success:
            self.success, _frame = self.video_reader.read()
            frame = self.frame
            self.frame = _frame
            self._progress.update()
            yield frame

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

    def __del__(self):
        if self.video_reader is not None:
            self.video_reader.release()

