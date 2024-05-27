from tqdm import tqdm
import cv2


class VideoReader:
    def __init__(self, fpath):
        self.fpath = fpath
        self.video_reader = cv2.VideoCapture(str(fpath))
        self.n_frames = int(self.video_reader.get(cv2.CAP_PROP_FRAME_COUNT)) - 2
        self.success, self.frame = self.video_reader.read()
        self.fps = int(self.video_reader.get(cv2.CAP_PROP_FPS))
        self._progress = tqdm(range(self.n_frames))

    def frame_generator(self):
        while self.success:
            self.success, _frame = self.video_reader.read()
            frame = self.frame
            self.frame = _frame
            self._progress.update()
            yield frame
