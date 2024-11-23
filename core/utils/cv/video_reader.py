import cv2
import os.path
from typing import Iterator

from core.utils.cv.video_properties import VideoProperties


class VideoReader:
    """
    Description:
        Base class for frames reading. Frames generator should be implemented in derived class.
    """

    def __init__(self, video_filepath: os.PathLike, *args, **kwargs):
        """
        Description:
            VideoReaderBase class constructor.

        :param video_filepath: video file path

        :raises FileNotFoundError: If video file is not presented at given ``video_filepath``.
        """
        if os.path.exists(str(video_filepath)):
            self.video_capture = cv2.VideoCapture(str(video_filepath))
        else:
            FileNotFoundError(f'Video file {video_filepath} does not exist')

        self.video_properties: VideoProperties = self._init_video_properties(video_filepath)

        self._success: bool = True
        self._current_frame: cv2.typing.MatLike | None = None
        self._current_frame_index: int = -1
        # self._init_video_capture()


    def __iter__(self) -> Iterator[cv2.typing.MatLike]:
        """
         Description:
            Frames generator without tqdm progress.

        :rtype: video frame
        """
        while True:
            # current_frame = self._read_frame()
            success, frame = self.video_capture.read()
            if not success:
                break
            self._current_frame_index += 1
            yield frame


    def __del__(self):
        self.video_capture.release()


    '''
    def _init_video_capture(self) -> None:
        """
        Description:
            Initialize frames capturing process.
        """
        if self.video_capture.isOpened():
            self._current_frame = self._read_frame()
    '''

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

    '''
    def _read_frame(self) -> cv2.typing.MatLike:
        """
        Description:
            Read frame from video capture frame generator

        :return: video frame
        """
        self._success, frame = self.video_capture.read()
        if self._success:
            self._current_frame_index += 1
        return frame
    '''


    @property
    def current_frame_index(self) -> int:
        """
        Description:
            Returns current video frame index, taking stride into account

        :return: current stride frame index
        """
        return self._current_frame_index

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
