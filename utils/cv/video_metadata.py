from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class VideoProperties:
    """
        Description:
            Video properties storage class

    :ivar filepath: video filepath;
    :ivar width: video width;
    :ivar height: video height;
    :ivar approximate_frames_number: approximate frames number given by cv2.VideoCapture();
    :ivar fps: video frames per second
    """
    filepath: str = ''
    width: int = 0
    height: int = 0
    approximate_frames_number: int = -1
    fps: float = 0.0

    @property
    def resolution(self) -> tuple[int, int]:
        """
        Description:
            Get resolution in (width, height) format.

        :return: video resolution
        """
        return self.width, self.height

    @property
    def duration(self) -> float:
        """
        Description:
            Get video duration in seconds.

        :return: video duration
        """
        return self.approximate_frames_number / self.fps

