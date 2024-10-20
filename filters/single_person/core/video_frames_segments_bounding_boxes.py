from attr import dataclass

from utils.cv.video_frames_segments import VideoFramesSegments
from utils.geometry.bounding_boxes_2d_array import BoundingBoxes2DArray


@dataclass(slots=True)
class VideoSegmentsWithBoundingBoxes:
    """

    """
    bounding_boxes_arrays: list[BoundingBoxes2DArray]
    segments: VideoFramesSegments
    stride: int = 1

    def add_data(self, frame, bounding_box):
        """
            Description:
        """

    def bridge_gaps_between_segments(self):
        """
            Description:
        """

