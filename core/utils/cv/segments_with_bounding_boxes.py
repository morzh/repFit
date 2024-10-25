from attr import dataclass

from core.utils.cv.segments import Segments
from core.utils.cv.video_properties import VideoProperties
from core.utils.geometry.bounding_boxes_2d_array import BoundingBoxes2DArray


@dataclass(slots=True)
class SegmentsWithBoundingBoxes(Segments):
    """

    """
    bounding_boxes_arrays: list[BoundingBoxes2DArray]
    segments: Segments
    video_properties: VideoProperties
    stride: int = 1

    def add_data(self, frame, bounding_box):
        """
            Description:
        """

    def bridge_gaps_between_segments(self):
        """
            Description:
        """

