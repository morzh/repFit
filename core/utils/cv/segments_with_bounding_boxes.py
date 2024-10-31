import numpy as np

from core.utils.cv.segments import Segments
from core.utils.cv.video_properties import VideoProperties
from core.utils.geometry.bounding_boxes_2d_array import BoundingBoxes2DArray


class SegmentsWithBoundingBoxes:  # (Segments):
    """
    Description:

    """
    __slots__ = ['segments', 'bounding_boxes_arrays', 'video_properties']

    def __init__(self):
        # super().__init__(None)
        self.segments = Segments()
        self.bounding_boxes_arrays: list[BoundingBoxes2DArray] = []
        self.video_properties: VideoProperties

    def add_data(self, frame_index: int, bounding_box: np.ndarray, interpolate=False):
        """
        Description:
        """
        if not self.segments.size:
            self.segments.append_segment((frame_index, frame_index))


        last_frame_index = self.segments[-1, -1]
        if frame_index == last_frame_index + 1:
            self.segments[-1, -1] = frame_index
            self.bounding_boxes_arrays[-1].append(bounding_box)
        elif frame_index > last_frame_index + 1 and interpolate:
            gap_size = frame_index = last_frame_index
            self.segments[-1, -1] = frame_index
            self.bounding_boxes_arrays[-1].append(bounding_box)
            self.bounding_boxes_arrays[-1].append(bounding_box)

    def bridge_gaps_between_segments(self):
        """
        Description:
        """

    def _check_consistency(self):
        """
        Description:
        """
        segments_consistency = self.segments._check_consistency()
        segments_lengths = self.segments.lengths
        bounding_boxes_amount = [bbox.boxes_number for bbox in self.bounding_boxes_arrays]

