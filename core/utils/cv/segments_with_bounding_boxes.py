import numpy as np

from core.utils.cv.frames_segments import FramesSegments
from core.utils.cv.video_properties import VideoProperties
from core.utils.geometry.bounding_boxes_2d_array import BoundingBoxes2DArray


class SegmentsWithBoundingBoxes:  # (Segments):
    """
    Description:

    """
    __slots__ = ['segments', 'bounding_boxes']

    def __init__(self):
        self.segments = FramesSegments()
        self.bounding_boxes: list[BoundingBoxes2DArray] = []

    def append(self, frame_index: int, bounding_box: np.ndarray, interpolate=False):
        """
        Description:
        """
        if self.segments.size == 0:
            self.segments.append_segment((frame_index, frame_index))


        last_frame_index = self.segments[-1, -1]
        if frame_index == last_frame_index + 1:
            self.segments[-1, -1] = frame_index
            self.bounding_boxes[-1].append(bounding_box)
        elif frame_index > last_frame_index + 1 and interpolate:
            gap_size = frame_index = last_frame_index
            self.segments[-1, -1] = frame_index
            self.bounding_boxes[-1].append(bounding_box)
            self.bounding_boxes[-1].append(bounding_box)

    def bridge_gaps(self, threshold: int):
        """
        Description:
        """

    def filter_by_time(self, threshold: int) -> None:
        """
        Description:
            Delete segments and corresponding bounding boxes if segments lengths is less than ``threshold``.

        :param threshold: time threshold
        """

    def mean_areas(self) -> np.ndarray:
        """
        Description:
            Calculates mean areas for bounding boxes in self.bounding_boxes list.
        """
        areas = [boxes.mean_area() for boxes in self.bounding_boxes]
        return np.array(areas)


    def _check_consistency(self):
        """
        Description:
        """
        segments_consistency = self.segments._check_consistency()
        segments_lengths = self.segments.lengths
        bounding_boxes_amount = [bbox.boxes_number for bbox in self.bounding_boxes]

