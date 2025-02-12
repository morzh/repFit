import copy
import unittest
import numpy as np

from core.filters.single_person.source.filter_addons.confidence_filter_addon import ConfidenceFilterAddon
from core.utils.cv.frames_indices import FramesIndices
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks
from core.filters.single_person.source.person_tracked_data import PersonTrackedData
from core.filters.single_person.source.single_person_track import SinglePersonTrack
from core.utils.cv.video_properties import VideoProperties
from core.utils.geometry.bounding_boxes.bounding_boxes_2d_array import BoundingBoxes2DArray


class TestBoundingBoxesIouFilterAddon(unittest.TestCase):
    ...