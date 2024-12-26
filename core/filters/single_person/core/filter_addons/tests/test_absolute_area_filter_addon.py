import copy
import unittest
import numpy as np


class TestAbsoluteAreaFilterAddon(unittest.TestCase):

    def setUp(self):
        self.number_checks = 1_500


    @staticmethod
    def generate_tracks(number_persons: int, number_video_frames: int, bounding_boxes_areas):
        ...
