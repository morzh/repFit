import numpy as np
import unittest

import core.utils.geometry.bounding_boxes.bounding_box_2d_dyadic as bd
from bounding_box_test_utils import generate_bounding_box, generate_outer_points
from core.utils.geometry.bounding_boxes.bounding_box_2d import BoundingBox2D


class TestBoundingBoxes2DArray(unittest.TestCase):

    def setUp(self):
        self.number_checks = 1_500


    def test_intersect(self):
        for _ in range(self.number_checks):
            box = generate_bounding_box()

            offset_value = np.random.randint(1, 10_000)
            offset_bounding_box = box.offset(offset_value)

            vertices = box.corners()
            outer_points = generate_outer_points(vertices, number_points=2)
            outer_bounding_box = BoundingBox2D.from_two_points(outer_points[0], outer_points[1])

            self.assertTrue(box == bd.intersect(box, offset_bounding_box))
            self.assertTrue(box == bd.intersect(offset_bounding_box, box))
            self.assertTrue(box == bd.intersect(box, box.extend(left=offset_value)))
            self.assertTrue(box == bd.intersect(box, box.extend(bottom=offset_value)))
            self.assertTrue(box == bd.intersect(box, box.extend(left=offset_value, right=offset_value)))
            self.assertTrue(box == bd.intersect(box, box.extend(top=offset_value, right=offset_value)))
            self.assertTrue(box == bd.intersect(box, box.extend(bottom=offset_value, right=offset_value)))
            self.assertTrue(bd.intersect(box, outer_bounding_box).is_degenerate())


    def test_subtract(self):
        ...


    def test_union(self):
        ...


    def test_circumscribe(self):
        ...


    def test_intersection_over_union(self):
        ...


    def test_intersections_grid(self):
        ...




