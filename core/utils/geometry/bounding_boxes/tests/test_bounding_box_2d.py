import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

from core.utils.geometry.bounding_boxes.bounding_box_2d import BoundingBox2D
from core.utils.geometry.bounding_boxes.tests.bounding_box_test_utils import (generate_inner_point, generate_outer_points, generate_bounding_box,
                                                                              vertices_to_bounding_box, generate_bounding_box_corners, offset_point_cloud)


class TestBoundingBox(unittest.TestCase):

    def setUp(self):
        self.number_checks = 1_500


    def test_offset(self):
        point_cloud = generate_bounding_box_corners()
        box = vertices_to_bounding_box(point_cloud)
        offset_value  = np.random.randint(-box.minimum_dimension_value(), box.minimum_dimension_value())

        offset_vertices = offset_point_cloud(point_cloud, offset_value)
        vertices_check = box.offset(offset_value).corners()

        self.assertTrue(np.all(vertices_check == offset_vertices))


    def test_vertices(self):
        for _ in range(self.number_checks):
            vertices_point_cloud = generate_bounding_box_corners()
            current_center = np.mean(vertices_point_cloud, axis=0)
            box = vertices_to_bounding_box(vertices_point_cloud)

            self.assertTrue(np.all(vertices_point_cloud == box.corners()))
            self.assertTrue(np.all(vertices_point_cloud[0] == box.left_top))
            self.assertTrue(np.all(vertices_point_cloud[1] == box.right_top))
            self.assertTrue(np.all(vertices_point_cloud[2] == box.right_bottom))
            self.assertTrue(np.all(vertices_point_cloud[3] == box.left_bottom))
            self.assertAlmostEqual(np.sum(current_center - box.center()), 0, delta=1e-9)


    def test_contain_methods(self):
        for _ in range(self.number_checks):
            box = generate_bounding_box()
            vertices = box.corners()
            inner_point_1 = generate_inner_point(vertices)
            inner_point_2 = generate_inner_point(vertices)
            outer_point = generate_outer_points(vertices)

            inner_bounding_box = BoundingBox2D.from_two_points(inner_point_1, inner_point_2)
            intersect_bounding_box = BoundingBox2D.from_two_points(inner_point_1, outer_point)

            offset_value = np.random.randint(1, 10_000)
            outer_bounding_box = box.offset(offset_value)

            self.assertTrue(box.contains_bounding_box(inner_bounding_box))
            self.assertFalse(box.contains_bounding_box(intersect_bounding_box))
            self.assertFalse(box.contains_bounding_box(outer_bounding_box))
            self.assertTrue(box.contains_bounding_box(box.offset(0)))

            self.assertFalse(box.contained_in_bounding_box(inner_bounding_box))
            self.assertFalse(box.contained_in_bounding_box(intersect_bounding_box))
            self.assertTrue(box.contained_in_bounding_box(outer_bounding_box))
            self.assertTrue(box.contained_in_bounding_box(box.offset(0)))


    def test_contains_point(self):
        for _ in range(self.number_checks):
            box = generate_bounding_box()
            center = box.center()
            vertices = box.corners()

            self.assertTrue(box.contains_single_point(center, use_border=False))
            for vertex in vertices:
                self.assertTrue(box.contains_single_point(vertex, use_border=True))

            self.assertTrue(box.contains_single_point(generate_inner_point(vertices), use_border=False))


    def test_enlarge_vertically_visually(self):
        for _ in range(self.number_checks):
            points = np.random.randint(-10_000, 10_000, size=(30, 2))
            distances = distance.cdist(points, points, 'euclidean')
            mean_distance = np.mean(distances)
            boxes = [BoundingBox2D.from_center_and_dimensions(center, mean_distance, mean_distance) for center in points]

            plt.scatter(points[:, 0], points[:, 1])
            plt.show()

    def test_enlarge_horizontally(self):
        pass
