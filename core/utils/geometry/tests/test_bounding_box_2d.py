import unittest
import numpy as np
import matplotlib.pyplot as plt
from core.utils.geometry.bounding_box_2d import BoundingBox2D


class TestBoundingBox(unittest.TestCase):

    def setUp(self):
        self.number_checks = 1_500


    def test_offset(self):
        point_cloud = self.generate_bounding_box_vertices()
        box = self.vertices_to_bounding_box(point_cloud)
        offset_value  = np.random.randint(-box.minimum_dimension_value(), box.minimum_dimension_value())

        offset_point_cloud = self.offset_point_cloud(point_cloud, offset_value)
        vertices_check = box.offset(offset_value).vertices()

        self.assertTrue(np.all(vertices_check == offset_point_cloud))


    def test_vertices(self):
        for _ in range(self.number_checks):
            vertices_point_cloud = self.generate_bounding_box_vertices()
            current_center = np.mean(vertices_point_cloud, axis=0)
            box = self.vertices_to_bounding_box(vertices_point_cloud)

            self.assertTrue(np.all(vertices_point_cloud == box.vertices()))
            self.assertTrue(np.all(vertices_point_cloud[0] == box.left_top))
            self.assertTrue(np.all(vertices_point_cloud[1] == box.right_top))
            self.assertTrue(np.all(vertices_point_cloud[2] == box.right_bottom))
            self.assertTrue(np.all(vertices_point_cloud[3] == box.left_bottom))
            self.assertAlmostEqual(np.sum(current_center - box.center()), 0, delta=1e-9)


    def test_contain_methods(self):
        for _ in range(self.number_checks):
            box = self.generate_bounding_box()
            vertices = box.vertices()
            inner_point_1 = self.inner_point(vertices)
            inner_point_2 = self.inner_point(vertices)
            outer_point = self.generate_outer_point(vertices)

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


    def test_contains_point(self):
        for _ in range(self.number_checks):
            box = self.generate_bounding_box()
            center = box.center()
            vertices = box.vertices()

            self.assertTrue(box.contains_single_point(center, use_border=False))
            for vertex in vertices:
                self.assertTrue(box.contains_single_point(vertex, use_border=True))

            self.assertTrue(box.contains_single_point(self.inner_point(vertices), use_border=False))


    def test_enlarge_vertically(self):
        for _ in range(self.number_checks):
            ...


    def test_enlarge_horizontally(self):
        pass


    @staticmethod
    def generate_bounding_box() -> BoundingBox2D:
        left_top = np.random.randint(-10_000, 10_000, 2)
        width_height = np.random.randint(1, 2500, 2)
        return BoundingBox2D(int(left_top[0]), int(left_top[1]), int(width_height[0]), int(width_height[1]))


    @staticmethod
    def generate_bounding_box_vertices() -> np.ndarray:
        values_x = np.random.randint(-10_000, 10_000, 2)
        values_y = np.random.randint(-10_000, 10_000, 2)

        left_top = np.array([min(values_x), min(values_y)])
        right_top = np.array([max(values_x), min(values_y)])
        right_bottom = np.array([max(values_x), max(values_y)])
        left_bottom = np.array([min(values_x), max(values_y)])

        return np.vstack((left_top, right_top, right_bottom, left_bottom))


    @staticmethod
    def vertices_to_bounding_box(point_cloud) -> BoundingBox2D:
        return BoundingBox2D.from_two_points(point_cloud[0], point_cloud[2])


    @staticmethod
    def offset_point_cloud(point_cloud, offset_value) -> np.ndarray:
        point_cloud_offset = point_cloud.copy()
        #  Offset in Y direction
        point_cloud_offset[0, 1] -= offset_value
        point_cloud_offset[1, 1] -= offset_value

        point_cloud_offset[2, 1] += offset_value
        point_cloud_offset[3, 1] += offset_value

        #  Offset in X direction
        point_cloud_offset[1, 0] += offset_value
        point_cloud_offset[2, 0] += offset_value

        point_cloud_offset[0, 0] -= offset_value
        point_cloud_offset[3, 0] -= offset_value

        return point_cloud_offset


    @staticmethod
    def inner_point(vertices: np.ndarray) -> np.ndarray:
        convex_combination_coefficients = np.random.random(4) + 1e-5
        coefficients_factor = np.sum(convex_combination_coefficients)
        convex_combination_coefficients /= coefficients_factor

        inner_point = np.zeros(2, )
        for index in range(4):
            inner_point += convex_combination_coefficients[index] * vertices[index]

        return inner_point


    @staticmethod
    def generate_outer_point(vertices: np.ndarray) -> np.ndarray:
        side = np.random.randint(0, 4)
        spread = 1_000

        match side:
            case 0:
                y_top_minus = np.random.randint(vertices[0, 1] - spread, vertices[0,1] - 1)
                x_value = np.random.randint(-10_000, 10_000)
                return np.array([x_value, y_top_minus])
            case 1:
                x_right_plus = np.random.randint(vertices[1, 0] + 1, vertices[1, 0] + spread)
                y_value = np.random.randint(-10_000, 10_000)
                return np.array([x_right_plus, y_value])
            case 2:
                y_bottom_plus = np.random.randint(vertices[2, 1] + 1, vertices[2, 1] + spread)
                x_value = np.random.randint(-10_000, 10_000)
                return np.array([x_value, y_bottom_plus])
            case 3:
                x_right_minus = np.random.randint(vertices[0, 0] - spread, vertices[0, 0] - 1)
                y_value = np.random.randint(-10_000, 10_000)
                return np.array([x_right_minus, y_value])
