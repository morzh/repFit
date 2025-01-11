import os.path
import pickle
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.spatial import distance
import unittest

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


    def test_enlarge_visually(self):
        def on_key_press(event):
            if event.key == 'd':
                data = {'box_to_enlarge': current_box_to_enlarge, 'obstacle_boxes': current_boxes}
                filename = str(time.time()) + '.pickle'
                filepath = os.path.join('box_enlarge_dump', filename)
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        number_boxes = 15
        for _ in range(self.number_checks):
            random_range = (-10_000, 10_000)
            borderline_box = BoundingBox2D(random_range[0], random_range[0], random_range[1] - random_range[0], random_range[1] - random_range[0])
            current_points = np.random.randint(random_range[0], random_range[1], size=(number_boxes, 2))
            current_distances = distance.cdist(current_points, current_points, 'euclidean')
            current_estimated_distance = 0.2 * np.mean(current_distances)
            current_boxes = [BoundingBox2D.from_center_and_dimensions(center, current_estimated_distance, current_estimated_distance) for center in current_points]

            current_box_pop_index = np.random.randint(0, len(current_boxes))
            current_box_to_enlarge = current_boxes.pop(current_box_pop_index)
            current_box_enlarged = current_box_to_enlarge.enlarge(current_boxes, borderline_box)

            fig, ax = plt.subplots()
            cid = fig.canvas.mpl_connect('key_press_event', on_key_press)
            fig.set_size_inches(22.5, 14.5)
            for box in current_boxes:
                rect = Rectangle((box.x, box.y), width=box.width, height=box.height, edgecolor='blue', facecolor=(1, 1, 1, 0))
                ax.add_patch(rect)
            rect = Rectangle((current_box_to_enlarge.x, current_box_to_enlarge.y), width=current_box_to_enlarge.width, height=current_box_to_enlarge.height,
                             edgecolor='red', facecolor=(1, 1, 1, 0), linewidth=3)
            ax.add_patch(rect)
            rect = Rectangle((current_box_enlarged.x, current_box_enlarged.y), width=current_box_enlarged.width, height=current_box_enlarged.height, edgecolor='black', facecolor=(1, 1, 1, 0))
            ax.add_patch(rect)

            plt.axvline(x=random_range[0], color=(1, 0, 0, 0.25), label='axvline - full height')
            plt.axvline(x=random_range[1], color=(0, 0, 1, 0.25), label='axvline - full height')

            plt.axhline(y=random_range[0], color=(1, 0, 0, 0.25), label='axhline - full width')
            plt.axhline(y=random_range[1], color=(0, 0, 1, 0.25), label='axhline - full width')

            plt.scatter(current_points[:, 0], current_points[:, 1], s=2)
            plt.tight_layout()
            plt.show()


    def test_intersect(self):
        for _ in range(self.number_checks):
            box = generate_bounding_box()

            offset_value = np.random.randint(1, 10_000)
            offset_bounding_box = box.offset(offset_value)

            vertices = box.corners()
            outer_points = generate_outer_points(vertices, number_points=2)
            outer_bounding_box = BoundingBox2D.from_two_points(outer_points[0], outer_points[1])

            self.assertTrue(box == BoundingBox2D.intersect(box, offset_bounding_box))
            self.assertTrue(box == BoundingBox2D.intersect(offset_bounding_box, box))
            self.assertTrue(box == BoundingBox2D.intersect(box, box.extend(left=offset_value)))
            self.assertTrue(box == BoundingBox2D.intersect(box, box.extend(bottom=offset_value)))
            self.assertTrue(box == BoundingBox2D.intersect(box, box.extend(left=offset_value, right=offset_value)))
            self.assertTrue(box == BoundingBox2D.intersect(box, box.extend(top=offset_value, right=offset_value)))
            self.assertTrue(box == BoundingBox2D.intersect(box, box.extend(bottom=offset_value, right=offset_value)))
            self.assertTrue(BoundingBox2D.intersect(box, outer_bounding_box).is_degenerate())


    def test_intersect_visually(self):
        for _ in range(self.number_checks):
            box_1 = generate_bounding_box()
            corner_number = np.random.randint(0,4)
            box_2_center = box_1.corners()[corner_number]
            box_2_dimensions =  np.random.randint(1, box_1.maximum_dimension_value(), 2)
            box_2 = BoundingBox2D.from_center_and_dimensions(box_2_center, box_2_dimensions[0], box_2_dimensions[1])
            boxes_intersection = BoundingBox2D.intersect(box_1, box_2)

            fig, ax = plt.subplots()
            fig.set_size_inches(22.5, 14.5)
            rect_1 = Rectangle((box_1.x, box_1.y), width=box_1.width, height=box_1.height, edgecolor='blue', facecolor=(1, 1, 1, 0))
            rect_2 = Rectangle((box_2.x, box_2.y), width=box_2.width, height=box_2.height, edgecolor='blue', facecolor=(1, 1, 1, 0))
            rect = Rectangle((boxes_intersection.x, boxes_intersection.y), width=boxes_intersection.width, height=boxes_intersection.height, edgecolor='yellow', facecolor=(1, 1, 1, 0))
            ax.add_patch(rect_1)
            ax.add_patch(rect_2)
            ax.add_patch(rect)
            ax.plot()
            plt.show()
