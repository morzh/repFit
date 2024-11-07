import copy
import unittest

import numpy as np

from core.utils.geometry.bounding_boxes_2d_array import BoundingBoxes2DArray


class TestBoundingBoxes2DArray(unittest.TestCase):

    def setUp(self):
        self.number_checks = 1_500
        self.maximum_number_boxes = 1_000
        self.top_left_range = (-1_000, 1_000)


    def test_init_inconsistent(self):
        for _ in range(self.number_checks):
            number_boxes = np.random.randint(1, self.maximum_number_boxes)
            boxes_array_top_left = np.random.randint(self.top_left_range[0], self.top_left_range[1], (number_boxes, 2))
            boxes_array_width_height = np.random.randint(-500, 0, (number_boxes, 2))
            boxes_array = np.hstack((boxes_array_top_left, boxes_array_width_height))

            with self.assertRaises(ValueError):
                BoundingBoxes2DArray(boxes_array)


    def test_append_correct_size(self):
        for _ in range(self.number_checks):
            number_boxes = np.random.randint(1, self.maximum_number_boxes)
            boxes_array_top_left = np.random.randint(self.top_left_range[0], self.top_left_range[1], (number_boxes, 2))
            boxes_array_width_height = np.random.randint(1, 500, (number_boxes, 2))
            boxes_array = np.hstack((boxes_array_top_left, boxes_array_width_height))

            bounding_boxes_2d = BoundingBoxes2DArray(copy.deepcopy(boxes_array))

            bounding_box_1 = np.random.randint(0, 500, (4,))
            bounding_box_2 = np.random.randint(0, 500, (1, 4))
            bounding_box_3 = np.random.randint(0, 500, (4, 1))

            bounding_boxes_2d.append(bounding_box_1)
            bounding_boxes_2d.append(bounding_box_2)
            bounding_boxes_2d.append(bounding_box_3)

            boxes_array = np.vstack((boxes_array, bounding_box_1.reshape(1, 4), bounding_box_2, bounding_box_3.reshape(1, 4)))
            self.assertTrue(np.alltrue(boxes_array == bounding_boxes_2d.values))


    def test_append_incorrect_size(self):
        for _ in range(self.number_checks):
            number_boxes = np.random.randint(1, self.maximum_number_boxes)
            boxes_array_top_left = np.random.randint(self.top_left_range[0], self.top_left_range[1], (number_boxes, 2))
            boxes_array_width_height = np.random.randint(1, 500, (number_boxes, 2))
            boxes_array = np.hstack((boxes_array_top_left, boxes_array_width_height))

            bounding_boxes_2d = BoundingBoxes2DArray(copy.deepcopy(boxes_array))
            random_size = np.random.randint(1, 100)
            if random_size == 4: random_size += 1

            bounding_box_1 = np.random.randint(0, 500, (random_size,))
            bounding_box_2 = np.random.randint(0, 500, (1, random_size))
            bounding_box_3 = np.random.randint(0, 500, (random_size, 1))

            with self.assertRaises(ValueError):
                bounding_boxes_2d.append(bounding_box_1)
                bounding_boxes_2d.append(bounding_box_2)
                bounding_boxes_2d.append(bounding_box_3)


    def test_append_incorrect_dimensions(self):
        for _ in range(self.number_checks):
            number_boxes = np.random.randint(0, self.maximum_number_boxes)
            boxes_array_top_left = np.random.randint(self.top_left_range[0], self.top_left_range[1], (number_boxes, 2))
            boxes_array_width_height = np.random.randint(1, 300, (number_boxes, 2))
            boxes_array = np.hstack((boxes_array_top_left, boxes_array_width_height))

            bounding_boxes_2d = BoundingBoxes2DArray(copy.deepcopy(boxes_array))
            bounding_box = np.random.randint(-500, 500, size=(1, 4))
            if np.alltrue(bounding_box[0, 2:] > 0):
                bounding_box[0, 2] *= -1.0
            if np.any(bounding_box[0, 2:] == 0):
                bounding_box[0, 3] = -1.0

            with self.assertRaises(ValueError):
                bounding_boxes_2d.append(bounding_box)


    def test_extend(self):
        ...

    def test_extend_incorrect_size(self):
        ...

    def test_extend_incorrect_dimensions(self):
        ...


    def test_circumscribe(self):
        for _ in range(self.number_checks):
            left_top = np.random.randint(self.top_left_range[0], self.top_left_range[1], (2,))
            width_height = np.random.randint(1, 512, (2,))
            apriori_known_circumscribed_box = np.array([*left_top, *width_height])

            number_boxes = np.random.randint(1, 512)
            right_bottom = left_top + width_height

            xs = np.random.randint(left_top[0], right_bottom[0], (number_boxes, 2))
            ys = np.random.randint(left_top[1], right_bottom[1], (number_boxes, 2))

            xyxys = np.hstack((xs[:, 0].reshape(-1, 1), ys[:, 0].reshape(-1, 1), xs[:, 1].reshape(-1, 1), ys[:, 1].reshape(-1, 1)))

            if number_boxes > 1:
                left_index = np.random.randint(0, number_boxes - 1)
                right_index = np.random.randint(0, number_boxes - 1)
                top_index = np.random.randint(0, number_boxes - 1)
                bottom_index = np.random.randint(0, number_boxes - 1)
            else:
                left_index = 0
                right_index = 0
                top_index = 0
                bottom_index = 0

            xyxys[left_index, 0] = left_top[0]
            xyxys[right_index, 2] = right_bottom[0]
            xyxys[top_index, 1] = left_top[1]
            xyxys[bottom_index, 3] = right_bottom[1]

            bounding_boxes = BoundingBoxes2DArray(xyxys, mode=BoundingBoxes2DArray.XYXY)
            circumscribed_box = bounding_boxes.circumscribe()

            self.assertTrue(np.alltrue(circumscribed_box == apriori_known_circumscribed_box))


    def test_areas(self):
        for _ in range(self.number_checks):
            number_boxes = np.random.randint(1, 512)
            apriori_known_widths = np.random.randint(1, 2048, (number_boxes,))
            apriori_known_heights = np.random.randint(1, 2048, (number_boxes,))
            apriori_known_areas = apriori_known_widths * apriori_known_heights

            left_tops = np.random.randint(self.top_left_range[0], self.top_left_range[1], (number_boxes, 2))
            boxes = np.hstack((left_tops, apriori_known_widths.reshape(-1, 1), apriori_known_heights.reshape(-1, 1)))
            bounding_boxes = BoundingBoxes2DArray(boxes)
            areas = bounding_boxes.areas()

            self.assertTrue(np.alltrue(apriori_known_areas == areas))


    def test_mean_area(self):
        for _ in range(self.number_checks):
            number_boxes = np.random.randint(1, 512)
            apriori_known_widths = np.random.randint(1, 2048, (number_boxes,))
            apriori_known_heights = np.random.randint(1, 2048, (number_boxes,))
            apriori_known_areas = apriori_known_widths * apriori_known_heights
            apriori_known_mean_area = np.mean(apriori_known_areas)

            left_tops = np.random.randint(self.top_left_range[0], self.top_left_range[1], (number_boxes, 2))
            boxes = np.hstack((left_tops, apriori_known_widths.reshape(-1, 1), apriori_known_heights.reshape(-1, 1)))
            bounding_boxes = BoundingBoxes2DArray(boxes)
            mean_area = bounding_boxes.mean_area()

            self.assertEqual(apriori_known_mean_area, mean_area)

    def test_perimeters(self):
        for _ in range(self.number_checks):
            number_boxes = np.random.randint(1, 512)
            apriori_known_widths = np.random.randint(1, 2048, (number_boxes,))
            apriori_known_heights = np.random.randint(1, 2048, (number_boxes,))
            apriori_known_perimeter = 2 * (apriori_known_widths + apriori_known_heights)

            left_tops = np.random.randint(self.top_left_range[0], self.top_left_range[1], (number_boxes, 2))
            boxes = np.hstack((left_tops, apriori_known_widths.reshape(-1, 1), apriori_known_heights.reshape(-1, 1)))
            bounding_boxes = BoundingBoxes2DArray(boxes)
            perimeters = bounding_boxes.perimeters()

            self.assertTrue(np.alltrue(apriori_known_perimeter == perimeters))


    def test_xyxy_to_xywh(self):
        for _ in range(self.number_checks):
            number_boxes = np.random.randint(1, self.maximum_number_boxes)
            left_tops = np.random.randint(self.top_left_range[0], self.top_left_range[1], (number_boxes, 2))
            width_heights = np.random.randint(1, 500, (number_boxes, 2))

            right_bottoms = left_tops + width_heights
            left_bottoms = np.hstack((left_tops[:, 0].reshape(-1, 1), right_bottoms[:, 1].reshape(-1, 1)))
            right_tops = np.hstack((right_bottoms[:, 0].reshape(-1, 1), left_tops[:, 1].reshape(-1, 1)))

            boxes_left_top_width_height = np.hstack((left_tops, width_heights))

            boxes_top_left_bottom_right = np.hstack((left_tops, right_bottoms))
            boxes_bottom_right_top_left = np.hstack((right_bottoms, left_tops))
            boxes_bottom_left_top_right = np.hstack((left_bottoms, right_tops))
            boxes_top_right_bottom_left = np.hstack((right_tops, left_bottoms))

            boxes_xywh_converted_1 = BoundingBoxes2DArray.xyxy_to_xywh(boxes_top_left_bottom_right)
            boxes_xywh_converted_2 = BoundingBoxes2DArray.xyxy_to_xywh(boxes_bottom_left_top_right)
            boxes_xywh_converted_3 = BoundingBoxes2DArray.xyxy_to_xywh(boxes_bottom_right_top_left)
            boxes_xywh_converted_4 = BoundingBoxes2DArray.xyxy_to_xywh(boxes_top_right_bottom_left)

            self.assertTrue(np.alltrue(boxes_left_top_width_height == boxes_xywh_converted_1))
            self.assertTrue(np.alltrue(boxes_left_top_width_height == boxes_xywh_converted_2))
            self.assertTrue(np.alltrue(boxes_left_top_width_height == boxes_xywh_converted_3))
            self.assertTrue(np.alltrue(boxes_left_top_width_height == boxes_xywh_converted_4))


    def test_xywh_to_xyxy(self):
        ...
