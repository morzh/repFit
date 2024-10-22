import unittest
from core.utils.geometry.bounding_box_2d import BoundingBox2D


class TestBoundingBox(unittest.TestCase):

    def setUp(self):
        self.bounding_box = BoundingBox2D()

    def test_right_bottom(self):
        # self.assertEqual('foo'.upper(), 'FOO')
        pass

    def test_area(self):
        pass

    def test_circumscribe(self):
        bounding_box_2 = BoundingBox2D()
        self.bounding_box.circumscribe(bounding_box_2)


if __name__ == '__main__':
    unittest.main()
