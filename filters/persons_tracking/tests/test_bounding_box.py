import unittest
from filters.persons_tracking.core.bounding_box import BoundingBox


class TestBoundingBox(unittest.TestCase):

    def setUp(self):
        self.bounding_box = BoundingBox()

    def test_right_bottom(self):
        # self.assertEqual('foo'.upper(), 'FOO')
        pass

    def test_area(self):
        pass

    def test_circumscribe(self):
        bounding_box_2 = BoundingBox()
        self.bounding_box.circumscribe(bounding_box_2)


if __name__ == '__main__':
    unittest.main()
