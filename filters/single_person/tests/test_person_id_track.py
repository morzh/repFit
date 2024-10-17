import unittest

from filters.single_person.core.single_person_track import SinglePersonTrack


class TestPersonIdTrack(unittest.TestCase):

    def setUp(self):
        self.id_track = SinglePersonTrack(0)


if __name__ == '__main__':
    unittest.main()
