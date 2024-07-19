import unittest

from filters.persons_tracking.core.person_id_track import PersonIdTrack


class TestPersonIdTrack(unittest.TestCase):

    def setUp(self):
        self.id_track = PersonIdTrack(0)


if __name__ == '__main__':
    unittest.main()
