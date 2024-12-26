import copy
import unittest
import numpy as np


class TestConfidenceFilterAddon(unittest.TestCase):

    def setUp(self):
        self.number_checks = 1_500