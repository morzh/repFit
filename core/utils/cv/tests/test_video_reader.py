import unittest

import cv2
import numpy as np

from core.utils.cv.video_reader import VideoReader


class TestVideoReader(unittest.TestCase):

    def setUp(self):
        """
        Description:
            Generates and saves small video with increasing per frames number sequence. In other words each frame depicts it's index.
        """
        self.number_frames = np.random.randint(30, 300)
        self.shape = (256, 256, 3)
        self.text_origin_point = (10, 160)
        self.test_video_filename = 'test_video.mp4'

        cv2.VideoWriter()
        video_writer = cv2.VideoWriter(self.test_video_filename, cv2.VideoWriter_fourcc(*'MP4V'), 30.0, self.shape[:2])

        for frame_index in range(self.number_frames):
            current_frame = 255 * np.ones(self.shape, dtype=np.uint8)
            cv2.putText(current_frame, str(frame_index).zfill(4), self.text_origin_point, cv2.FONT_HERSHEY_SIMPLEX, 2.9, (0, 0, 0), 8)
            video_writer.write(current_frame)

        video_writer.release()

    def __del__(self):
        """

        """

    def test_stride(self):
        """

        """
        stride = 1
        # stride = np.random.randint(1, 10)

        video_reader = VideoReader(self.test_video_filename, stride=stride)

        for _ in video_reader:
            ...

        self.assertEqual(self.number_frames, video_reader.current_frame_index)
        # self.assertEqual(self.number_frames, video_reader.)


    def test_stride_visually(self):
        stride = np.random.randint(1, 10)

        video_reader = VideoReader(self.test_video_filename, stride=stride)
        window_name = f'Stride={stride}, number of frames is {self.number_frames}'

        for frame in video_reader:
            cv2.imshow(window_name, frame)
            cv2.waitKey(-1)