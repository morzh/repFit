import cv2
import numpy as np
import unittest

from core.utils.cv.video_stride_reader import VideoStrideReader


class TestVideoStrideReader(unittest.TestCase):

    def setUp(self):
        """
        Description:
            Generates and saves small video with increasing per frames number sequence. In other words each frame depicts it's index.
        """
        self.frames_number = np.random.randint(100, 600)
        self.shape = (256, 256, 3)
        self.text_origin_point = (10, 160)
        self.test_video_filename = 'test_video_stride_reader.mp4'
        self.number_checks = 500

        cv2.VideoWriter()
        video_writer = cv2.VideoWriter(self.test_video_filename, cv2.VideoWriter_fourcc(*'MP4V'), 30.0, self.shape[:2])

        for frame_index in range(self.frames_number):
            current_frame = 255 * np.ones(self.shape, dtype=np.uint8)
            cv2.putText(current_frame, str(frame_index).zfill(4), self.text_origin_point, cv2.FONT_HERSHEY_SIMPLEX, 2.9, (0, 0, 0), 8)
            video_writer.write(current_frame)

        video_writer.release()
        print(f'Frames number: {self.frames_number}')


    def test_attributes_values(self):
        """

        """
        for check_index in range(self.number_checks):
            stride = np.random.randint(2, 11)
            video_reader = VideoStrideReader(self.test_video_filename, stride=stride)

            for _ in video_reader:
                ...

            stride_frames_number = (self.frames_number - 1) // stride - 1
            if (self.frames_number - 1) % stride == 0:
                stride_frames_number += 1

            self.assertEqual(self.frames_number - 1, video_reader.current_frame_index)


    def test_stride_visually(self):
        number_checks = 2
        for check_index in range(number_checks):
            stride = np.random.randint(2, 10)

            video_stride_reader = VideoStrideReader(self.test_video_filename, stride=stride)
            window_name = f'Test #{check_index + 1}; stride={stride}, number of frames is {self.frames_number}'

            for frame in video_stride_reader:
                cv2.imshow(window_name, frame)
                cv2.waitKey(-1)

            cv2.destroyAllWindows()
