import cv2
import numpy as np
import unittest

from core.utils.cv.video_reader import VideoReader


class TestVideoReader(unittest.TestCase):

    def setUp(self):
        """
        Description:
            Generates and saves small video with increasing per frames number sequence. In other words each frame depicts it's index.
        """
        self.frames_number = np.random.randint(300, 1200)
        self.shape = (256, 256, 3)
        self.text_origin_point = (10, 160)
        self.test_video_filename = 'test_video_reader.mp4'
        self.number_checks = 500

        cv2.VideoWriter()
        video_writer = cv2.VideoWriter(self.test_video_filename, cv2.VideoWriter_fourcc(*'MP4V'), 30.0, self.shape[:2])

        for frame_index in range(self.frames_number):
            current_frame = 255 * np.ones(self.shape, dtype=np.uint8)
            cv2.putText(current_frame, str(frame_index).zfill(4), self.text_origin_point, cv2.FONT_HERSHEY_SIMPLEX, 2.9, (0, 0, 0), 8)
            video_writer.write(current_frame)

        video_writer.release()
        print(f'Frames number: {self.frames_number}')


    def test_frame_index(self):
        video_reader = VideoReader(self.test_video_filename)
        for _ in video_reader:
            ...
        self.assertEqual(self.frames_number - 1, video_reader.current_frame_index)


    def test_visual(self):
        number_checks = 5
        for check_index in range(number_checks):
            stride = np.random.randint(1, 10)

            video_reader = VideoReader(self.test_video_filename, stride=stride)
            window_name = f'Test #{check_index + 1}; stride={stride}, number of frames is {self.frames_number}'

            for frame in video_reader:
                cv2.imshow(window_name, frame)
                cv2.waitKey(-1)

            cv2.destroyAllWindows()