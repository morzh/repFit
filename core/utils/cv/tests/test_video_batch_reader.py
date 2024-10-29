import cv2
import numpy as np
import unittest

from core.utils.cv.video_reader_frames_batch import VideoReaderFramesBatch

class TestVideoReaderFramesBatch(unittest.TestCase):

    def setUp(self):
        self.frames_number = np.random.randint(30, 2000)
        self.shape = (256, 256, 3)
        self.text_origin_point = (10, 150)
        self.test_video_filename = 'test_video_reader_frames_batch.mp4'
        self.maximum_mosaic_resolution = (1500, 2460)
        self.write_test_video()

    def test_attributes_exceptions(self):
        """
        Description:
            VideoReaderFramesBatch class derived from VideoReader, but in VideoReaderFramesBatch we do not need
            current_stride_frame_index and stride attributes to be presented.
        """
        print(f'Frames number is {self.frames_number}')
        batch_size = np.random.randint(3, 18)
        video_reader_frames_batch = VideoReaderFramesBatch(self.test_video_filename, batch_size=batch_size)

        for _ in video_reader_frames_batch:
            ...

        with self.assertRaises(AttributeError):
            _ = video_reader_frames_batch.current_stride_frame_index
            _ = video_reader_frames_batch.stride


    def test_attributes_values(self):
        print(f'Frames number is {self.frames_number}')
        number_checks = 150
        for _ in range(number_checks):
            current_batch_size = np.random.randint(3, 18)
            current_number_batches = self.frames_number // current_batch_size + min((self.frames_number - 1) % current_batch_size, 1)  - 1
            video_reader_frames_batch = VideoReaderFramesBatch(self.test_video_filename, batch_size=current_batch_size)

            for _ in video_reader_frames_batch:
                ...

            self.assertEqual(self.frames_number - 1, video_reader_frames_batch.current_source_frame_index)
            self.assertEqual(current_batch_size, video_reader_frames_batch.batch_size)
            self.assertEqual(current_number_batches, video_reader_frames_batch.current_batch_index)


    def test_batches_visually(self):
        """
        Description:
            Check frames, presented in each frames batch visually
        """
        number_checks = 5
        for check_index in range(number_checks):
            batch_size = np.random.randint(5, 50)
            video_reader_frames_batch = VideoReaderFramesBatch(self.test_video_filename, batch_size=batch_size)
            window_name = f'Test #{check_index + 1}; number of frames is {self.frames_number}'

            for frames_batch in video_reader_frames_batch:
                frames_mosaic = self.images_grid_from_frames_batch(frames_batch)
                cv2.imshow(window_name, frames_mosaic)
                cv2.waitKey(-1)

            cv2.destroyAllWindows()

    def images_grid_from_frames_batch(self, frames_batch) -> cv2.typing.MatLike:
        """
        Description:
            Makes big image which is grid of images from ``frames_batch``. For visualization purposes.

        :return: grid image
        """
        batch_size = frames_batch.shape[0]
        mosaic_width_cells = int(np.ceil(np.sqrt(frames_batch.shape[0])))
        mosaic_height_cells = int(np.ceil(frames_batch.shape[0] / mosaic_width_cells))
        mosaic_number_channels = frames_batch.shape[-1] if len(frames_batch.shape) == 4 else 1

        frame_width = frames_batch.shape[1]
        frame_height = frames_batch.shape[2]

        mosaic_image = np.zeros((mosaic_height_cells * frame_height, mosaic_width_cells * frame_width, mosaic_number_channels), dtype=np.uint8)

        frames_batch_index = 0
        for column_index in range(mosaic_height_cells):
            for row_index in range(mosaic_width_cells):
                mosaic_image[frame_width * column_index: frame_width * (column_index + 1), frame_height * row_index: frame_height * (row_index + 1)] = frames_batch[frames_batch_index]
                frames_batch_index += 1
                if frames_batch_index == batch_size:
                    break

        scale_factor = 1.0
        if mosaic_image.shape[0] > self.maximum_mosaic_resolution[0]:
            scale_factor = self.maximum_mosaic_resolution[0] / mosaic_image.shape[0]
        if mosaic_image.shape[1] > self.maximum_mosaic_resolution[1]:
            scale_factor = self.maximum_mosaic_resolution[1] / mosaic_image.shape[1]

        mosaic_image = cv2.resize(mosaic_image, None, fx=scale_factor, fy=scale_factor, interpolation= cv2.INTER_LINEAR)

        return mosaic_image


    def write_test_video(self):
        """
        Description:
            Generates and saves small video with increasing per frames number sequence. In other words each frame depicts it's index.
        """
        video_writer = cv2.VideoWriter(self.test_video_filename, cv2.VideoWriter_fourcc(*'MP4V'), 30.0, self.shape[:2])
        for frame_index in range(self.frames_number):
            current_frame = 255 * np.ones(self.shape, dtype=np.uint8)
            cv2.putText(current_frame, str(frame_index).zfill(4), self.text_origin_point, cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 5)
            video_writer.write(current_frame)

        video_writer.release()
