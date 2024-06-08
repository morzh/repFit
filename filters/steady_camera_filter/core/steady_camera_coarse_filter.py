import matplotlib.pyplot as plt
import numpy as np
from cv_utils.video_frames_batch import VideoFramesBatch
from filters.steady_camera_filter.core.image_registration_poc import ImageSequenceRegistrationPoc


class SteadyCameraCoarseFilter:
    def __init__(self, video_filepath: str, frames_to_average=20):
        self.video_frames_batch = VideoFramesBatch(video_filepath, frames_to_average)
        self.poc = ImageSequenceRegistrationPoc(self.video_frames_batch.video_reader.resolution)
        self.steady_camera_frames_ranges: list[range] = []
        self.registration_results: list = []

    def process(self, number_frames_to_average=15):
        self.video_frames_batch.batch_size = number_frames_to_average
        image_frames_batch = next(self.video_frames_batch)
        averaged_frames = np.mean(image_frames_batch, axis=(0, 3))
        self.poc.update_deque(averaged_frames)

        if __debug__:
            reference_image = averaged_frames

        while True:
            current_image_frames_batch = next(self.video_frames_batch)
            current_averaged_frames = np.mean(current_image_frames_batch, axis=(0, 3))
            self.poc.update_deque(current_averaged_frames)
            current_registration_result = self.poc.register()
            self.registration_results.append(current_registration_result)

            if __debug__:
                target_image = current_averaged_frames
                plt.figure(figsize=(25, 15))
                plt.suptitle(str(current_registration_result.shift) + ' ' + str(current_registration_result.peak_value))
                plt.subplot(121)
                plt.imshow(reference_image, cmap='gray')
                plt.subplot(122)
                plt.imshow(target_image, cmap='gray')
                plt.tight_layout()
                plt.show()
                reference_image = target_image


