import warnings
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True, slots=True)
class ImageRegistrationResult:
    """
    Data storage for ImageRegistration classes registration results
    """
    shift: tuple[int, int] = (0, 0)
    confidence: float = 0


class ImageSequenceRegistrationPoc:
    """
    Images sequence registration using phase only correlation (POC) approach.
    https://en.wikipedia.org/wiki/Phase_correlation
    """
    def __init__(self, images_resolution: tuple[int, int], windowing_function=np.hamming):
        """
        :param images_resolution: images resolution (height, width)
        :param windowing_function: windowing function for image FFT
        """
        self.fft_deque = deque(maxlen=2)
        self.reference_fft: Optional[cv2.typing.MatLike] = None
        self.target_fft: Optional[cv2.typing.MatLike] = None
        self.windowing = self.image_windowing(images_resolution, windowing_function=windowing_function)

    def update_deque(self, new_image: cv2.typing.MatLike) -> None:
        """
        Description:
            Update filter's double queue (deque) with new image. Not, queue has fixed size, adding new image causes deleting the oldest one
        :param new_image: image to add to deque
        """
        # windowing and FFT
        new_image_fft = np.fft.fft2(new_image * self.windowing)
        # Shift the zero-frequency component to the center of the spectrum.
        new_image_fft = np.fft.fftshift(new_image_fft)
        self.fft_deque.append(new_image_fft)

    def cross_power_spectrum(self, image_reference: cv2.typing.MatLike, image_target: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """
        Description:
            Cross power spectrum of reference and target images (from current deque)
        :param image_reference: reference grayscale image
        :param image_target: target grayscale image
        :return: values of cross power spectrum
        """
        assert image_target.shape == image_reference.shape
        assert len(image_reference.shape) == 2
        assert len(image_target.shape) == 2

        fft_frame_1 = np.fft.fft2(image_target * self.windowing)
        fft_frame_2 = np.fft.fft2(image_reference * self.windowing)
        fft_frame_1 = np.fft.fftshift(fft_frame_1)
        fft_frame_2 = np.fft.fftshift(fft_frame_2)
        cross_power_spectrum = fft_frame_1 * np.conjugate(fft_frame_2)
        return cross_power_spectrum

    def register(self) -> ImageRegistrationResult:
        """
        Description:
            Register two current images (strictly speaking their FFTs) in deque.
        :return: registration result
        """
        if len(self.fft_deque) != 2:
            warnings.warn('Only one image provided for registration. '
                          'Add new image using ImageSequenceRegistrationPoc.update_deque() method')
            return ImageRegistrationResult(shift=(0, 0), confidence=-1)

        if self.fft_deque[0].shape != self.fft_deque[1].shape:
            raise ValueError('All images in deque should have the same shape')

        cross_power_spectrum = self.fft_deque[0] * np.ma.conjugate(self.fft_deque[1])
        registration_result = self.registration_result_from_cross_power_spectrum(cross_power_spectrum)
        return registration_result

    def registration_result_from_cross_power_spectrum(self, cross_power_spectrum) -> ImageRegistrationResult:
        """
        Description:
            Calculate pixel shift in images registration from cross power spectrum.
        :return: registration result
        """
        absolute_cross_power_spectrum = np.absolute(cross_power_spectrum)
        if np.all(absolute_cross_power_spectrum) > 0:
            cross_power_spectrum /= absolute_cross_power_spectrum

        cross_correlation = np.fft.ifft2(cross_power_spectrum).real
        pixel_shift = np.unravel_index(cross_correlation.argmax(), cross_correlation.shape)  # [row, column] format
        peak_value = float(cross_correlation[pixel_shift])
        pixel_shift = np.array(pixel_shift)  # [row, column] format

        #  negative pixel shift case
        if pixel_shift[0] > self.fft_deque[0].shape[0] / 2:
            pixel_shift[0] = self.fft_deque[0].shape[0] - 1 - pixel_shift[0]
        if pixel_shift[1] > self.fft_deque[0].shape[1] / 2:
            pixel_shift[1] = self.fft_deque[0].shape[1] - 1 - pixel_shift[1]

        pixel_shift = (int(pixel_shift[1]), int(pixel_shift[0]))  # convert [row, column] to (X, Y) format
        return ImageRegistrationResult(shift=pixel_shift, confidence=peak_value)

    def register_images(self, image_reference: cv2.typing.MatLike, image_target: cv2.typing.MatLike) -> ImageRegistrationResult:
        """
        Description:
            Register images using phase only correlation
        :param image_reference:  reference image
        :param image_target: target image
        :return: images registration result
        """
        assert image_target.shape == image_reference.shape

        if len(image_reference.shape) == 3:
            image_reference = image_reference.mean(axis=2)
        if len(image_target.shape) == 3:
            image_target = image_target.mean(axis=2)

        assert image_target.shape == self.windowing.shape
        assert image_reference.shape == self.windowing.shape

        cross_power_spectrum = self.cross_power_spectrum(image_reference, image_target)
        return self.registration_result_from_cross_power_spectrum(cross_power_spectrum)

    def plot_windowing_2d(self) -> None:
        """
        Description:
            Plot current windowing 2D function values
        """
        x = np.linspace(0, 1.5, 51)
        y = np.linspace(0, 1.5, 51)
        grid_2d_x, grid_2d_y = np.meshgrid(x, y)
        z = self.windowing
        # fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(grid_2d_x, grid_2d_y, z, 50, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    @staticmethod
    def image_windowing(shape: tuple[int, int] = (512, 512), windowing_function: callable = np.hamming) -> cv2.typing.MatLike:
        """
        Description:
            https://en.wikipedia.org/wiki/Window_function
            What should be considered when selecting a windowing function when smoothing a time series:
            https://dsp.stackexchange.com/questions/208/what-should-be-considered-when-selecting-a-windowing-function-when-smoothing-a-t
        :param shape: shape of 2D windowing function values in (width, height) format
        :param windowing_function: windowing function (Hamming by default)
        :return: 2D windowing function values
        """
        window_in_rows = windowing_function(shape[1])
        window_in_rows = np.clip(window_in_rows, 1e-6, 1.0)

        window_in_columns = windowing_function(shape[0])
        window_in_columns = np.clip(window_in_columns, 1e-6, 1.0)

        return np.sqrt(np.outer(window_in_rows, window_in_columns))
