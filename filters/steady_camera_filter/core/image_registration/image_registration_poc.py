import warnings
from dataclasses import dataclass
from collections import deque
import numpy as np
import cv2
from scipy import ndimage
from numpy.typing import NDArray
from typing import Annotated, Literal, Optional
import matplotlib.pyplot as plt

ndimageNxMx3 = Annotated[NDArray[np.uint8], Literal["N", "M", 3]]
ndimageNxM = Annotated[NDArray[np.uint8], Literal["N", "M"]]
ndimageNxMf = Annotated[NDArray[np.float32], Literal["N", "M"]]
ndimageNxMc = Annotated[NDArray[np.csingle], Literal["N", "M"]]


@dataclass(frozen=True, slots=True)
class PhaseCorrelationResult:
    """
    Data storage for ImageRegistrationPoc.register_images() and ImageRegistrationPoc.register() return value
    """
    shift: tuple[int, int] = (0, 0)
    peak_value: float = 0


class ImageSequenceRegistrationPoc:
    """
    Images sequence registration using phase only correlation (POC) approach.
    As wee need to cache FFTs ...... bla bla bla
    """
    def __init__(self, images_resolution, windowing_function=np.hamming):
        self.fft_deque = deque(maxlen=2)
        self.reference_fft: Optional[ndimageNxM] = None
        self.target_fft: Optional[ndimageNxM] = None
        self.windowing = self.image_windowing(images_resolution, windowing_function=windowing_function)

    def update_deque(self, new_image: ndimageNxM) -> None:
        # windowing and FFT
        new_image_fft = np.fft.fft2(new_image * self.windowing)
        # Shift the zero-frequency component to the center of the spectrum.
        new_image_fft = np.fft.fftshift(new_image_fft)
        self.fft_deque.append(new_image_fft)

    def cross_power_spectrum(self, image_reference: ndimageNxM, image_target: ndimageNxM) -> ndimageNxMc:
        """
        Cross power spectrum of reference and target images
        @param image_reference: reference grayscale image
        @param image_target: target grayscale image
        @return: values of cross power spectrum
        """
        assert image_target.shape == image_reference.shape
        assert len(image_reference.shape) == 2
        assert len(image_target.shape) == 2

        fft_frame_1 = np.fft.fft2(image_target * self.windowing)
        fft_frame_2 = np.fft.fft2(image_reference * self.windowing)
        fft_frame_1 = np.fft.fftshift(fft_frame_1)
        fft_frame_2 = np.fft.fftshift(fft_frame_2)
        cross_power_spectrum = fft_frame_1 * np.conjugate(fft_frame_2)
        # absolute_cross_power_spectrum = np.absolute(cross_power_spectrum)
        # if np.all(absolute_cross_power_spectrum) > 0:
        #     cross_power_spectrum /= absolute_cross_power_spectrum
        return cross_power_spectrum

    def register(self) -> PhaseCorrelationResult:
        if len(self.fft_deque) != 2:
            warnings.warn('Only one image provided for registration. '
                          'Add new image using ImageSequenceRegistrationPoc.update_deque() method')
            return PhaseCorrelationResult(shift=(0, 0), peak_value=-1)

        if self.fft_deque[0].shape != self.fft_deque[1].shape:
            raise ValueError('All images in deque should have the same shape')

        cross_power_spectrum = self.fft_deque[0] * np.ma.conjugate(self.fft_deque[1])
        poc_result = self.registration_result_from_cross_power_spectrum(cross_power_spectrum)
        return poc_result

    def registration_result_from_cross_power_spectrum(self, cross_power_spectrum) -> PhaseCorrelationResult:
        absolute_cross_power_spectrum = np.absolute(cross_power_spectrum)
        if np.all(absolute_cross_power_spectrum) > 0:
            cross_power_spectrum /= absolute_cross_power_spectrum

        cross_correlation = np.fft.ifft2(cross_power_spectrum).real
        pixel_shift = np.unravel_index(cross_correlation.argmax(), cross_correlation.shape)  # [row, column] format
        peak_value = float(cross_correlation[pixel_shift])
        pixel_shift = np.array(pixel_shift)  # [row, column] format

        #  negative shifts registration
        if pixel_shift[0] > self.fft_deque[0].shape[0] / 2:
            pixel_shift[0] = self.fft_deque[0].shape[0] - 1 - pixel_shift[0]
        if pixel_shift[1] > self.fft_deque[0].shape[1] / 2:
            pixel_shift[1] = self.fft_deque[0].shape[1] - 1 - pixel_shift[1]

        pixel_shift = (int(pixel_shift[1]), int(pixel_shift[0]))  # convert [row, column] to (X, Y) format
        return PhaseCorrelationResult(shift=pixel_shift, peak_value=peak_value)

    def register_images(self, image_reference: ndimageNxMx3 | ndimageNxM, image_target: ndimageNxMx3 | ndimageNxM) -> PhaseCorrelationResult:
        """
        Register images using POC
        @param image_reference:  reference image
        @param image_target: target image
        @return:  images registration result
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
        Plot current windowing 2D function values
        @return: None
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
        https://en.wikipedia.org/wiki/Window_function
        What should be considered when selecting a windowing function when smoothing a time series:
        https://dsp.stackexchange.com/questions/208/what-should-be-considered-when-selecting-a-windowing-function-when-smoothing-a-t
        @param shape: shape of 2D windowing function values in (width, height) format
        @param windowing_function: windowing function (Hamming by default)
        @return: 2D windowing function values
        """
        window_in_rows = windowing_function(shape[1])
        window_in_rows = np.clip(window_in_rows, 1e-6, 1.0)

        window_in_columns = windowing_function(shape[0])
        window_in_columns = np.clip(window_in_columns, 1e-6, 1.0)

        return np.sqrt(np.outer(window_in_rows, window_in_columns))
