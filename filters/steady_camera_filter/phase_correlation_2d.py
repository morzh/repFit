from dataclasses import dataclass

import numpy as np
import cv2
from numpy.typing import NDArray
from typing import Annotated, Literal, TypeVar, Optional
import matplotlib.pyplot as plt

ndimageNxMx3 = Annotated[NDArray[np.uint8], Literal["N", "M", 3]]
ndimageNxM = Annotated[NDArray[np.uint8], Literal["N", "M"]]
ndimageNxMf = Annotated[NDArray[np.float32], Literal["N", "M"]]
ndimageNxMc = Annotated[NDArray[np.csingle], Literal["N", "M"]]


@dataclass(frozen=True, slots=True)
class PhaseCorrelationResult:
    """
    Data storage for ImageRegistrationPoc.register_images() return value
    """
    shift: tuple[int, int] = (0, 0)
    peak_value: float = 0
    cross_correlation_mean: float = 0
    cross_correlation_std: float = 0


class ImageRegistrationPoc:
    """
    Image registration using phase only correlation (POC) approach
    """
    def __init__(self):
        self.windowing: Optional[np.ndarray] = None

    @staticmethod
    def highpass_box_filter(image: ndimageNxMx3, filter_size: int = 10) -> ndimageNxMf:
        """
        Simple image high pass filter using box filter for image smoothing
        @param image: input image
        @param filter_size: box filter kernel size (squared)
        @return: filtered image
        """
        image = image.mean(axis=2)  # make color image grey
        image_highpass = image - cv2.boxFilter(image, ddepth=0, ksize=(filter_size, filter_size))
        return image_highpass

    def cross_power_spectrum(self, image_reference: ndimageNxMx3 | ndimageNxM, image_target: ndimageNxMx3 | ndimageNxM) -> ndimageNxMc:
        """
        Cross power spectrum of reference and target images
        @param image_reference: reference image
        @param image_target: target image
        @return: values of cross power spectrum
        """
        assert image_target.shape == image_reference.shape

        if len(image_target.shape) == 3:
            image_target = image_target.mean(axis=2)
        if len(image_reference.shape) == 3:
            image_reference = image_reference.mean(axis=2)

        fft_frame_1 = np.fft.fft2(image_target * self.windowing)
        fft_frame_2 = np.fft.fft2(image_reference * self.windowing)
        fft_frame_1 = np.fft.fftshift(fft_frame_1)
        fft_frame_2 = np.fft.fftshift(fft_frame_2)
        cross_power_spectrum = fft_frame_1 * np.conjugate(fft_frame_2)
        absolute_cross_power_spectrum = np.absolute(cross_power_spectrum)
        if np.all(absolute_cross_power_spectrum) > 0:
            cross_power_spectrum /= absolute_cross_power_spectrum
        '''
        if plot_windowed_images:
            plt.figure(figsize=(25, 20))
            plt.subplot(121)
            plt.imshow(frame_target * self.windowing)
            plt.subplot(122)
            plt.imshow(frame_reference * self.windowing)
            plt.tight_layout()
            plt.show()
        '''

        return cross_power_spectrum

    def register_images(self, image_reference: ndimageNxMx3 | ndimageNxM, image_target: ndimageNxMx3 | ndimageNxM) -> PhaseCorrelationResult:
        """
        Register images using POC
        @param image_reference:  reference image
        @param image_target: target image
        @return:  images registration result
        """
        assert image_target.shape == image_reference.shape
        self.windowing = self.windowing_2d(image_target.shape[:2], np.blackman)
        cross_correlation = np.fft.ifft2(self.cross_power_spectrum(image_reference, image_target))
        cross_correlation = np.real(cross_correlation)
        pixel_shift = np.unravel_index(cross_correlation.argmax(), cross_correlation.shape)
        pixel_shift = tuple(pixel_shift)  # [row, column] format
        peak_value = float(cross_correlation[pixel_shift])
        pixel_shift = (pixel_shift[1], pixel_shift[0])  # (X, Y) format
        mean = np.mean(cross_correlation)
        std = np.std(cross_correlation)
        return PhaseCorrelationResult(shift=pixel_shift, peak_value=peak_value, cross_correlation_mean=mean, cross_correlation_std=std)

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
    def windowing_2d(shape: tuple[int, int] = (512, 512), windowing_function: callable = np.hamming) -> cv2.typing.MatLike:
        """
        https://en.wikipedia.org/wiki/Window_function
        What should be considered when selecting a windowing function when smoothing a time series:
        https://dsp.stackexchange.com/questions/208/what-should-be-considered-when-selecting-a-windowing-function-when-smoothing-a-t
        @param shape: shape of 2D windowing function values
        @param windowing_function: windowing function (Hamming by default)
        @return: 2D windowing function values
        """
        window_1 = windowing_function(shape[0])
        window_1 = np.clip(window_1, 1e-6, 1.0)
        window_2 = windowing_function(shape[1])
        window_2 = np.clip(window_2, 1e-6, 1.0)
        return np.sqrt(np.outer(window_1, window_2))
