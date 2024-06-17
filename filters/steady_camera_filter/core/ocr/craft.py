import numpy as np
from CRAFT import CRAFTModel, draw_polygons
import cv2
from cv2 import typing
from filters.steady_camera_filter.core.ocr.ocr_base import OcrBase

from easyocr.craft import CRAFT
craft_weights_folder = '/home/anton/work/fitMate/repFit/3rd_party/weights/craft'


class Craft(OcrBase):
    """
    CRAFT: Character-Region Awareness For Text detection
        https://github.com/fcakyon/craft-text-detector
    arXiv:
        https://arxiv.org/abs/1904.01941
    arXiv PDF:
        https://arxiv.org/pdf/1904.01941
    """
    def __init__(self, use_cuda=True, use_refiner=False, use_fp16=False):
        if use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.craft = CRAFTModel(craft_weights_folder, self.device, use_refiner=use_refiner, fp16=use_fp16)

    def pixel_mask(self, image, output_resolution) -> typing.MatLike:
        if len(image.shape) == 2:
            image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)
        image_dimensions = image.shape[:2]
        polygons = self.craft.get_polygons(image)
        mask = self.draw_polygons(image_dimensions, polygons)
        mask = cv2.resize(mask, output_resolution)
        return mask

    @staticmethod
    def draw_polygons(image_shape: tuple[int, int], polygons: list[list[list[int]]]) -> np.ndarray:
        mask = np.zeros(image_shape)
        for i, poly in enumerate(polygons):
            poly_ = np.array(poly).astype(np.int32).reshape((-1))
            poly_ = poly_.reshape(-1, 2)
            mask = cv2.fillPoly(mask, [poly_.reshape((-1, 1, 2))], color=(1, 1, 1))
        return mask
