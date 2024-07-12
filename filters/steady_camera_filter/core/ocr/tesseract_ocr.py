import cv2
import numpy as np
import pytesseract

from filters.steady_camera_filter.core.ocr.ocr_base import OcrBase


class TesseractOcr(OcrBase):
    """
    Description:
        Google Tesseract text recognition engine for text masking
    """
    alias = 'tesseract'

    def __init__(self, **kwargs):
        """
        Description:
            Google Tesseract OCR class constructor.

        :key confidence_threshold: minimum confidence for detected text, value should be in [0, 1] segment

        :return: __init__() should return None
        """

        super().__init__(**kwargs)
        self.confidence_threshold = 100 * kwargs.get('confidence_threshold', 0.1)

    def pixel_mask(self, image: cv2.typing.MatLike, output_resolution: tuple[int, int]) -> cv2.typing.MatLike:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        number_boxes = len(data['level'])
        image_resolution = (image.shape[0], image.shape[1])
        mask = np.zeros(image_resolution, dtype=np.float32)
        for i in range(number_boxes):
            if data['conf'][i] > self.confidence_threshold:
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                mask = cv2.rectangle(mask, (x, y), (x + w, y + h), 1, -1)
        mask = cv2.resize(mask, output_resolution)

        return mask
