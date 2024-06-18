from ocr_base import OcrBase
import pytesseract


class TesseractOcr(OcrBase):
    """
    Google Tesseract text recognition
    """
    def pixel_mask(self, image, output_resolution):
        raise NotImplementedError('Text mask is not implemented for tesseract OCR yet.')
