from filters.steady_camera_filter.core.ocr.ocr_factory import OcrFactory
from filters.steady_camera_filter.core.ocr.craft import Craft, create_craft_instance
from filters.steady_camera_filter.core.ocr.easy_ocr import EasyOcr, create_easy_ocr_instance
from filters.steady_camera_filter.core.ocr.tesseract_ocr import TesseractOcr, create_tesseract_instance

factory = OcrFactory()

factory.register_builder(Craft.alias, create_craft_instance)
factory.register_builder(EasyOcr.alias, create_easy_ocr_instance)
factory.register_builder(TesseractOcr.alias, create_tesseract_instance)
