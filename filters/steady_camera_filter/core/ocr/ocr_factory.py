from filters.steady_camera_filter.core.factory import Factory
from filters.steady_camera_filter.core.ocr.craft import Craft
from filters.steady_camera_filter.core.ocr.easy_ocr import EasyOcr
from filters.steady_camera_filter.core.ocr.tesseract_ocr import TesseractOcr

factory = Factory()

factory.register_builder(Craft.alias, Craft.create_instance)
factory.register_builder(EasyOcr.alias, EasyOcr.create_instance)
factory.register_builder(TesseractOcr.alias, TesseractOcr.create_instance)
