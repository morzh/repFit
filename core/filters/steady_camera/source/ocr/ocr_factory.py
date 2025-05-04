from core.filters.steady_camera.source.factory import Factory
from core.filters.steady_camera.source.ocr.craft import Craft
from core.filters.steady_camera.source.ocr.easy_ocr import EasyOcr
from core.filters.steady_camera.source.ocr.tesseract_ocr import TesseractOcr

factory = Factory()

factory.register_builder(Craft.alias, Craft.create_instance)
factory.register_builder(EasyOcr.alias, EasyOcr.create_instance)
factory.register_builder(TesseractOcr.alias, TesseractOcr.create_instance)
