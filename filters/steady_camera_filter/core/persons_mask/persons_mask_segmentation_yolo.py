import numpy as np
import torch
from ultralytics import YOLO
import cv2
from filters.steady_camera_filter.core.persons_mask.person_mask_base import PersonsMaskBase


class PersonsMaskSegmentationYolo(PersonsMaskBase):
    def __init__(self, **kwargs):
        self.detector = YOLO('yolov8m-seg.pt')
        self.detector_confidence: float = kwargs.get('confidence_threshold', 0.4)
        self._person_class_index: int = 0

    def pixel_mask(self, image: cv2.typing.MatLike, output_resolution: tuple[int, int]) -> cv2.typing.MatLike:
        prediction_result = self.detector.predict(image, verbose=False, retina_masks=True)

        current_person_masks_indices = torch.argwhere(prediction_result[0].boxes.conf > self.detector_confidence)
        current_person_classes_indices = torch.argwhere(prediction_result[0].boxes.cls == self._person_class_index)

        current_person_masks_indices = current_person_masks_indices.cpu().data.numpy().flatten()
        current_person_classes_indices = current_person_classes_indices.cpu().data.numpy().flatten()
        current_person_confident_indices = np.intersect1d(current_person_masks_indices, current_person_classes_indices)

        if prediction_result[0].masks is None:
            unified_mask = np.zeros(output_resolution)
        else:
            persons_masks = prediction_result[0].masks[current_person_confident_indices].cpu().data.numpy()
            unified_mask = np.sum(persons_masks, axis=0)
            unified_mask = np.clip(unified_mask, 0.0, 1.0)
            unified_mask = cv2.resize(unified_mask, output_resolution)

        return unified_mask
