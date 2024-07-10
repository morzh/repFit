import os
import numpy as np
import torch
from ultralytics import YOLO
import cv2
from filters.steady_camera_filter.core.persons_mask.persons_mask_base import PersonsMaskBase


class PersonsMaskYoloSegmentation(PersonsMaskBase):
    def __init__(self, **kwargs):
        """

        """
        weights_path = kwargs.get('weights_path', '')
        if not os.path.exists(weights_path):
            weights_path = ''

        model_type = kwargs.get('model_type', 'medium')

        match model_type:
            case 'nano':
                model_file = 'yolov8n-seg.pt'
            case 'small':
                model_file = 'yolov8s-seg.pt'
            case 'medium':
                model_file = 'yolov8m-seg.pt'
            case 'large':
                model_file = 'yolov8l-seg.pt'
            case _:
                raise ValueError("Models other than 'n', 's', 'm' or 'l' are not supported.")

        self.model = YOLO(os.path.join(weights_path, model_file))
        self.confidence: float = kwargs.get('confidence_threshold', 0.4)
        self._person_class_index: int = 0

    def pixel_mask(self, image: cv2.typing.MatLike, output_resolution: tuple[int, int]) -> cv2.typing.MatLike:
        prediction_result = self.model.predict(image, verbose=False, retina_masks=True)

        current_person_masks_indices = torch.argwhere(prediction_result[0].boxes.conf > self.confidence)
        current_person_classes_indices = torch.argwhere(prediction_result[0].boxes.cls == self._person_class_index)

        current_person_masks_indices = current_person_masks_indices.cpu().data.numpy().flatten()
        current_person_classes_indices = current_person_classes_indices.cpu().data.numpy().flatten()
        current_person_confident_indices = np.intersect1d(current_person_masks_indices, current_person_classes_indices)

        if prediction_result[0].masks is None:
            unified_mask = np.zeros((output_resolution[1], output_resolution[0]))
        else:
            persons_masks = prediction_result[0].masks[current_person_confident_indices].cpu().data.numpy()
            unified_mask = np.sum(persons_masks, axis=0)
            unified_mask = np.clip(unified_mask, 0.0, 1.0)

        return unified_mask
