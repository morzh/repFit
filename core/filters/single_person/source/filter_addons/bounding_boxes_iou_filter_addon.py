import numpy as np
from ordered_set import OrderedSet

from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks
from core.utils.geometry.bounding_boxes.bounding_boxes_2d_array import BoundingBoxes2DArray


class BoundingBoxesIouFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        This is inter-person filter  which filters out frames indices of partial and full body data with IoU greater than the given threshold.
        Person's bounding box confidence is also taken into account using respective threshold value (bounding box with low confidence are not considered).
        This filter add-on should be applied as one of the final filter add-ons in filtering chain.

    :ivar iou_threshold: IoU threshold.
    :ivar confidence_threshold: confidence threshold.
    """
    def __init__(self, iou_threshold=0.4, confidence_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold


    def process(self, tracks: MultiplePersonsTracks) -> None:
        persons_ids_list = tracks.persons.keys()

        for person_reference_id_index, person_reference_id in enumerate(persons_ids_list[:-2]):
            current_reference_track = tracks.persons[person_reference_id]
            if not len(current_reference_track.tracked_data) or not current_reference_track.is_active: continue
            if not len(current_reference_track.partial_body_data.frames_indices) and len(current_reference_track.full_body_data.frames_indices): continue

            for person_target_id in persons_ids_list[person_reference_id_index + 1:]:
                current_target_track = tracks.persons[person_target_id]
                if not len(current_target_track.tracked_data) or not current_target_track.is_active: continue
                if not len(current_target_track.partial_body_data.frames_indices) and len(current_target_track.full_body_data.frames_indices): continue

                # unify partial and full body indices
                current_reference_body_frames_indices = current_reference_track.full_body_data.frames_indices + current_reference_track.partial_body_data.frames_indices
                current_target_body_frames_indices = current_target_track.full_body_data.frames_indices + current_target_track.partial_body_data.frames_indices

                # find tracked frames indices with confidence greater than given threshold for reference person
                current_reference_frames_indices_set = OrderedSet(current_reference_body_frames_indices.values)
                current_reference_body_frames_indices_keys = current_reference_frames_indices_set.index(current_reference_body_frames_indices.values)
                current_reference_confidences = current_reference_track.tracked_data.confidences[current_reference_body_frames_indices_keys]
                current_reference_confidences_mask = current_reference_confidences > self.confidence_threshold
                current_reference_body_frames_confident_indices = current_reference_body_frames_indices_keys[current_reference_confidences_mask]

                # find tracked frames indices with confidence greater than given threshold for target person
                current_target_frames_indices_set = OrderedSet(current_target_body_frames_indices.values)
                current_target_body_frames_indices_keys = current_target_frames_indices_set.index(current_target_body_frames_indices.values)
                current_target_confidences = current_target_track.tracked_data.confidences[current_target_body_frames_indices_keys]
                current_target_confidences_mask = current_target_confidences > self.confidence_threshold
                current_target_body_frames_confident_indices = current_target_body_frames_indices_keys[current_target_confidences_mask]

                # find frames indices which are common for reference and target body and full body data
                current_common_reference_target_frames_indices = np.intersect1d(current_reference_body_frames_confident_indices,
                                                                                current_target_body_frames_confident_indices)
                current_reference_body_frames_indices_keys = current_reference_frames_indices_set.index(current_common_reference_target_frames_indices)
                current_target_body_frames_indices_keys = current_target_frames_indices_set.index(current_common_reference_target_frames_indices)

                current_reference_bounding_boxes = current_reference_track.tracked_data.bounding_boxes[current_reference_body_frames_indices_keys]
                current_target_bounding_boxes = current_reference_track.tracked_data.bounding_boxes[current_target_body_frames_indices_keys]
                current_bounding_boxes_iou = BoundingBoxes2DArray.intersection_over_union(current_reference_bounding_boxes, current_target_bounding_boxes)
                current_iou_above_threshold_mask =  current_bounding_boxes_iou > self.iou_threshold

                current_common_reference_target_frames_indices_filtered = current_common_reference_target_frames_indices[current_iou_above_threshold_mask]

                if not np.all(current_iou_above_threshold_mask == False):
                    current_reference_track.partial_body_data.frames_indices -= current_common_reference_target_frames_indices_filtered
                    current_reference_track.full_body_data.frames_indices -= current_common_reference_target_frames_indices_filtered

                    current_target_track.partial_body_data.frames_indices -= current_common_reference_target_frames_indices_filtered
                    current_target_track.full_body_data.frames_indices -= current_common_reference_target_frames_indices_filtered
