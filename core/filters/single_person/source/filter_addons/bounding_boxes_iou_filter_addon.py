import numpy as np
from ordered_set import OrderedSet

from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks
from core.utils.geometry.bounding_boxes.bounding_boxes_2d_array import BoundingBoxes2DArray


class BoundingBoxesIouFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter every frame's and person bounding box if .....
        This filter add-on should be applied as the final filter add-on in filtering chain.

    :ivar iou_threshold: duration threshold in seconds.
    """
    def __init__(self, iou_threshold=0.4):
        self.iou_threshold = iou_threshold


    def process(self, tracks: MultiplePersonsTracks, **kwargs) -> None:
        persons_ids_list = tracks.persons.keys()

        for person_reference_id_index, person_reference_id in enumerate(persons_ids_list[:-2]):
            current_reference_track = tracks.persons[person_reference_id]
            if not len(current_reference_track.tracked_data) or not current_reference_track.is_active: continue
            if not len(current_reference_track.body_data.frames_indices) and len(current_reference_track.full_body_data.frames_indices): continue

            for person_target_id in persons_ids_list[person_reference_id_index + 1:]:
                current_target_track = tracks.persons[person_target_id]
                if not len(current_target_track.tracked_data) or not current_target_track.is_active: continue
                if not len(current_target_track.body_data.frames_indices) and len(current_target_track.full_body_data.frames_indices): continue

                current_reference_body_unified_frames_indices = current_reference_track.full_body_data.frames_indices + current_reference_track.body_data.frames_indices
                current_target_body_unified_frames_indices = current_target_track.full_body_data.frames_indices + current_target_track.body_data.frames_indices
                # find frames indices which are common for reference and target body and full body data
                current_common_reference_target_frames_indices = np.intersect1d(current_reference_body_unified_frames_indices.values,
                                                                                current_target_body_unified_frames_indices.values)

                current_reference_frames_indices_set = OrderedSet(current_reference_body_unified_frames_indices.values)
                current_reference_body_frames_indices_keys = current_reference_frames_indices_set.index(current_common_reference_target_frames_indices)
                current_reference_bounding_boxes = current_reference_track.tracked_data.bounding_boxes[current_reference_body_frames_indices_keys]

                current_target_frames_indices_set = OrderedSet(current_target_body_unified_frames_indices.values)
                current_target_body_frames_indices_keys = current_target_frames_indices_set.index(current_common_reference_target_frames_indices)
                current_target_bounding_boxes = current_reference_track.tracked_data.bounding_boxes[current_target_body_frames_indices_keys]

                current_bounding_boxes_iou = BoundingBoxes2DArray.intersection_over_union(current_reference_bounding_boxes, current_target_bounding_boxes)
                current_iou_above_threshold_mask =  current_bounding_boxes_iou > self.iou_threshold

                current_common_reference_target_frames_indices_filtered = current_common_reference_target_frames_indices[current_iou_above_threshold_mask]

                if not np.all(current_iou_above_threshold_mask == False):
                    current_reference_track.body_data.frames_indices -= current_common_reference_target_frames_indices_filtered
                    current_reference_track.full_body_data.frames_indices -= current_common_reference_target_frames_indices_filtered

                    current_target_track.body_data.frames_indices -= current_common_reference_target_frames_indices_filtered
                    current_target_track.full_body_data.frames_indices -= current_common_reference_target_frames_indices_filtered
