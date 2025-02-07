import numpy as np
from ordered_set import OrderedSet

from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.utils.cv.frames_indices import FramesIndices
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks


class ConfidenceFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter out person's frames (body or full body), whose bounding box confidence is less than  the given threshold.

    :ivar confidence_threshold: confidence threshold
    """
    def __init__(self, confidence_threshold=0.25):
        self.confidence_threshold = confidence_threshold


    def process(self, tracks: MultiplePersonsTracks, filter_full_body_person=False) -> None:
        if not len(tracks.persons):
            return

        for person_track in tracks.persons.values():
            if not len(person_track.tracked_data): continue
            current_data_reference = person_track.full_body_data if filter_full_body_person else person_track.body_data

            current_frames_indices_set = OrderedSet(person_track.tracked_data.frames_indices.values)
            current_frames_keys = current_frames_indices_set.index(current_data_reference.frames_indices.values)
            if not len(current_frames_keys): continue

            current_frames_keys = np.array(current_frames_keys)
            current_persons_confidences = person_track.tracked_data.confidences[current_frames_keys]

            confidences_mask = current_persons_confidences > self.confidence_threshold
            current_data_reference.frames_indices = FramesIndices(current_data_reference.frames_indices.values[confidences_mask])
