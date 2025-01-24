import numpy as np
from ordered_set import OrderedSet

from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.source.frames_indices import FramesIndices
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks


class ConfidenceFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter out person's frames, whose (bounding box) confidence less than given threshold.

    :ivar confidence_threshold: confidence threshold
    """
    def __init__(self, confidence_threshold=0.25):
        self.confidence_threshold = confidence_threshold


    def process(self, tracks: MultiplePersonsTracks, filter_full_body_person=False) -> None:
        if not len(tracks.persons):
            return

        if filter_full_body_person:
            self.filter_person_track_full_body_data(tracks)
        else:
            self.filter_person_track_data(tracks)


    def filter_person_track_data(self, tracks: MultiplePersonsTracks):
        for person_id, person_track in tracks.persons.items():
            if not len(person_track.tracked_data):
                continue

            current_frames_indices_set = OrderedSet(person_track.tracked_data.frames_indices.values)
            person_frames_keys = current_frames_indices_set.index(person_track.data.frames_indices.values)
            persons_confidences = person_track.tracked_data.confidences[person_frames_keys]
            confidences_mask = persons_confidences > self.confidence_threshold
            person_track.data.frames_indices = FramesIndices(person_track.data.frames_indices.values[confidences_mask])


    def filter_person_track_full_body_data(self, tracks: MultiplePersonsTracks):
        for person_id, person_track in tracks.persons.items():
            if not len(person_track.tracked_data):
                continue

            current_frames_indices_set = OrderedSet(person_track.tracked_data.frames_indices.values)
            person_full_body_frames_keys = current_frames_indices_set.index(person_track.full_body_data.frames_indices.values)
            if not len(person_full_body_frames_keys): return
            person_full_body_frames_keys = np.array(person_full_body_frames_keys)
            full_body_persons_confidences = person_track.tracked_data.confidences[person_full_body_frames_keys]
            confidences_mask = full_body_persons_confidences > self.confidence_threshold
            filtered_keys = person_full_body_frames_keys[confidences_mask]
            person_track.full_body_data.frames_indices = FramesIndices(person_track.tracked_data.frames_indices[filtered_keys])