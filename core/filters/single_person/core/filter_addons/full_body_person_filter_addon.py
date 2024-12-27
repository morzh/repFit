import numpy as np
from ordered_set import OrderedSet

from core.filters.single_person.core.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.frames_indices import FramesIndices
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class FullBodyPersonFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Select frames at which confident joints number is greater than the given joints number.

    :ivar joints_confidence_threshold: confident joints threshold;
    :ivar joints_number_threshold: maximum number of joints in frame (if less, filter frame).
    """
    def __init__(self, **parameters):
        self.joints_confidence_threshold = parameters.get('joints_confidence_threshold', 0.7)
        self.joints_number_threshold = parameters.get('joints_number_threshold', 16)


    def process(self, tracks: MultiplePersonsTracks, filter_full_body_person=False) -> None:
        for person_id, person_track in tracks.persons.items():
            current_frames_indices_set = OrderedSet(person_track.tracked_data.frames_indices.values)
            if filter_full_body_person:
                joints_indices = current_frames_indices_set.index(person_track.full_body_data.frames_indices.values)
            else:
                joints_indices = current_frames_indices_set.index(person_track.data.frames_indices.values)

            current_keypoints = person_track.tracked_data.keypoints[joints_indices]
            current_keypoints_confidences = current_keypoints[:, :, 2]

            current_keypoints_confidence_threshold = current_keypoints_confidences > self.joints_confidence_threshold
            current_keypoints_bounds_threshold = (current_keypoints[:, :, 0] +  current_keypoints[:, :, 1]) > 1e-6
            current_keypoints_above_confidence_thresholds = np.logical_and(current_keypoints_confidence_threshold, current_keypoints_bounds_threshold)

            current_joints_number_above_confidence_thresholds = np.sum(current_keypoints_above_confidence_thresholds, axis=1)
            current_confident_joints_number_mask = current_joints_number_above_confidence_thresholds > self.joints_number_threshold

            if filter_full_body_person:
                current_filtered_indices = person_track.full_body_data.frames_indices[current_confident_joints_number_mask]
                person_track.full_body_data.frames_indices = FramesIndices(current_filtered_indices)
            else:
                current_filtered_indices = person_track.data.frames_indices[current_confident_joints_number_mask]
                person_track.data.frames_indices = FramesIndices(current_filtered_indices)
