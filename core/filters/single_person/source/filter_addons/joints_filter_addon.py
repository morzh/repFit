import numpy as np
from ordered_set import OrderedSet

from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.source.frames_indices import FramesIndices
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks


class JointsFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        This is two steps filter.
        1. Select frames at which confident joints number is greater than the given joints number;
        2. Assign selected frames indices to body or full body data.

    :ivar joints_confidence_threshold: confident joints threshold;
    :ivar joints_number_threshold: maximum number of joints in frame (if less, filter frame).
    """
    def __init__(self, **parameters):
        self.joints_confidence_threshold = parameters.get('joints_confidence_threshold', 0.7)
        self.joints_number_threshold = parameters.get('joints_number_threshold', 16)


    def process(self, tracks: MultiplePersonsTracks, filter_full_body_person=True) -> None:
        for person_id, person_track in tracks.persons.items():
            if person_track.tracked_data.joints is None: continue

            current_joints_confidences = person_track.tracked_data.joints[:, :, 2]
            current_joints_confidence_mask = current_joints_confidences >  self.joints_confidence_threshold
            current_joints_number_above_confidence_thresholds = np.sum(current_joints_confidence_mask, axis=1)
            current_confident_joints_number_mask = current_joints_number_above_confidence_thresholds > self.joints_number_threshold
            current_joints_indices_above_thresholds = person_track.tracked_data.frames_indices[current_confident_joints_number_mask]


            if filter_full_body_person:
                person_track.full_body_data.frames_indices = FramesIndices(current_joints_indices_above_thresholds)
            else:
                person_track.body_data.frames_indices = FramesIndices(current_joints_indices_above_thresholds)

            '''
            if filter_full_body_person:
                joints_indices = current_frames_indices_set.index(person_track.full_body_data.frames_indices.values)
            else:
                joints_indices = current_frames_indices_set.index(person_track.data.frames_indices.values)

            current_keypoints = person_track.tracked_data.joints[joints_indices]
            current_keypoints_confidences = current_keypoints[:, :, 2]

            current_keypoints_confidence_threshold = current_keypoints_confidences > self.joints_confidence_threshold
            current_keypoints_bounds_threshold = (current_keypoints[:, :, 0] +  current_keypoints[:, :, 1]) > 1e-6
            current_keypoints_above_thresholds = np.logical_and(current_keypoints_confidence_threshold, current_keypoints_bounds_threshold)

            current_joints_number_above_confidence_thresholds = np.sum(current_keypoints_above_thresholds, axis=1)
            current_confident_joints_number_mask = current_joints_number_above_confidence_thresholds > self.joints_number_threshold

            if filter_full_body_person:
                current_filtered_indices = person_track.full_body_data.frames_indices[current_confident_joints_number_mask]
                person_track.full_body_data.frames_indices = FramesIndices(current_filtered_indices)
            else:
                current_filtered_indices = person_track.data.frames_indices[current_confident_joints_number_mask]
                person_track.data.frames_indices = FramesIndices(current_filtered_indices)
            '''
