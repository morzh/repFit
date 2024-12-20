import numpy as np

from core.filters.single_person.core.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class WholePersonFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter frames at which there are at least given amount of  confident joints.

    :ivar joints_confidence_threshold: confident joints threshold;
    :ivar joints_number_threshold: maximum number of joints in frame (if less, filter frame).
    """
    def __init__(self, **parameters):
        self.joints_confidence_threshold = parameters.get('joints_confidence_threshold', 0.7)
        self.joints_number_threshold = parameters.get('joints_number_threshold', 16)


    def process(self, tracks: MultiplePersonsTracks) -> None:
        """
        Description:
            Select frame with keypoints if number of confident within threshold joints is greater than the given joints number.

        :param tracks: person's track
        """
        for person_id, person in tracks.persons.items():
            current_keypoints = person.data.keypoints
            current_keypoints_confidences = current_keypoints[:, :, 2]

            current_keypoints_confidence_threshold = current_keypoints_confidences > self.joints_confidence_threshold
            current_keypoints_bounds_threshold = (current_keypoints[:, :, 0] +  current_keypoints[:, :, 1]) > 1e-6
            current_keypoints_above_confidence_thresholds = np.logical_and(current_keypoints_confidence_threshold, current_keypoints_bounds_threshold)

            current_joints_number_above_confidence_thresholds = np.sum(current_keypoints_above_confidence_thresholds, axis=1)
            current_confident_joints_number_mask = current_joints_number_above_confidence_thresholds > self.joints_number_threshold

            person.whole_person_frame_indices = person.data.frames_indices[current_confident_joints_number_mask]
