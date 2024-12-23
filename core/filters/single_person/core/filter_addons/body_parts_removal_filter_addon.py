import numpy as np

from core.filters.single_person.core.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class BodyPartsRemovalFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter out frames with only few detected joints.

    :ivar joints_percents:
    :ivar joints_confidence_threshold
    """
    def __init__(self, joints_percents, confidence_threshold):
        self.joints_percents = joints_percents
        self.joints_confidence_threshold = confidence_threshold


    def process(self, tracks: MultiplePersonsTracks, filter_full_body_person=False) -> None:
        """
        Description:

        :param tracks:
        :param filter_full_body_person:
        """
        for person in tracks.persons.values():
            for frame_index, frame in enumerate(person.tracked_data.frames_indices):
                current_keyframes_confidences = person.tracked_data.keypoints_confidences(frame)
                joints_number = person.tracked_data.joints_number
                current_joint_threshold_indices = np.argwhere(current_keyframes_confidences < self.joints_confidence_threshold)
                if (current_joint_threshold_indices.shape[0] / joints_number) < self.joints_percents:
                    del person.tracked_data[frame_index]
