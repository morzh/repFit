import numpy as np

from core.filters.single_person.core.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks
from core.filters.single_person.core.single_person_track import SinglePersonStatus


class BodyPartsRemovalFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
    """
    def __init__(self, joints_percents, confidence_threshold):
        self.joints_percents = joints_percents
        self.joints_confidence_threshold = confidence_threshold


    def process(self, tracks: MultiplePersonsTracks) -> None:
        """
        Description:
        """
        for person in tracks.persons.values():
            for frame_index, frame in enumerate(person.data.frames_indices):
                current_keyframes_confidences = person.data.keypoints_confidences(frame)
                joints_number = person.data.joints_number
                current_joint_threshold_indices = np.argwhere(current_keyframes_confidences < self.joints_confidence_threshold)
                if (current_joint_threshold_indices.shape[0] / joints_number) < self.joints_percents:
                    del person.data[frame_index]
