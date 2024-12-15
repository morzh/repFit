import numpy as np

from core.filters.single_person.core.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class PartialPersonFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter frames at which there are only few confident joints.

    :ivar joints_threshold: confident joints threshold;
    :ivar joints_number: maximum number of joints in frame (if less, filter frame).
    """
    def __init__(self, **parameters):
        self.joints_threshold = parameters.get('joints_confidence_threshold', 0.65)
        self.joints_number = parameters.get('joints_number', 5)


    def process(self, tracks: MultiplePersonsTracks) -> None:
        """
        Description:
            Delete frame with keypoints if number of confident within threshold joints is less than given joints number.

        :param tracks: person's track
        """
        degenerate_tracks = []

        for person_id, person in tracks.persons.items():
            current_keypoints = person.data.keypoints
            current_keypoints_confidences = current_keypoints[:, :, 2]
            current_keypoints_threshold = current_keypoints_confidences > self.joints_threshold
            current_keypoints_bounds = (current_keypoints[:, :, 0] +  current_keypoints[:, :, 1]) > 1e-6
            selected_keypoints = np.logical_and(current_keypoints_threshold, current_keypoints_bounds)
            current_frames_threshold = np.sum(selected_keypoints, axis=1)
            current_frames_mask = current_frames_threshold > self.joints_number
            person.data.apply_mask(current_frames_mask)

            if person.is_data_empty():
                degenerate_tracks.append(person_id)


        for key in degenerate_tracks:
            tracks.persons.pop(key, None)
