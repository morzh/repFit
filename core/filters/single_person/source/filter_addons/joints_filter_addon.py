import numpy as np

from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.utils.cv.frames_indices import FramesIndices
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks


class JointsFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        This is two steps filter.
        1. Select frames at which confident joints number is greater than the given joints number;
        2. Assign selected frames indices to body or full body data.

    :ivar joints_confidence_threshold: confident joints threshold;
    :ivar joints_minimum_number_threshold: minimum number of joints in frame (if less, delete respective frame index);
    :ivar joints_maximum_number_threshold: maximum number of joints in frame (if more, delete respective frame index);
    :ivar filter_full_body_person: if True filter full body data, just body data otherwise.
    """
    def __init__(self, **parameters):
        self.joints_confidence_threshold = parameters.get('joints_confidence_threshold', 0.7)
        self.joints_minimum_number_threshold = parameters.get('joints_minimum_number_threshold', 16)
        self.joints_maximum_number_threshold = parameters.get('joints_maximum_number_threshold', 100)
        self.filter_full_body_person = parameters.get('filter_full_body_person', True)


    def process(self, tracks: MultiplePersonsTracks) -> None:
        for person_id, person_track in tracks.persons.items():
            if person_track.tracked_data.joints is None or not person_track.is_active: continue

            current_joints_confidences = person_track.tracked_data.joints[:, :, 2]
            current_joints_confidence_mask = current_joints_confidences >  self.joints_confidence_threshold
            current_joints_number_above_confidence_thresholds = np.sum(current_joints_confidence_mask, axis=1)
            current_confident_joints_number_mask = self.joints_minimum_number_threshold <= current_joints_number_above_confidence_thresholds < self.joints_maximum_number_threshold
            current_joints_indices_above_thresholds = person_track.tracked_data.frames_indices[current_confident_joints_number_mask]

            if self.filter_full_body_person:
                person_track.full_body_data.frames_indices = FramesIndices(current_joints_indices_above_thresholds)
            else:
                person_track.body_data.frames_indices = FramesIndices(current_joints_indices_above_thresholds)
