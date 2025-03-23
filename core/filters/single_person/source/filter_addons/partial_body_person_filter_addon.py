import numpy as np

from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks
from core.utils.cv.frames_indices import FramesIndices


class PartialBodyPersonFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Select frames at which number of confident body joints are in range of given thresholds.

    :ivar body_joints_confidence_threshold: confident body joints threshold;
    :ivar low_number_body_joints: minimum joints number above body joints threshold;;
    :ivar high_number_body_joints: maximum joints number above body joints threshold;;
    """
    def __init__(self, **parameters):
        self.body_joints_confidence_threshold = parameters.get('body_joints_confidence_threshold', 0.7)
        self.low_number_body_joints = parameters.get('low_number_body_joints', 5)
        self.high_number_body_joints = parameters.get('high_number_body_joints', 12)


    def process(self, tracks: MultiplePersonsTracks) -> None:
        for person_track in tracks.persons.values():
            if person_track.tracked_data.joints is None or not person_track.is_active: continue

            body_joints = person_track.tracked_data.joints[:, 5:]
            body_joints_confidence_mask = body_joints[:, :, 2] >= self.body_joints_confidence_threshold
            body_joints_occlusion_mask = (body_joints[:, :, 0] + body_joints[:, :, 1]) > 0
            body_joints_mask = np.logical_and(body_joints_confidence_mask, body_joints_occlusion_mask)
            body_joints_number_above_confidence_thresholds = np.sum(body_joints_mask, axis=1)

            body_joints_frames_low_mask = body_joints_number_above_confidence_thresholds >= self.low_number_body_joints
            body_joints_frames_high_mask = body_joints_number_above_confidence_thresholds <= self.high_number_body_joints

            body_joints_frames_mask = np.logical_and(body_joints_frames_low_mask, body_joints_frames_high_mask)

            person_track.partial_body_data.frames_indices = FramesIndices(person_track.tracked_data.frames_indices[body_joints_frames_mask])
