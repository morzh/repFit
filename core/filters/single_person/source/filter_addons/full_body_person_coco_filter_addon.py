import numpy as np

from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks
from core.utils.cv.frames_indices import FramesIndices


class FullBodyPersonCocoFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Select frames at which number of confident head and body joints are above given thresholds.

    :ivar head_joints_confidence_threshold: confident head joints threshold;
    :ivar accepted_number_head_joints: minimum joints number above head joints threshold;
    :ivar body_joints_confidence_threshold: confident body joints threshold;
    :ivar accepted_number_body_joints: minimum joints number above body joints threshold;;
    """
    def __init__(self, **parameters):
        self.head_joints_confidence_threshold = parameters.get('head_joints_confidence_threshold', 0.6)
        self.accepted_number_head_joints = parameters.get('accepted_number_head_joints', 1)
        self.body_joints_confidence_threshold = parameters.get('body_joints_confidence_threshold', 0.7)
        self.accepted_number_body_joints = parameters.get('accepted_number_body_joints', 11)


    def process(self, tracks: MultiplePersonsTracks) -> None:
        for person_track in tracks.persons.values():
            if person_track.tracked_data.joints is None or not person_track.is_active: continue

            head_joints = person_track.tracked_data.joints[:, :5]
            body_joints = person_track.tracked_data.joints[:, 5:]

            head_joints_confidence_mask = head_joints[:, :, 2] >= self.head_joints_confidence_threshold
            head_joints_occlusion_mask = (head_joints[:, :, 0] + head_joints[:, :, 1]) > 0
            head_joints_mask = np.logical_and(head_joints_confidence_mask, head_joints_occlusion_mask)
            head_joints_number_above_confidence_thresholds = np.sum(head_joints_mask, axis=1)
            head_joints_frames_mask = head_joints_number_above_confidence_thresholds >= self.accepted_number_head_joints

            body_joints_confidence_mask = body_joints[:, :, 2] >= self.body_joints_confidence_threshold
            body_joints_occlusion_mask = (body_joints[:, :, 0] + body_joints[:, :, 1]) > 0
            body_joints_mask = np.logical_and(body_joints_confidence_mask, body_joints_occlusion_mask)
            body_joints_number_above_confidence_thresholds = np.sum(body_joints_mask, axis=1)
            body_joints_frames_mask = body_joints_number_above_confidence_thresholds >= self.accepted_number_body_joints

            joints_frames_mask = np.logical_and(head_joints_frames_mask, body_joints_frames_mask)

            person_track.full_body_data.frames_indices = FramesIndices(person_track.tracked_data.frames_indices[joints_frames_mask])
