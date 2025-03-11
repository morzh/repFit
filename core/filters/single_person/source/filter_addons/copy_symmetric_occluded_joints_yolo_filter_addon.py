import numpy as np

from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks


class CopySymmetricOccludedJointsYoloFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        In YOLO, occluded joints have coordinates (0, 0).
        This filter simply copies position and confidence from symmetric non occluded joints to occluded ones.
    """
    def __init__(self):
        ...

    def process(self, tracks: MultiplePersonsTracks) -> None:
        if not len(tracks.persons):
            return

        for person_track in tracks.persons.values():
            if not len(person_track.tracked_data) or not person_track.is_active: continue
            current_joints = person_track.tracked_data.joints
            frames_number = current_joints.shape[0]
            symmetric_joints_indices = self.symmetric_joints_coco(frames_number)
            occluded_joints_indices = np.argwhere(current_joints[:, :, 0] + current_joints[:, :, 1] < 1e-6)  # in format [frame, joint_number]
            occluded_joints_mask = (current_joints[:, :, 0] + current_joints[:, :, 1]) < 1e-6
            current_joints[occluded_joints_indices] = current_joints[symmetric_joints_indices[occluded_joints_indices]]


    @staticmethod
    def symmetric_joints_h36m(number_frames: int) -> np.ndarray:
        h36m_joints_symmetry =  np.array([[0, 0],
                                          [1, 4],
                                          [2, 5],
                                          [3, 6],
                                          [4, 1],
                                          [5, 2],
                                          [6, 3],
                                          [7, 7],
                                          [8, 8],
                                          [9, 9],
                                          [10, 10],
                                          [11, 14],
                                          [12, 15],
                                          [13, 16],
                                          [14, 11],
                                          [15, 12],
                                          [16, 13]])

        return np.repeat(np.expand_dims(h36m_joints_symmetry, axis=0), number_frames, axis=0)


    @staticmethod
    def symmetric_joints_coco(number_frames: int) -> np.ndarray:
        # left to right
        coco_joints_symmetry =  np.array([[0, 0],
                                          [1, 2],
                                          [2, 1],
                                          [3, 4],
                                          [4, 3],
                                          [5, 6],
                                          [6, 5],
                                          [7, 8],
                                          [8, 7],
                                          [9, 10],
                                          [10, 9],
                                          [11, 12],
                                          [12, 11],
                                          [13, 14],
                                          [14, 13],
                                          [15, 16],
                                          [16, 15]])

        return np.repeat(np.expand_dims(coco_joints_symmetry, axis=0), number_frames, axis=0)
