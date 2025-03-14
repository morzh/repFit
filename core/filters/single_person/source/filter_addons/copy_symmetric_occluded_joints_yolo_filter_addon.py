import copy

import numpy as np

from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks


class CopySymmetricOccludedJointsYoloFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        In YOLO, occluded joints have coordinates (0, 0).
        This filter simply copies position and confidence from symmetric non occluded joints to occluded ones.
    """
    def __init__(self, add_noise=True):
        self.add_noise = add_noise

    def process(self, tracks: MultiplePersonsTracks) -> None:
        if not len(tracks.persons):
            return

        for person_track in tracks.persons.values():
            if not len(person_track.tracked_data) or not person_track.is_active: continue
            current_joints = person_track.tracked_data.joints
            frames_number = current_joints.shape[0]
            symmetric_joints_indices = self.symmetric_joints_coco(frames_number)
            target_joints_indices = np.argwhere(current_joints[:, :, 0] + current_joints[:, :, 1] < 1e-6)  # in format [frame, joint_number]
            source_joints_indices = np.hstack((target_joints_indices[:, 0].reshape(-1, 1), symmetric_joints_indices[target_joints_indices[:, 1]].reshape(-1, 1)))

            occluded_joints_indices_0 = np.hstack((target_joints_indices, np.zeros((target_joints_indices.shape[0], 1), dtype=np.int64)))
            occluded_joints_indices_1 = np.hstack((target_joints_indices, np.ones((target_joints_indices.shape[0], 1), dtype=np.int64)))
            occluded_joints_indices_2 = np.hstack((target_joints_indices, 2 * np.ones((target_joints_indices.shape[0], 1), dtype=np.int64)))
            target_joints_indices = np.vstack((occluded_joints_indices_0, occluded_joints_indices_1, occluded_joints_indices_2))

            target_joints_indices_0 = np.hstack((source_joints_indices, np.zeros((source_joints_indices.shape[0], 1), dtype=np.int64)))
            target_joints_indices_1 = np.hstack((source_joints_indices, np.ones((source_joints_indices.shape[0], 1), dtype=np.int64)))
            target_joints_indices_2 = np.hstack((source_joints_indices, 2 * np.ones((source_joints_indices.shape[0], 1), dtype=np.int64)))
            source_joints_indices = np.vstack((target_joints_indices_0, target_joints_indices_1, target_joints_indices_2))

            current_joints_copy = copy.deepcopy(current_joints)

            target_rows, target_columns, target_depths = zip(*target_joints_indices)
            source_rows, source_columns, source_depths = zip(*source_joints_indices)

            current_joints[target_rows, target_columns, target_depths] =  current_joints_copy[source_rows, source_columns, source_depths]
            current_joints[target_rows, target_columns, target_depths] += np.random.randint(1, 10)


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
        coco_joints_symmetry =  np.array([0,
                                          2,
                                          1,
                                          4,
                                          3,
                                          6,
                                          5,
                                          8,
                                          7,
                                          10,
                                          9,
                                          12,
                                          11,
                                          14,
                                          13,
                                          16,
                                          15])

        return coco_joints_symmetry
        # return np.repeat(np.expand_dims(coco_joints_symmetry, axis=0), number_frames, axis=0)
