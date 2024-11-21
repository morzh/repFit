import os
import pickle

import cv2
import numpy as np
import torch

from core.filters.single_person.core.filters.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.utils.cv.video_properties import VideoProperties
from core.filters.single_person.core.single_person_track import SinglePersonTrack
from core.utils.cv.video_reader import VideoReader


class MultiplePersonsTracks:
    """
    Description:
        Persons data storage class.

    :ivar persons: person_id -> person data mapping
    :ivar video_properties:  video properties data
    """
    def __init__(self, video_properties: VideoProperties):
        self.persons: dict[int, SinglePersonTrack] = {}
        self.video_properties = video_properties
        self.frames_number: int  = -1


    def update(self, data: torch.Tensor, frame_number: int) -> None:
        """
        Description:
            Update persons tracks data.

        :param data: person's data, obtained from AI model;
        :param frame_number: frame number.
        """
        for index in range(data.shape[0]):
            current_person_id = int(data[index, 4])
            current_confidence = float(data[index, 5])
            if current_person_id not in self.persons:
                self.persons[current_person_id] = SinglePersonTrack()
            bounding_box = data[index].numpy()
            self.persons[current_person_id].append(bounding_box, frame_number, confidence=current_confidence)


    def apply_filter(self, filter_visitor: MultiPersonsFilterAddonBase) -> None:
        """
        Description:
            Apply ``filter_visitor`` filter in place.

        :param filter_visitor: filter class instance.
        """
        filter_visitor.process(self)


    def serialize(self, filepath: os.PathLike) -> None:
        """
        Description:
            Serialize class instance.

        :param filepath: filepath to write class instance to.
        """
        with open(filepath, mode='wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)


    def visualize(self, **options) -> None:
        """
        Description:
            Visualize multiple persons tracks.

        :params person_ids: person's ids
        """
        boxes_thickness = options.get('frames_thickness', 2)
        next_frame_wait = options.get('next_frame_wait', 10)

        video_reader = VideoReader(self.video_properties.filepath)

        for frame in video_reader:
            for person_id, person_track in self.persons.values():
                current_bounding_box = self.persons[person_id].data.bounding_box[video_reader.current_frame_index]
                current_point_1 = (current_bounding_box[0], current_bounding_box[1])
                current_point_2 = (current_bounding_box[0] + current_bounding_box[2], current_bounding_box[1] + current_bounding_box[3])
                current_color = (125, 125, 125)
                frame = cv2.rectangle(frame, current_point_1, current_point_2, current_color, boxes_thickness)

            cv2.imshow('Multiple Persons Track', frame)
            cv2.waitKey(next_frame_wait)


    @staticmethod
    def compare_visualization(persons_tracks_1, persons_tracks_2, **options) -> None:
        """
         Description:
            Visualize two multiple persons tracks (for comparison purpose).
        """

        boxes_thickness = options.get('frames_thickness', 2)
        next_frame_wait = options.get('next_frame_wait', 10)

        video_reader_1 = VideoReader(persons_tracks_1.video_properties.filepath)
        video_reader_2 = VideoReader(persons_tracks_2.video_properties.filepath)

        for frame_1, frame_2 in zip(video_reader_1, video_reader_2):
            frame = np.hstack((frame_1, frame_2))
            cv2.imshow('Multiple Persons Track', frame)
            cv2.waitKey(next_frame_wait)
