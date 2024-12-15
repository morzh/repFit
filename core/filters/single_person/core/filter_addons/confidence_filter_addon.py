from core.filters.single_person.core.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class ConfidenceFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:

    """

    def __init__(self, confidence_threshold=0.25):
        self.confidence_threshold = confidence_threshold


    def process(self, tracks: MultiplePersonsTracks) -> None:
        keys_to_delete = [int]
        for person_key, person in tracks.persons.items():
            current_data = person.data
            number_samples = len(current_data)
            for sample_index in reversed(range(number_samples)):
                if current_data.confidences[sample_index] < self.confidence_threshold:
                    del current_data[sample_index]

            if len(current_data) == 0:
                keys_to_delete.append(person_key)
            else:
                person.calculate_segments(tracks.frames_stride)


        for key in keys_to_delete:
            tracks.persons.pop(key, None)
