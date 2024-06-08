from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VideoSegmentsWriter:
    filepath: str

    def write(self, segments: list[range]):
        pass
