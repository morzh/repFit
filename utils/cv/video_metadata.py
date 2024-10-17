from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class VideoMetadata:
    filename: str
    width: int
    height: int
    frames_number: int
    fps: float
