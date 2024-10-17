from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class VideoMetadata:
    video_filename: str
    video_width: int
    video_height: int
    frames_number: int
    video_fps: float
