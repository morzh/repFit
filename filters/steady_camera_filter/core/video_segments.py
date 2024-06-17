import numpy as np

from dataclasses import dataclass
from typing import Annotated, Literal
from numpy.typing import NDArray

segments_list = Annotated[NDArray[np.int32], Literal["N", 2]]


@dataclass(slots=True)
class VideoSegments:
    video_filename: str
    video_width: int
    video_height: int
    video_fps: float
    frames_number: int
    segments: segments_list
