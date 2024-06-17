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
    frames_number: int
    video_fps: float
    segments: segments_list
