import numpy as np
import numpy.typing as npt
from typing import Annotated, Literal,TypeVar, Union

numeric = Union[int, float, np.float32, np.float64]
vec2d = Annotated[npt.NDArray[np.float32 | np.float64], Literal[2]] | tuple[float, float]
bbox2d = TypeVar("bbox2d", bound="BoundingBox2D")
line2d = TypeVar("line2d", bound="Line2D")
segment2d = TypeVar("segment2d", bound="Segment2D")
