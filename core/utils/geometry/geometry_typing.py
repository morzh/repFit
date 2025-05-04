import numpy as np
import numpy.typing as npt
from typing import Annotated, Literal

numeric = int | float | np.float32 | np.float64
vec2d = Annotated[npt.NDArray[np.float32 | np.float64], Literal[2]] | tuple[float, float]
