from enum import Enum

class BoundingBoxMode(Enum):
    """
    Description:
        BoxMode defines the mode in which the bounding box is defined.

        Most data sources have bounding boxes defined as ``XYWH`` where `XY` is the top left corner
            and `W` and `H` are the width and height of the box, respectively.

        However, many algorithms prefer to deal with bounding boxes as ``XYXY`` where the box is \
            defined is defined by the top-left corner and the bottom-right corner.

        To help disambiguate between these two configurations, `bbox` provides a means to specify the \
            mode and maintains the state internally.
    """
    XYWH = 0
    XYXY = 1