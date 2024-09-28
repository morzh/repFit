
VectorF2D = tuple[float, float]

class Segment2D:
    """
    Description:
        Representing 2D segment using start and end points.
    """
    __slots__ = ['start', 'end']

    def __init__(self, start: VectorF2D, end: VectorF2D):
        self.start = start
        self.end = end

    def __eq__(self, other: VectorF2D):
        return self.start == other.start and self.end == other.end

    def __repr__(self):
        return f"Segment2D({self.start}, {self.end})"

    def intersection(self, other) -> VectorF2D | None:
        """
        Description:
            Calculates intersection point of this segment with the given onr.

        :param other:

        :return: intersection point (if any) or None (if there is no intersection)
        """

    def distance(self, other) -> float:
        """
        Description:
            Closest distance from this segment to the given one.

        :param other:

        :return: closest distance
        """

    def normal(self):
        """
        Description:
        """
