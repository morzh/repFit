import numpy as np
from scipy.interpolate import interp1d


def interpolation(joints: list, exist_ids: list, n_extra_points: int = 10) -> np.ndarray:
    """ If exist_ids contains gaps (frames without joints) - fill the gaps with interpolated joints.
    arguments:
        joints: list - list of 1d arrays
        exist_ids - int indexes or the position of the joints
        n_extra_points - additional frames around joints list for avoid interpolation side artefacts

    return:
        joints_array - 2d np.ndarray
    """
    joints_array = np.empty((exist_ids[-1] + 1, len(joints[0])))
    joints_array[exist_ids] = joints

    joints = ([joints[0] for _ in range(n_extra_points)] +
                 joints +
                 [joints[-1] for _ in range(n_extra_points)])
    exist_ids = ([i for i in range(exist_ids[0] - n_extra_points, exist_ids[0])] +
                 exist_ids +
                 [i for i in range(exist_ids[-1] + 1, exist_ids[-1] + n_extra_points + 1)])

    not_exist_ids = [i for i in range(exist_ids[-1]) if i not in exist_ids]
    interp_func = interp1d(exist_ids, np.array(joints), kind='cubic', axis=0)
    new_points = interp_func(not_exist_ids)
    joints_array[not_exist_ids] = new_points
    return joints_array
