import numpy as np

from core.utils.geometry.bounding_boxes.bounding_box_2d import BoundingBox2D


def generate_bounding_box() -> BoundingBox2D:
    left_top = np.random.randint(-10_000, 10_000, 2)
    width_height = np.random.randint(1, 2500, 2)
    return BoundingBox2D(int(left_top[0]), int(left_top[1]), int(width_height[0]), int(width_height[1]))


def generate_bounding_box_corners() -> np.ndarray:
    values_x = np.random.randint(-10_000, 10_000, 2)
    values_y = np.random.randint(-10_000, 10_000, 2)

    left_top = np.array([min(values_x), min(values_y)])
    right_top = np.array([max(values_x), min(values_y)])
    right_bottom = np.array([max(values_x), max(values_y)])
    left_bottom = np.array([min(values_x), max(values_y)])

    return np.vstack((left_top, right_top, right_bottom, left_bottom))


def vertices_to_bounding_box(point_cloud) -> BoundingBox2D:
    return BoundingBox2D.from_two_points(point_cloud[0], point_cloud[2])


def offset_point_cloud(point_cloud, offset_value) -> np.ndarray:
    point_cloud_offset = point_cloud.copy()
    #  Offset in Y direction
    point_cloud_offset[0, 1] -= offset_value
    point_cloud_offset[1, 1] -= offset_value

    point_cloud_offset[2, 1] += offset_value
    point_cloud_offset[3, 1] += offset_value

    #  Offset in X direction
    point_cloud_offset[1, 0] += offset_value
    point_cloud_offset[2, 0] += offset_value

    point_cloud_offset[0, 0] -= offset_value
    point_cloud_offset[3, 0] -= offset_value

    return point_cloud_offset


def generate_inner_point(vertices: np.ndarray) -> np.ndarray:
    convex_combination_coefficients = np.random.random(4) + 1e-5
    coefficients_factor = np.sum(convex_combination_coefficients)
    convex_combination_coefficients /= coefficients_factor

    inner_point = np.zeros(2, )
    for index in range(4):
        inner_point += convex_combination_coefficients[index] * vertices[index]

    return inner_point


def generate_outer_points(vertices: np.ndarray, number_points=1) -> np.ndarray:
    side = np.random.randint(0, 4)
    spread = 1_000

    match side:
        case 0:
            y_top_minus = np.random.randint(vertices[0, 1] - spread, high=vertices[0,1] - 1, size=(number_points, 1))
            x_value = np.random.randint(-10_000, 10_000, size=(number_points, 1))
            return np.hstack((x_value, y_top_minus))
        case 1:
            x_right_plus = np.random.randint(vertices[1, 0] + 1, vertices[1, 0] + spread, size=(number_points, 1))
            y_value = np.random.randint(-10_000, 10_000, size=(number_points, 1))
            return np.hstack((x_right_plus, y_value))
        case 2:
            y_bottom_plus = np.random.randint(vertices[2, 1] + 1, vertices[2, 1] + spread, size=(number_points, 1))
            x_value = np.random.randint(-10_000, 10_000, size=(number_points, 1))
            return np.hstack((x_value, y_bottom_plus))
        case 3:
            x_right_minus = np.random.randint(vertices[0, 0] - spread, vertices[0, 0] - 1, size=(number_points, 1))
            y_value = np.random.randint(-10_000, 10_000, size=(number_points, 1))
            return np.hstack((x_right_minus, y_value))

