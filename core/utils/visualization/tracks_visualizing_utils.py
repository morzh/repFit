import numpy as np
import cv2


def stepped_color(step, factor):
    hsv_color = np.uint8([[[(factor * step) % 180, 255, 255]]])
    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
    rgb_color = rgb_color[0, 0]
    rgb_color = (int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))
    return rgb_color


def alpha_from_confidence(minimum, maximum, confidence, normalized=True) -> int | float:
    values_range = maximum - minimum
    alpha_value = np.clip(confidence, minimum, maximum) - minimum
    alpha_value /= values_range
    if not normalized:
        alpha_value *= 255
        return int(alpha_value)
    return alpha_value


def draw_bounding_boxes(person_id, frame, bounding_box, color, boxes_thickness) -> np.ndarray:
    person_id_text = str(person_id).zfill(2)

    rectangle_point_1 = tuple(bounding_box[:2])
    rectangle_point_2 = tuple(bounding_box[2:])
    id_label_point_2 = (rectangle_point_1[0] + len(person_id_text) * 24, rectangle_point_1[1] + 30)
    id_text_point = (rectangle_point_1[0] + 5, rectangle_point_1[1] + 25)

    overlay = frame.copy()
    cv2.rectangle(overlay, rectangle_point_1, rectangle_point_2, color, boxes_thickness)
    cv2.rectangle(overlay, rectangle_point_1, id_label_point_2, color, -1)
    cv2.putText(overlay, person_id_text, id_text_point, cv2.FONT_HERSHEY_DUPLEX, 1., color=(200, 200, 200), thickness=6, lineType=cv2.LINE_AA)
    cv2.putText(overlay, person_id_text, id_text_point, cv2.FONT_HERSHEY_DUPLEX, 1., color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    return overlay


def draw_skeleton_joints(frame: cv2.typing.MatLike, color, radius: int, keypoints: np.ndarray) -> cv2.typing.MatLike:
    number_joints = keypoints.shape[0]
    for joint_index in range(number_joints):
        frame_overlay = frame.copy()
        joint_center = (int(keypoints[joint_index, 0]), int(keypoints[joint_index, 1]))
        joint_confidence = float(keypoints[joint_index, 2])
        frame_overlay = cv2.circle(frame_overlay, joint_center, radius, color=color, thickness=-1, lineType=cv2.LINE_AA)
        frame =  cv2.addWeighted(frame_overlay, joint_confidence, frame, 1 - joint_confidence, 0)

    return frame
