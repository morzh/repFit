import numpy as np
import cv2


def stepped_color(step_value, steps_number) -> tuple[int, int, int]:
    """
    Description:
        Calculates color using HSV model with given ``step_value`` and ``steps_number``.
        Output is BGR color values.

    :param step_value: HSV color model hue step size;
    :param steps_number: number of steps.

    :return: color BGR values.
    """
    hsv_color = np.uint8([[[(steps_number * step_value) % 180, 255, 255]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
    bgr_color = bgr_color[0, 0]
    bgr_color = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
    return bgr_color


def alpha_from_confidence(minimum: float, maximum: float, confidence: float, normalized=True) -> int | float:
    """
    Description:
        Calculates color alpha value from ``confidence`` value and ``minimum`` -- ``maximum`` range.

    :param minimum: minimum bound value;
    :param maximum: maximum bound value;
    :param confidence: confidence;
    :param normalized: if True, alpha is in [0, 1] range or in [0, 255] otherwise.

    :return: alpha color value.
    """
    values_range = maximum - minimum
    alpha_value = np.clip(confidence, minimum, maximum) - minimum
    alpha_value /= values_range
    if not normalized:
        alpha_value *= 255
        return int(alpha_value)
    return alpha_value


def draw_bounding_boxes(person_id: int, frame: cv2.typing.MatLike, bounding_box: np.ndarray, color: tuple[int, int, int], boxes_thickness: int) -> np.ndarray:
    """
    Description:
        DraW bounding box with ''person_id`` label on a ``frame``.

    :param person_id: person ID;
    :param frame: video frame;
    :param bounding_box: bounding box array;
    :param color: BGR color values;
    :param boxes_thickness: bounding box thickness.

    :return: frame with bounding box drawn.
    """
    person_id_text = str(person_id).zfill(2)

    rectangle_point_1 = tuple(bounding_box[:2])
    rectangle_point_2 = tuple(bounding_box[2:])

    if rectangle_point_1[1] < 25:
        id_label_point_2 = (rectangle_point_1[0] + len(person_id_text) * 24, rectangle_point_1[1] + 30)
        id_text_point = (rectangle_point_1[0] + 5, rectangle_point_1[1] + 25)
    else:
        id_label_point_2 = (rectangle_point_1[0] + len(person_id_text) * 24, rectangle_point_1[1] - 30)
        id_text_point = (rectangle_point_1[0] + 5, rectangle_point_1[1] - 5)

    overlay = frame.copy()
    cv2.rectangle(overlay, rectangle_point_1, rectangle_point_2, color, boxes_thickness)
    cv2.rectangle(overlay, rectangle_point_1, id_label_point_2, color, -1)
    cv2.putText(overlay, person_id_text, id_text_point, cv2.FONT_HERSHEY_DUPLEX, 1., color=(200, 200, 200), thickness=6, lineType=cv2.LINE_AA)
    cv2.putText(overlay, person_id_text, id_text_point, cv2.FONT_HERSHEY_DUPLEX, 1., color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    return overlay


def draw_skeleton_joints(frame: cv2.typing.MatLike, color, radius: int, keypoints: np.ndarray, confidence_threshold=0.25) -> cv2.typing.MatLike:
    """
    Description:
        Draw COCO 17 points skeleton joints on an image.

    Remarks:
        COCO 17 joints format:
        https://github.com/robertklee/COCO-Human-Pose/blob/main/figures/skeleton_442619_flip_107_labelled.png
        0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear
        5: Left Shoulder 6: Right Shoulder 7: Left Elbow 8: Right Elbow 9: Left Wrist 10: Right Wrist
        11: Left Hip 12: Right Hip 13: Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle

    :param frame: video frame;
    :param color: BGR color values;
    :param radius: joint radius;
    :param keypoints: keypoints array;
    :param confidence_threshold: confidence threshold.

    :return: image with depicted joints.
    """
    number_joints = keypoints.shape[0]
    for joint_index in range(number_joints):
        if keypoints[joint_index, 2] > confidence_threshold and keypoints[joint_index, 0] > 0 and keypoints[joint_index, 1] > 0:
            frame_overlay = frame.copy()
            joint_center = (int(keypoints[joint_index, 0]), int(keypoints[joint_index, 1]))
            joint_confidence = float(keypoints[joint_index, 2])
            frame_overlay = cv2.circle(frame_overlay, joint_center, radius, color=color, thickness=-1, lineType=cv2.LINE_AA)
            frame =  cv2.addWeighted(frame_overlay, joint_confidence, frame, 1 - joint_confidence, 0)
    return frame


def draw_skeleton_bones(frame: cv2.typing.MatLike, color: tuple[int, int, int], thickness: int, keypoints: np.ndarray, confidence_threshold=0.1) -> cv2.typing.MatLike:
    """
    Description
        Draw COCO 17 points skeleton bones on an image.

    Remarks:
        COCO 17 joints format:
        https://github.com/robertklee/COCO-Human-Pose/blob/main/figures/skeleton_442619_flip_107_labelled.png
        0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear
        5: Left Shoulder 6: Right Shoulder 7: Left Elbow 8: Right Elbow 9: Left Wrist 10: Right Wrist
        11: Left Hip 12: Right Hip 13: Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle

    :param frame: video frame
    :param color: BGR color values;
    :param thickness: bones thickness;
    :param keypoints: coco keypoints array;
    :param confidence_threshold: confidence threshold.

    :return: image with depicted bones.
    """
    joints_pairs = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15), (6, 12), (12, 14), (14, 16), (11, 12)]
    for joints_pair in joints_pairs:
        point1 = (int(keypoints[joints_pair[0], 0]), int(keypoints[joints_pair[0], 1]))
        point2 = (int(keypoints[joints_pair[1], 0]), int(keypoints[joints_pair[1], 1]))
        if (keypoints[joints_pair[0], 2] > confidence_threshold and  keypoints[joints_pair[0], 2] > confidence_threshold and
            keypoints[joints_pair[0], 0] > 0 and keypoints[joints_pair[0], 1] > 0 and
            keypoints[joints_pair[1], 0] > 0 and keypoints[joints_pair[1], 1] > 0):

            confidence = float(0.5 * (keypoints[joints_pair[0], 2] + keypoints[joints_pair[1], 2]))
            frame_overlay = frame.copy()
            frame_overlay = cv2.line(frame_overlay, point1, point2, color=color, thickness=thickness, lineType=cv2.LINE_AA)
            frame = cv2.addWeighted(frame_overlay, confidence, frame, 1.0 - confidence, gamma=0)

    return frame
