import sys
import cv2
import json
import bpy
import bmesh
import glob
import os.path
from typing import Type
import numpy as np

class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_h36m_joints_colors() -> list[tuple]:
    number_joints = 17
    joints_colors = [(1, 1, 1, 1)] * number_joints
    light_blue = (0.25, 0.25, 1.0, 1.0)
    light_green = (0.25, 1.0, 0.25, 1.0)
    orange = (0.7, 0.5, 0.1, 1.0)
    red = (1.0, 0.25, 0.25, 1.0)

    joints_colors[0] = orange  # root

    joints_colors[1] = light_blue  # left leg
    joints_colors[2] = light_blue  # left leg
    joints_colors[3] = light_blue  # left leg

    joints_colors[4] = light_green  # right leg
    joints_colors[5] = light_green  # right leg
    joints_colors[6] = light_green  # right leg

    joints_colors[7] = orange  # spine
    joints_colors[8] = orange  # spine

    joints_colors[9] = red  # head
    joints_colors[10] = red  # head

    joints_colors[11] = light_green  # right arm
    joints_colors[12] = light_green  # right arm
    joints_colors[13] = light_green  # right arm

    joints_colors[14] = light_blue  # left arm
    joints_colors[15] = light_blue  # left arm
    joints_colors[16] = light_blue  # left arm

    return joints_colors


def get_h36m_joints_names():
    h36m_number_joints = 17
    joints_names = [str] * h36m_number_joints

    joints_names[0] = '00 Bottom Torso'

    joints_names[1] = '01 Left Hip'
    joints_names[2] = '02 Left Knee'
    joints_names[3] = '03 Left Foot'

    joints_names[4] = '04 Right Hip'
    joints_names[5] = '05 Right Knee'
    joints_names[6] = '06 Right Foot'

    joints_names[7] = '07 Center Torso'
    joints_names[8] = '08 Upper Torso'

    joints_names[9] = '09 Neck Base'
    joints_names[10] = '10 Center Head'

    joints_names[11] = '11 Right Shoulder'
    joints_names[12] = '12 Right Elbow'
    joints_names[13] = '13 Right Hand'

    joints_names[14] = '14 Left Shoulder'
    joints_names[15] = '15 Left Elbow'
    joints_names[16] = '16 Left Hand'

    return joints_names


def get_h36m_skeleton_curves(joints_locations, joints_colors) -> dict[str, any]:
    assert joints_locations.shape == (17, 3)

    left_leg_initial_locations = joints_locations[:4]
    right_leg_initial_locations = np.vstack((joints_locations[0], joints_locations[4:7]))
    torso_head_locations = np.vstack((joints_locations[0], joints_locations[7:11]))
    left_arm_locations = np.vstack((joints_locations[8], joints_locations[14:17]))
    right_arm_locations = np.vstack((joints_locations[8], joints_locations[11:14]))

    left_leg_color = joints_colors[1]
    right_leg_color = joints_colors[4]
    torso_head_color = joints_colors[7]
    left_arm_color = joints_colors[14]
    right_arm_color = joints_colors[12]

    left_leg_curve = create_curve_3d('LeftLegCurve', left_leg_initial_locations, left_leg_color)
    right_leg_curve = create_curve_3d('RightLegCurve', right_leg_initial_locations, right_leg_color)
    torso_head_curve = create_curve_3d('TorsoHeadCurve', torso_head_locations, torso_head_color)
    left_arm_curve = create_curve_3d('LeftArmCurve', left_arm_locations, left_arm_color)
    right_arm_curve = create_curve_3d('RightArmCurve', right_arm_locations, right_arm_color)

    return {'LeftLegCurve': left_leg_curve, 'RightLegCurve': right_leg_curve,
            'TorsoHeadCurve': torso_head_curve,
            'LeftArmCurve': left_arm_curve, 'RightArmCurve': right_arm_curve}


def parent_skeleton_curves(skeleton_curves, parent) -> None:
    for curve_object in skeleton_curves.values():
        curve_object.parent = parent


def hook_skeleton_h36m_curves(skeleton_curves: dict[str, any], joints_spheres: list) -> None:
    for curve_name, data_object in skeleton_curves.items():
        match curve_name:
            case 'LeftLegCurve':
                left_leg_hooks = {0: joints_spheres[0],
                                  1: joints_spheres[1],
                                  2: joints_spheres[2],
                                  3: joints_spheres[3]}
                hook_skeleton_h36m_single_curve(data_object, left_leg_hooks)
            case 'RightLegCurve':
                right_leg_hooks = {0: joints_spheres[0],
                                   1: joints_spheres[4],
                                   2: joints_spheres[5],
                                   3: joints_spheres[6]}
                hook_skeleton_h36m_single_curve(data_object, right_leg_hooks)
            case 'TorsoHeadCurve':
                torso_head_hooks = {0: joints_spheres[0],
                                    1: joints_spheres[7],
                                    2: joints_spheres[8],
                                    3: joints_spheres[9],
                                    4: joints_spheres[10]}
                hook_skeleton_h36m_single_curve(data_object, torso_head_hooks)
            case 'LeftArmCurve':
                left_arm_hooks = {0: joints_spheres[8],
                                  1: joints_spheres[14],
                                  2: joints_spheres[15],
                                  3: joints_spheres[16]}
                hook_skeleton_h36m_single_curve(data_object, left_arm_hooks)
            case 'RightArmCurve':
                right_arm_hooks = {0: joints_spheres[8],
                                   1: joints_spheres[11],
                                   2: joints_spheres[12],
                                   3: joints_spheres[13]}
                hook_skeleton_h36m_single_curve(data_object, right_arm_hooks)


def create_curve_3d(curve_name: str, initial_locations: np.ndarray, color=(1., 1., 1., 1.), depth=0.01) -> any:
    curve_data = bpy.data.curves.new(curve_name, 'CURVE')
    curve_data.dimensions = '3D'

    spline = curve_data.splines.new('POLY')
    spline.points.add(len(initial_locations) - 1)

    for point_index, point in enumerate(spline.points):
        current_coordinates = (initial_locations[point_index, 0],
                               initial_locations[point_index, 1],
                               initial_locations[point_index, 2],
                               1)
        point.co = current_coordinates

    curve_object = bpy.data.objects.new(curve_name + 'Object', curve_data)
    curve_object.color = color
    curve_object.data.bevel_depth = 0.01
    curve_object.data.bevel_resolution = 10
    curve_object.data.splines[0].use_smooth = False

    view_layer = bpy.context.view_layer
    view_layer.active_layer_collection.collection.objects.link(curve_object)

    return curve_object


def clear_material(material):
    if material.node_tree:
        material.node_tree.links.clear()
        material.node_tree.nodes.clear()


def assign_material_skeleton_curves(skeleton_curves: dict, joints_skeleton_material):
    for curve in skeleton_curves.values():
        curve.data.materials.append(joints_skeleton_material)


def hook_skeleton_h36m_single_curve(curve_object: any, hooks: dict[int, str]) -> None:
    # assert len(curve_object.points) == len(hooks)
    bpy.ops.object.select_all(action='DESELECT')
    # bpy.context.space_data.context = 'MODIFIER'

    bpy.context.view_layer.objects.active = curve_object
    bpy.ops.object.mode_set(mode='EDIT')

    for curve_point_index, hook_object in hooks.items():
        bpy.ops.object.modifier_add(type='HOOK')
        current_hook_name = 'Hook' if curve_point_index == 0 else 'Hook.{:03d}'.format(curve_point_index)
        bpy.context.object.modifiers[current_hook_name].object = hook_object

        current_point = curve_object.data.splines[0].points[curve_point_index]
        current_point.select = True
        bpy.ops.object.hook_assign(modifier=current_hook_name)
        current_point.select = False

    bpy.ops.object.mode_set(mode='OBJECT')


def create_joints_skeleton_material() -> any:
    materials = bpy.data.materials
    material = materials.new('joints_skeleton_matrial')
    material.use_nodes = True
    clear_material(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    object_info = nodes.new(type='ShaderNodeObjectInfo')
    diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
    output = nodes.new(type='ShaderNodeOutputMaterial')

    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    links.new(object_info.outputs['Color'], diffuse.inputs['Color'])

    return material


def get_video_plane(video_file_pathname: str) -> any:
    video_capture = cv2.VideoCapture(video_file_pathname)
    video_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    # print(video_height, video_width)

    video_plane_scale = 2 * np.array([video_width, video_height]) / video_height
    # video_plane_scale = 2 * np.array([video_width, video_height]) / max(video_height, video_width)
    bpy.ops.mesh.primitive_plane_add(size=1,
                                     location=(1. + 0.5 * video_plane_scale[0], 0., 0.),
                                     rotation=(0., 0., 0.))
    video_plane_object = bpy.context.active_object
    video_plane_object.scale = (video_plane_scale[0], -video_plane_scale[1], 1.0)
    video_capture.release()

    return video_plane_object


def create_constant_video_material(video_file_pathname: str) -> any:
    video_capture = cv2.VideoCapture(video_file_pathname)
    video_number_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    materials = bpy.data.materials
    material = materials.new('constant_video')
    material.use_nodes = True
    clear_material(material)

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    texture = nodes.new(type='ShaderNodeTexImage')
    output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(texture.outputs['Color'], output.inputs['Surface'])

    texture.image = bpy.data.images.load(video_file_pathname)
    texture.image_user.use_auto_refresh = True
    texture.image_user.frame_duration = int(video_number_frames)

    return material


def create_skeleton_video_sample(videos_folder: str, joints_3d_animations_folder: str, joints_3d_animations_pca_folder: str,
                                 video_filename: str, joints_skeleton_material: any) -> Type[bpy.context.active_object]:
    video_filename_base = os.path.splitext(video_filename)[0]
    joints_filename = video_filename_base + '.npy'

    joints_filepath = os.path.join(joints_3d_animations_folder, joints_filename)
    joints_pca_filepath = os.path.join(joints_3d_animations_pca_folder, joints_filename)
    video_filepath = os.path.join(videos_folder, video_filename)

    if not os.path.exists(joints_filepath):
        exit()

    skeleton_animation = np.load(joints_filepath)
    skeleton_pca_animation = np.load(joints_pca_filepath)
    number_animation_frames = skeleton_animation.shape[0]
    number_skeleton_joints = skeleton_animation.shape[1]  # 17
    initial_joints_location = skeleton_animation[0]

    print(BColors.OKBLUE + BColors.BOLD, 'filename:', BColors.ENDC,
          BColors.OKGREEN, joints_filename, BColors.ENDC,
          BColors.OKBLUE + BColors.BOLD, '; number skeleton frames:', BColors.ENDC,
          BColors.WARNING, skeleton_animation.shape[0], BColors.ENDC)

    joints_spheres = [None] * number_skeleton_joints
    joints_colors = get_h36m_joints_colors()
    joints_names = get_h36m_joints_names()

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = number_animation_frames
    bpy.context.scene.frame_set(0)

    bpy.ops.object.empty_add(type='ARROWS', location=(0, 0, 0))
    bpy.context.active_object.name = video_filename
    joints_animation_coordinate_frame = bpy.context.active_object

    skeleton_curves = get_h36m_skeleton_curves(initial_joints_location, joints_colors)

    video_plane = get_video_plane(video_filepath)
    video_plane_material = create_constant_video_material(video_filepath)
    video_plane.data.materials.append(video_plane_material)
    video_plane.parent = joints_animation_coordinate_frame

    parent_skeleton_curves(skeleton_curves, joints_animation_coordinate_frame)
    assign_material_skeleton_curves(skeleton_curves, joints_skeleton_material)

    for joint_index in range(number_skeleton_joints):
        bpy.ops.mesh.primitive_ico_sphere_add(radius=0.025)
        bpy.ops.object.shade_smooth()
        bpy.context.active_object.color = joints_colors[joint_index]
        bpy.context.active_object.name = joints_names[joint_index]
        bpy.context.active_object.data.materials.append(joints_skeleton_material)
        joints_spheres[joint_index] = bpy.context.active_object

        for frame_index in range(number_animation_frames):
            joints_spheres[joint_index].location = skeleton_animation[frame_index, joint_index, :]
            joints_spheres[joint_index].keyframe_insert(data_path="location", frame=frame_index)

        joints_spheres[joint_index].parent = joints_animation_coordinate_frame

    for frame_index in range(number_animation_frames):
        joints_animation_coordinate_frame['pca'] = skeleton_pca_animation[frame_index]
        joints_animation_coordinate_frame.keyframe_insert(data_path='["pca"]', frame=frame_index)

    bpy.context.scene.frame_set(0)
    hook_skeleton_h36m_curves(skeleton_curves, joints_spheres)
    joints_animation_coordinate_frame.rotation_euler = (-0.5 * np.pi, 0, 0.5 * np.pi)

    return joints_animation_coordinate_frame


def run():
    root_directory = '/media/anton/4c95a564-35ea-40b5-b747-58d854a622d0/home/anton/work/fitMate/datasets/squats_2022_skeletons/results_base_video_mp3'

    videos_folder = os.path.join(root_directory, 'filtered_final_video')

    joints_3d_folder = os.path.join(root_directory, 'joints3d')
    joints_aligned_heights_folder = os.path.join(root_directory, 'joints3d_heights_aligned')
    joints_aligned_aligned_to_global_folder = os.path.join(root_directory, 'joints3d_aligned_to_global_frame')
    joints_stacked_folder = os.path.join(root_directory, 'joints3d_stacked')
    joints_pca_folder = os.path.join(root_directory, 'joints3d_pca')

    video_files_extensions = ['.mp4', '.mkv', '.webm']
    grid_width = 6

    video_filenames = [f for f in os.listdir(videos_folder) if os.path.splitext(f)[1] in video_files_extensions]
    joints_skeleton_material = create_joints_skeleton_material()

    for video_index, video_filename in enumerate(video_filenames):
        grid_index_row = video_index // grid_width
        grid_index_column = video_index - grid_width * grid_index_row

        sample_frame = create_skeleton_video_sample(videos_folder, joints_3d_folder, joints_pca_folder, video_filename, joints_skeleton_material)
        sample_frame.location[1] = 5.0 * grid_index_column
        sample_frame.location[2] = 5.0 * grid_index_row


run()
