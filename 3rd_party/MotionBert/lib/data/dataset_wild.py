import torch
import numpy as np
import ipdb
import glob
import os
import io
import math
import random
import json
import pickle
import math

from scipy.signal import impulse2
from torch.utils.data import Dataset, DataLoader
from lib.utils.utils_data import crop_scale
from scipy.interpolate import interp1d


def halpe2h36m(x):
    '''
        Input: x (T x V x C)  
       //Halpe 26 body keypoints
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
    {17,  "Head"},
    {18,  "Neck"},
    {19,  "Hip"},
    {20, "LBigToe"},
    {21, "RBigToe"},
    {22, "LSmallToe"},
    {23, "RSmallToe"},
    {24, "LHeel"},
    {25, "RHeel"},
    '''
    T, V, C = x.shape
    y = np.zeros([T, 17, C])
    y[:, 0, :] = x[:, 19, :]
    y[:, 1, :] = x[:, 12, :]
    y[:, 2, :] = x[:, 14, :]
    y[:, 3, :] = x[:, 16, :]
    y[:, 4, :] = x[:, 11, :]
    y[:, 5, :] = x[:, 13, :]
    y[:, 6, :] = x[:, 15, :]
    y[:, 7, :] = (x[:, 18, :] + x[:, 19, :]) * 0.5
    y[:, 8, :] = x[:, 18, :]
    y[:, 9, :] = x[:, 0, :]
    y[:, 10, :] = x[:, 17, :]
    y[:, 11, :] = x[:, 5, :]
    y[:, 12, :] = x[:, 7, :]
    y[:, 13, :] = x[:, 9, :]
    y[:, 14, :] = x[:, 6, :]
    y[:, 15, :] = x[:, 8, :]
    y[:, 16, :] = x[:, 10, :]
    return y


def read_input(json_path, vid_size, scale_range, length):
    with open(json_path, "r") as read_file:
        results = json.load(read_file)

    keypoints, image_ids = [], []
    for r in results:
        img_id = int(r['image_id'][:-4])
        img_keypoints = r['keypoints']
        if image_ids and image_ids[-1] == img_id:
            if len(image_ids) == 1:
                continue
            dist1 = np.sum(np.abs(np.array(keypoints[-2]) - img_keypoints))
            dist2 = np.sum(np.abs(np.array(keypoints[-2]) - np.array(keypoints[-1])))
            if dist2 > dist1:
                keypoints.pop(-1)
                image_ids.pop(-1)
            else:
                continue
        keypoints.append(img_keypoints)
        image_ids.append(img_id)

    non_image_ids = [i for i in range(length) if i not in image_ids]

    # if first skeleton is not in 0 frame, fill empty started frames with first met skeleton
    start_keypoints = []
    for i in range(image_ids[-1]):
        if i in non_image_ids:
            start_keypoints.append(keypoints[0])
            non_image_ids.pop(0)
            image_ids.insert(i, i)
        else:
            break

    stop_keypoints = []
    extra_indexes = []
    for i in range(length-1, 0, -1):
        if i in non_image_ids:
            stop_keypoints.append(keypoints[-1])
            non_image_ids.pop(-1)
            extra_indexes.append(i)
        else:
            break
    image_ids.extend(reversed(extra_indexes))
    keypoints = start_keypoints + keypoints + stop_keypoints

    if non_image_ids:
        try:
            # add extra points for avoid interpolation side artifacts
            n_extra_points = 10

            keypoints_array = np.empty((image_ids[-1] + 1, len(keypoints[0])))
            keypoints_array = np.empty((length, len(keypoints[0])))
            keypoints_array[image_ids] = keypoints

            keypoints = ([keypoints[0] for _ in range(n_extra_points)] +
                         keypoints +
                         [keypoints[-1] for _ in range(n_extra_points)])
            image_ids = ([i for i in range(image_ids[0] - n_extra_points, image_ids[0])] +
                         image_ids +
                         [i for i in range(image_ids[-1]+1, image_ids[-1] + n_extra_points + 1)])

            interp_func = interp1d(image_ids, np.array(keypoints), kind='cubic', axis=0)
            new_points = interp_func(non_image_ids)
            keypoints_array[non_image_ids] = new_points
        except Exception as ex:
            r=0
    else:
        keypoints_array = np.array(keypoints)
    # kpts_all = []
    # for item in results:
    #     if focus != None and item['idx'] != focus:
    #         continue
    #     kpts = np.array(item['keypoints']).reshape([-1, 3])
    #     kpts_all.append(kpts)
    # kpts_all = np.array(kpts_all)
    # kpts_all = halpe2h36m(kpts_all)

    kpts_all = halpe2h36m(keypoints_array.reshape([-1, 26, 3]))
    if kpts_all.shape[0] != length:
        d=0
    if vid_size:
        w, h = vid_size
        scale = min(w, h) / 2.0
        kpts_all[:, :, :2] = kpts_all[:, :, :2] - np.array([w, h]) / 2.0
        kpts_all[:, :, :2] = kpts_all[:, :, :2] / scale
        motion = kpts_all
    if scale_range:
        motion = crop_scale(kpts_all, scale_range)
    motion =  motion.astype(np.float32)
    return motion


class WildDetDataset(Dataset):
    def __init__(self, json_path, length=None, clip_len=243, vid_size=None, scale_range=None, focus=None):
        self.json_path = json_path
        self.clip_len = clip_len
        self.length = length
        self.vid_all = read_input(json_path, vid_size, scale_range, length)

    def __len__(self):
        'Denotes the total number of samples'
        return math.ceil(len(self.vid_all) / self.clip_len)

    def __getitem__(self, index):
        'Generates one sample of data'
        st = index * self.clip_len
        end = min((index + 1) * self.clip_len, len(self.vid_all))
        return self.vid_all[st:end]
