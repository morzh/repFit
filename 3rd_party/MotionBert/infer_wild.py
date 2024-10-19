import os
import numpy as np
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs_input_output/pose3d/MB_ft_h36m_global_lite.yaml",
                        help="Path to the config file.")
    parser.add_argument('-e', '--evaluate',
                        default='checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin',
                        type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-j', '--json_path',
                        default="/home/ubuntu/PycharmProjects/FitMate/AlphaPose/data/AlphaPose/joints2d/1 1_2 kyykky - 1 1_2 squat-I2uXMPxPkKk.mp4.json",
                        type=str, help='alphapose detection result json path')
    parser.add_argument('-v', '--vid_path',
                        default='/home/ubuntu/PycharmProjects/FitMate/AlphaPose/data/videos/1 1_2 kyykky - 1 1_2 squat-I2uXMPxPkKk.mp4',
                        type=str, help='video path')
    parser.add_argument('-o', '--out_path',
                        default='/home/ubuntu/PycharmProjects/FitMate/AlphaPose/data/MotionBERT',
                        type=str, help='output path')
    parser.add_argument('--pixel', action='store_true', help='align with pixle coordinates')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    opts = parser.parse_args()
    return opts


opts = parse_args()
args = get_config(opts.config)

model_backbone = load_backbone(args)
if torch.cuda.is_available():
    model_backbone = nn.DataParallel(model_backbone)
    model_backbone = model_backbone.cuda()

print('Loading checkpoint', opts.evaluate)
checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)
model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
model_pos = model_backbone
model_pos.eval()
testloader_params = {
    'batch_size': 1,
    'shuffle': False,
    # 'num_workers': 0,
    'pin_memory': True,
    # 'prefetch_factor': 4,
    # 'persistent_workers': True,
    'drop_last': False
}

sys.path.append("/home/ubuntu/PycharmProjects/FitMate/repFit")
from filters.video_filter.paths import (
    JOINTS2d_TRACK_DPATH,
    VIDEO_WITH_2D_JOINTS,
    FINAL_VIDEOS_DPATH,
    JOINTS_3D_DPATH
)

os.makedirs(opts.out_path, exist_ok=True)

joints2d_files = os.listdir(JOINTS2d_TRACK_DPATH)
for joints_fname in joints2d_files:
    print(joints_fname)
    joints_fpath = os.path.join(JOINTS2d_TRACK_DPATH, joints_fname)

    try:
        wild_dataset = WildDetDataset(joints_fpath, clip_len=opts.clip_len, scale_range=[1, 1],
                                      focus=opts.focus)
    except Exception as ex:
        continue
    test_loader = DataLoader(wild_dataset, **testloader_params)

    results_all = []
    with torch.no_grad():
        batches = list(test_loader)
        for batch_input in tqdm(batches):
            N, T = batch_input.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            # if args.no_conf:
            #     batch_input = batch_input[:, :, :, :2]
            # if args.flip:
            batch_input_flip = flip_data(batch_input)
            predicted_3d_pos_1 = model_pos(batch_input)
            predicted_3d_pos_flip = model_pos(batch_input_flip)
            predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
            predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
            # else:
            #     predicted_3d_pos = model_pos(batch_input)
            # if args.rootrel:
            #     predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
            # else:
            predicted_3d_pos[:, 0, 0, 2] = 0
            # pass
            # if args.gt_2d:
            #     predicted_3d_pos[..., :2] = batch_input[..., :2]
            results_all.append(predicted_3d_pos.cpu().numpy())

    results_all = np.hstack(results_all)
    results_all = np.concatenate(results_all)
    # render_and_save(results_all, '%s/X3D.mp4' % (opts.out_path), keep_imgs=False, fps=fps_in)
    # if opts.pixel:
    #     # Convert to pixel coordinates
    #     results_all = results_all * (min(vid_size) / 2.0)
    #     results_all[:, :, :2] = results_all[:, :, :2] + np.array(vid_size) / 2.0

    shutil.copyfile(
        (VIDEO_WITH_2D_JOINTS / joints_fname).with_suffix(".mp4"),
        (FINAL_VIDEOS_DPATH / joints_fname).with_suffix(".mp4")
    )
    joints_3d_fpath = (JOINTS_3D_DPATH / joints_fname).with_suffix(".npy")
    np.save(joints_3d_fpath, results_all)
    print(f"Save joints3d {joints_3d_fpath}")
