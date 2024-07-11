"""Script for single-gpu/multi-gpu demo."""
import argparse
import json
import os
import sys
import torch
from tqdm import tqdm
from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.writer import DataWriter
from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt
from multiprocessing import Pool

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
# for i in {1..200}; do    python inference.py --nn $i; done
# python scripts/demo_inference.py
# --cfg configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml
checkpoint = "/home/ubuntu/PycharmProjects/FitMate/AlphaPose/pretrained_models/halpe26_fast_res50_256x192.pth"
configs = "/home/ubuntu/PycharmProjects/FitMate/AlphaPose/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml"
parser.add_argument('--cfg', type=str, default=configs, help='experiment configure file name')
parser.add_argument('--nn', type=int, help='number of video')
parser.add_argument('--checkpoint', type=str, default=checkpoint, help='checkpoint file name')

parser.add_argument('--detector', dest='detector', help='detector name', default="yolo")
parser.add_argument('--detfile', dest='detfile', help='detection result file', default="")
parser.add_argument('--indir', dest='inputpath', help='image-directory', default="")
parser.add_argument('--list', dest='inputlist', help='image-list', default="")
parser.add_argument('--image', dest='inputimg', help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath', help='output-directory', default="data/joints2d")
parser.add_argument('--save_img', default=False, action='store_true', help='save result as image')
parser.add_argument('--vis', default=False, action='store_true', help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true', help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0, help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5, help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=64,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true', help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true', help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video', help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int, help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video', help='whether to save rendered video',
                    default=True, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast', help='use fast rendering', action='store_true',
                    default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow', help='track humans in video with PoseFlow',
                    action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track', help='track humans in video with reid',
                    action='store_true', default=False)

args = parser.parse_args()
cfg = update_config(args.cfg)

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = args.pose_track or args.pose_flow or args.detector == 'tracker'


n_process = 5

sys.path.append("/home/ubuntu/PycharmProjects/FitMate/repFit/filters")


from video_filter.paths import (
    FILTERED_VIDEO_DPATH,
    JOINTS2d_TRACK_DPATH,
    JOINTS2d_EXTRA_INFO_DPATH,
    VIDEO_WITH_2D_JOINTS
)


def proc(video_name):
    save_video_fpath = VIDEO_WITH_2D_JOINTS / video_name
    save_joints_fpath = (JOINTS2d_TRACK_DPATH / video_name).with_suffix(".json")

    if save_joints_fpath.is_file():
        print(f"Skip exist file {video_name}")
        return
    print(f"Start processing of file {video_name}")
    # Load detection loader
    input_source = os.path.join(FILTERED_VIDEO_DPATH, video_name)
    det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=args.detbatch, mode='video', queueSize=args.qsize)

    # Init data writer
    save_opt = video_save_opt.copy()
    save_opt['savepath'] = str(save_video_fpath)
    save_opt.update(det_loader.videoinfo)
    writer = DataWriter(cfg, args, str(save_video_fpath), str(save_joints_fpath), save_video=True, video_save_opt=save_opt)

    data_len = det_loader.length - 2
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)
    tcfg.frame_rate = writer.video_save_opt['fps']
    tracker = Tracker(tcfg, args)

    batchSize = args.posebatch
    if args.flip:
        batchSize = int(batchSize / 2)

    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print('Loading pose model from %s...' % (args.checkpoint,))
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)

    pose_model.to(args.device)
    pose_model.eval()

    for i in im_names_desc:
        with torch.no_grad():
            inps, orig_img, im_name, boxes, scores, ids, cropped_boxes = det_loader.read()
            if orig_img is None:
                break
            if boxes is None or boxes.nelement() == 0:
                writer.update(None, None, None, None, None, orig_img, im_name)
                continue
            # Pose Estimation
            inps = inps.to(args.device)
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                if args.flip:
                    inps_j = torch.cat((inps_j, flip(inps_j)))
                hm_j = pose_model(inps_j)
                if args.flip:
                    hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs,
                                             shift=True)
                    hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                hm.append(hm_j)
            hm = torch.cat(hm)
            # START TRACKING
            hm = hm.cpu().data.numpy()

            online_targets = tracker.update(
                inps, boxes, hm, cropped_boxes, im_name, scores, _debug=False
            )

            new_boxes, new_scores, new_ids, new_hm, new_crop = [], [], [], [], []

            for t in online_targets:
                new_boxes.append(t.tlbr)
                new_crop.append(t.crop_box)
                new_hm.append(t.pose)
                new_ids.append(t.track_id)
                new_scores.append(t.detscore)

            new_hm = torch.Tensor(new_hm).to(args.device)
            boxes, scores, ids, hm, cropped_boxes = torch.tensor(
                new_boxes), new_scores, new_ids, new_hm, new_crop

            hm = hm.cpu()
            writer.update(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)

    result = {i: [] for i in range(len(writer.final_result))}
    for i, frame_result in enumerate(writer.final_result):
        for frame_result2 in frame_result['result']:
            result[i].append({'kp_score': frame_result2['kp_score'].tolist(),
                              'proposal_score': frame_result2['proposal_score'].tolist()})

    writer.stop()
    with open((JOINTS2d_EXTRA_INFO_DPATH/video_name).with_suffix(".json"), 'w') as file:
        json.dump(result, file)

    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')


if __name__ == "__main__":
    videos = os.listdir(FILTERED_VIDEO_DPATH)
    # proc(videos[0])
    # for video in videos:
    #     proc(video)
    with Pool(n_process) as p:
        p.map(proc, videos)
