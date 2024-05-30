from paths import VIDEO_DPATH, CUT_VIDEO_DPATH
from utils.multiprocess import run_pool

from extract_stable_tracks import extract
from cut_boxes import cut_video_by_yolo_boxes
from make_joints import make_joints


def run():
    base_videos = list(VIDEO_DPATH.glob('*'))
    run_pool(extract, base_videos)
    run_pool(cut_video_by_yolo_boxes, base_videos)

    cut_videos = list(CUT_VIDEO_DPATH.glob('*'))
    run_pool(make_joints, cut_videos)