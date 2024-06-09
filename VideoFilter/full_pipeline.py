from paths import VIDEO_DPATH, CUT_VIDEO_DPATH
from utils.multiprocess import run_pool

from extract_stable_tracks import extract_stable_tracks
from cut_boxes import cut_video_by_yolo_boxes
from make_mp_joints import make_mp_joints

from track_filter import (
    filter_video_by_joints_credibility,
    filter_by_joints_count,
    found_video_with_only_one_track_id
)


def first_step():
    base_videos = list(VIDEO_DPATH.glob('*'))
    filter_video_by_joints_credibility()
    exit(0)
    run_pool(extract_stable_tracks, base_videos)
    extract_stable_tracks(base_videos[0])
    # cut_video_by_yolo_boxes(base_videos[0])
    filter_by_joints_count()
    found_video_with_only_one_track_id()



    # run_pool(cut_video_by_yolo_boxes, base_videos)
    #



    # Then run AlphaPose for add files into joints2d* folders and run second_step()


def second_step():
    """ After AlphaPose """
    filter_by_joints_count()
    found_video_with_only_one_track_id()

    # now build 3d joints with MotionBert


def single_video_processing():
    video_fpath = list(VIDEO_DPATH.glob('*'))[0]
    extract_stable_tracks(video_fpath)
    cut_video_by_yolo_boxes(video_fpath)
    cut_videos = list(CUT_VIDEO_DPATH.glob('*'))
    for f in cut_videos:
        make_mp_joints(f)
    filter_video_by_joints_credibility()


def single_video_second_step():
    """ After AlphaPose """
    filter_by_joints_count()
    found_video_with_only_one_track_id()

    # now build 3d joints with MotionBert


if __name__ == '__main__':
    first_step()
    # second_step()