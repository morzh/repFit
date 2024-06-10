from paths import VIDEO_DPATH, STEADY_VIDEO_DPATH
from utils.multiprocess import run_pool

from extract_stable_tracks import extract_stable_tracks
from cut_boxes import cut_videos_by_filters
from cut_steady_video import trim_video_by_steady

from track_filter import (
    filter_by_joints_credibility,
    filter_by_joints_count,
    found_video_with_only_one_track_id
)


def run():
    base_videos = list(VIDEO_DPATH.glob('*'))
    run_pool(trim_video_by_steady, base_videos)

    steady_videos = list(STEADY_VIDEO_DPATH.glob('*'))
    run_pool(extract_stable_tracks, steady_videos)
    exit(0)
    joints_credibility = filter_by_joints_credibility()
    joints_count = filter_by_joints_count()
    filter_result = merge_filters(joints_credibility, joints_count)

    cut_videos_by_filters(steady_videos, filter_result)



def first_step():
    # base_videos = list(VIDEO_DPATH.glob('*'))
    base_videos = list(VIDEO_DPATH.glob('*'))
    # for v in base_videos:
    #     trim_video_by_steady(v)

    run_pool(trim_video_by_steady, base_videos, 4)
    exit(0)
    joints_credibility = filter_by_joints_credibility()
    joints_count = filter_by_joints_count()
    filter_result = merge_filters(joints_credibility, joints_count)

    steady_videos = list(STEADY_VIDEO_DPATH.glob('*'))
    cut_videos_by_filters(steady_videos, filter_result)
    exit(0)
    # run_pool(extract_stable_tracks, base_videos)
    # extract_stable_tracks(base_videos[0])
    # cut_video_by_yolo_boxes(base_videos[0])

    found_video_with_only_one_track_id()


def merge_filters(*args, drop_filtered=True) -> dict:
    result = args[0]
    if len(result) == 1:
        return result

    for key in result:
        for filter_data in args[1:]:
            result[key].extend(filter_data[key])

    for key in result:
        filtered_indexes = []
        for v in set(result[key]):
            if result[key].count(v)>1:
                filtered_indexes.append(v)
        result[key] = filtered_indexes

    if drop_filtered:
        filter_result = {}
        for key in result:
            if result[key]:
                filter_result[key] = result[key]
        return filter_result
    return result


if __name__ == '__main__':
    run()