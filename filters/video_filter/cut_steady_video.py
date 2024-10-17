from pathlib import Path


from paths import STEADY_VIDEO_DPATH, RESULTS_ROOT
from utils.file_reader import read_pickle
from utils.cv.video_tools import trim

camera_steady = read_pickle(RESULTS_ROOT/"camera_steady_segments.pickle")
video_seg = {yid: seg for fname, _, yid, seg in zip(*list(camera_steady.values()))}


def trim_video_by_steady(video_fpath: Path):
    video_fname = video_fpath.stem
    video_youtube_id = video_fname[-11:]
    if video_youtube_id not in video_seg:
        print(f"skip video {video_fname} by steady filter")
        return

    for seg_n, seg in enumerate(video_seg[video_youtube_id]):
        cut_video_fpath = STEADY_VIDEO_DPATH / (video_youtube_id + f"_{seg_n}.mp4")
        if cut_video_fpath.is_file():
            print(f"Skip already exist fragment {str(cut_video_fpath)}")
            return
        trim(video_fpath, cut_video_fpath, *seg)
        print(f"Save cut video {str(cut_video_fpath)}")
