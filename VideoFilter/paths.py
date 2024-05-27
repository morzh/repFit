from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS_DPATH = PROJECT_ROOT / "datasets"
RESULTS_ROOT = PROJECT_ROOT / "results"

VIDEO_DPATH = DATASETS_DPATH / "base_videos"


RESULTS_DPATH = RESULTS_ROOT / "video_filter_v1.0"
INFO_DPATH = RESULTS_DPATH / "yolo_bboxes"
CUT_VIDEO_DPATH = RESULTS_DPATH / "cut_video"
FILTERED_VIDEO_DPATH = RESULTS_DPATH / "filtered_video"
JOINTS_DPATH = RESULTS_DPATH / "mediapipe_joints"
JOINTS2d_DPATH = RESULTS_DPATH / "joints2d"
JOINTS2d_INFO_DPATH = RESULTS_DPATH / "joints2d_info"
JOINTS2d_TRACK_DPATH = RESULTS_DPATH / "joints2d_track"
CREDIBILITY_FILTERED_DPATH = RESULTS_DPATH / "cred_filtered_video"
ONE_PERSON_FILTERED_DPATH = RESULTS_DPATH / "joints2d_one_person"

for path in [INFO_DPATH, CUT_VIDEO_DPATH, FILTERED_VIDEO_DPATH, JOINTS_DPATH, CREDIBILITY_FILTERED_DPATH,
             ONE_PERSON_FILTERED_DPATH, JOINTS2d_TRACK_DPATH]:
    path.mkdir(exist_ok=True, parents=True)
