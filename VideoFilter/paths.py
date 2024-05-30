from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS_DPATH = PROJECT_ROOT / "datasets"
RESULTS_ROOT = PROJECT_ROOT / "results"

VIDEO_DPATH = DATASETS_DPATH / "base_videos"

RESULTS_DPATH = RESULTS_ROOT / "video_filter_v1.0"

YOLO_BBOXES_DPATH = RESULTS_DPATH / "yolo_bboxes"
CUT_VIDEO_DPATH = RESULTS_DPATH / "cut_video"
FILTERED_VIDEO_DPATH = RESULTS_DPATH / "filtered_video"
JOINTS_MP_DPATH = RESULTS_DPATH / "mediapipe_joints"

JOINTS2d_EXTRA_INFO_DPATH = RESULTS_DPATH / "joints2d_info"
JOINTS2d_TRACK_DPATH = RESULTS_DPATH / "joints2d_track"
CRED_FILTERED_VIDEO_DPATH = RESULTS_DPATH / "cred_filtered_video"
ONE_PERSON_FILTERED_VIDEO_DPATH = RESULTS_DPATH / "joints2d_one_person"

VIDEO_WITH_2D_JOINTS = RESULTS_DPATH / "video_with_joints"
FINAL_VIDEOS_DPATH = RESULTS_DPATH / "final_videos"
JOINTS_3D_DPATH = RESULTS_DPATH / "joints3d"

for path in [YOLO_BBOXES_DPATH, CUT_VIDEO_DPATH, FILTERED_VIDEO_DPATH, JOINTS_MP_DPATH,
             CRED_FILTERED_VIDEO_DPATH, ONE_PERSON_FILTERED_VIDEO_DPATH, JOINTS2d_TRACK_DPATH,
             JOINTS2d_EXTRA_INFO_DPATH, VIDEO_WITH_2D_JOINTS, FINAL_VIDEOS_DPATH, JOINTS_3D_DPATH]:
    path.mkdir(exist_ok=True, parents=True)

video_stats_fpath = RESULTS_DPATH / 'cut_video_stats.json'
