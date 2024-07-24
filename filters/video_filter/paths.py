from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS_DPATH = PROJECT_ROOT / "datasets"
RESULTS_ROOT = PROJECT_ROOT / "results"


RESULTS_DPATH = DATASETS_DPATH / "PCA_5.07.24" / "results"
# RESULTS_DPATH = RESULTS_ROOT / "squats_2022_steady"

VIDEO_DPATH = DATASETS_DPATH / "PCA_5.07.24" / "filtered_final_video"

# VIDEO_DPATH = DATASETS_DPATH / "base_videos"
# VIDEO_DPATH = DATASETS_DPATH / "squats_2022_coarse_steady_camera_yolo_segmentation-m" / "steady"

# RESULTS_DPATH = RESULTS_ROOT / "video_filter_v2.0"


YOLO_BBOXES_DPATH = RESULTS_DPATH / "yolo_bboxes"
STABLE_FILTER_DPATH = RESULTS_DPATH / "stable_filtered"
STEADY_VIDEO_DPATH = RESULTS_DPATH / "steady_video"

STEADY_VIDEO_DPATH = VIDEO_DPATH

# FILTERED_VIDEO_DPATH = RESULTS_DPATH / "filtered_video"
FILTERED_VIDEO_DPATH = VIDEO_DPATH

JOINTS2d_EXTRA_INFO_DPATH = RESULTS_DPATH / "joints2d_info"
JOINTS2d_TRACK_DPATH = RESULTS_DPATH / "joints2d_track"
CRED_FILTERED_VIDEO_DPATH = RESULTS_DPATH / "cred_filtered_video"
ONE_PERSON_FILTERED_VIDEO_DPATH = RESULTS_DPATH / "joints2d_one_person"

VIDEO_WITH_2D_JOINTS = RESULTS_DPATH / "video_with_joints"
FINAL_VIDEOS_DPATH = RESULTS_DPATH / "final_videos"
JOINTS_3D_DPATH = RESULTS_DPATH / "joints3d"

for path in [YOLO_BBOXES_DPATH, STEADY_VIDEO_DPATH, FILTERED_VIDEO_DPATH,
             CRED_FILTERED_VIDEO_DPATH, ONE_PERSON_FILTERED_VIDEO_DPATH, JOINTS2d_TRACK_DPATH,
             JOINTS2d_EXTRA_INFO_DPATH, VIDEO_WITH_2D_JOINTS, FINAL_VIDEOS_DPATH, JOINTS_3D_DPATH,
             STABLE_FILTER_DPATH]:
    path.mkdir(exist_ok=True, parents=True)

video_stats_fpath = RESULTS_DPATH / 'cut_video_stats.json'
