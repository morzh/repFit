import os.path
from pathlib import Path
from filters.persons_tracking.core.persons_tracking import  PersonsTracker


root_folder = '/media/anton/4c95a564-35ea-40b5-b747-58d854a622d0/home/anton/work/fitMate/datasets'
source_folder = os.path.join(root_folder, 'squats_2022_coarse_steady_camera_yolo_segmentation_yolov9-c/steady')
target_folder = os.path.join(root_folder, 'squats_2022_steady_tracking')

tracking = PersonsTracker()
source_video_folder = Path(source_folder)
source_video_filepaths = source_video_folder.glob('*.mp4')

for video_filepath in source_video_filepaths:
    tracking.extract_tracks(str(video_filepath), target_folder)
