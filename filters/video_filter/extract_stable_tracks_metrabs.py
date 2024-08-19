import os.path
from ultralytics import YOLO
from paths import YOLO_BBOXES_DPATH, VIDEO_DPATH, STABLE_FILTER_DPATH
from pathlib import Path
from constants import min_track_length_sec, min_occurrence_rate
from utils.cv.video_reader import VideoReader
from utils.file_reader import write_pickle
import tensorflow as tf
import tensorflow_hub as tfhub

class HumanTrackerMetrabs:
    def __init__(self, model_name: str = 'https://bit.ly/metrabs_l'):
        self.detector_model = tfhub.load(model_name)
        self.video_reader = None

    def extract_tracks(self, video_fpath: str) -> dict:
        if not os.path.isfile(video_fpath):
            raise Exception(f"Video {video_fpath} was not found")

        self.video_reader = VideoReader(video_fpath)
        tracks = {}
        # image = tf.image.decode_jpeg(tf.io.read_file('img/test_image_3dpw.jpg'))
        # pred['boxes'], pred['poses2d'], pred['poses3d']

        for frame in self.video_reader.frame_generator():
            pred = self.detector_model.detect_poses(frame)


        return tracks


def extract_stable_tracks(video_fpath: Path):
    """ Make files with bboxes info """
    video_fname = video_fpath.name
    try:
        result_fpath = (YOLO_BBOXES_DPATH / video_fname).with_suffix('.pickle')
        filtered_fpath = (STABLE_FILTER_DPATH / video_fname).with_suffix('.pickle')
        # if result_fpath.is_file():
        #     print(f"Skip processed video {video_fname}. Result fpath: {str(result_fpath)}")
        #     return
        result_fpath.parents[0].mkdir(exist_ok=True, parents=True)
        print(f"Start processing of video {str(video_fpath)}")

        tracker = HumanTrackerMetrabs()
        tracks = tracker.extract_tracks(video_fpath)
        return
        write_pickle(tracks, result_fpath)

        min_track_length_frames = min_track_length_sec * tracker.video_reader.fps
        stable_tracks = tracks_filter(tracks, min_track_length_frames)

        write_pickle(stable_tracks, filtered_fpath)
        print(f"Save stable track detection results to: {str(filtered_fpath)}")
    except Exception as ex:
        print(ex)


def tracks_filter(tracks: dict, min_track_length_frames: int):
    stable_tracks = {}
    for idx, t in tracks.items():
        if len(t) > min_track_length_frames:
            idxs = list(t.keys())
            start_idx = int(idxs[0])
            stop_idx = int(idxs[-1])
            if len(t) / (stop_idx - start_idx) > min_occurrence_rate:
                stable_tracks[idx] = t
    return stable_tracks


if __name__ == '__main__':
    videos = list(VIDEO_DPATH.glob('*'))
    extract_stable_tracks(videos[0])