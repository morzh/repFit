from paths import VIDEO_DPATH
from pathlib import Path
from cv_utils.video_reader import VideoReader
import time


def run():
    for video_fpath in list(VIDEO_DPATH.glob("*.mp4"))[1:]:
        video_reader = VideoReader(video_fpath)
        print(str(video_fpath))
        for i, frame in enumerate(video_reader.frame_generator()):
            video_reader.imshow(frame)
            time.sleep(0.05)
            pass



if __name__ == '__main__':
    run()