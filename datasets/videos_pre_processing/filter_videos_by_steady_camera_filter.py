from loguru import logger

from filters.steady_camera.steady_camera_video_segments import read_yaml
from filters.steady_camera.steady_camera_video_segments import steady_camera_filter


if __name__ == '__main__':
    config_io = read_yaml('configs/squats_chapters.yaml')
    logger.add('cut_videos_by_steady_camera_filter.log', format="{time} {message}", level="DEBUG", retention="5 days", compression='zip')
    steady_camera_filter(**config_io)
