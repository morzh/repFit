from loguru import logger

from filters.steady_camera.extract_video_segmens import read_yaml
from filters.steady_camera.extract_video_segmens import steady_camera_filter


if __name__ == '__main__':
    parameters = read_yaml('configs/squats_chapters.yaml')
    logger.add('cut_videos_by_steady_camera_filter.log', format="{time} {message}", level="DEBUG", retention="5 days", compression='zip')
    steady_camera_filter(**parameters)
