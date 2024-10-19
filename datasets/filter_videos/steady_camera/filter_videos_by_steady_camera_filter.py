from loguru import logger

from utils.io.file_read_write import read_yaml
from filters.steady_camera.core.steady_camera_utils import process_videos_by_steady_camera_filter


if __name__ == '__main__':
    config_io = read_yaml('configs_input_output/squats_chapters.yaml')
    filter_parameters = read_yaml('configs_filter/steady_camera_filter_parameters.yaml')
    logger.add('cut_videos_by_steady_camera_filter.log', format="{time} {message}", level="DEBUG", retention="5 days", compression='zip')
    process_videos_by_steady_camera_filter(config_io, filter_parameters)
