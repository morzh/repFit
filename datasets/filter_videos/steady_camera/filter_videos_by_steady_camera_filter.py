from loguru import logger

from core.utils.io.files_operations import read_yaml
from core.filters.steady_camera.core.steady_camera_tools import process_videos_by_steady_camera_filter


if __name__ == '__main__':
    input_output_config = read_yaml('configs_input_output/squats_chapters.yaml')
    steady_camera_filter_parameters = read_yaml('configs_filter/steady_camera_filter_parameters.yaml')
    logger.add('cut_videos_by_steady_camera_filter.log', format="{time} {message}", level="DEBUG", retention="5 days", compression='zip')
    process_videos_by_steady_camera_filter(input_output_config, steady_camera_filter_parameters)
