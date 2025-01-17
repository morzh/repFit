from loguru import logger
from core.utils.io.files_operations import read_yaml
from core.filters.single_person.source.single_persons_tools import process_videos_by_single_persons_filter


if __name__ == '__main__':
    input_output_config = read_yaml('parameters_input_output/squats_base_videos.yaml')
    single_persons_filter_parameters = read_yaml('parameters_filter/single_persons_filter_parameters.yaml')
    logger.add('logs/filter_videos_by_single_persons_filter.log', format="{time} {message}", level="DEBUG", retention="5 days", compression='zip')
    process_videos_by_single_persons_filter(input_output_config, single_persons_filter_parameters)
