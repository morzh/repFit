import warnings
from filters.steady_camera_filter.core.steady_camera_coarse_filter import SteadyCameraCoarseFilter
from cv_utils.video_segments_writer import VideoSegmentsWriter


def extract_coarse_steady_camera_video_segments(video_filepath, number_frames_to_average=15) -> list[range]:
    if number_frames_to_average < 10:
        warnings.warn(f'Value {number_frames_to_average} of number_frames_to_average is low, results could be non applicable')
    camera_filter = SteadyCameraCoarseFilter(video_filepath, number_frames_to_average=number_frames_to_average)
    camera_filter.process()
    return camera_filter.steady_camera_frames_ranges


def write_video_segments(video_filepath, segments):
    video_segments_writer = VideoSegmentsWriter(video_filepath)
    video_segments_writer.write(segments)
