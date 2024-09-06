import os

from filters.steady_camera.extract_video_segmens import read_yaml
from utils.youtube_links_database.videos_download import download_youtube_videos


if __name__ == '__main__':
    folders_parameters = read_yaml('download_squats_io_folders.yaml')
    download_parameters = read_yaml('download_video_chapters_parameters.yaml')

    database_input = folders_parameters['database']
    database_file_path = os.path.join(database_input['folder'], database_input['filename'])

    promts_folder = '/home/anton/work/fitMate/datasets/exercises_filter_promts'
    promts_filename = 'squats_non_compound.json'
    promts_file_path = os.path.join(promts_folder, promts_filename)

    output_videos_folder = '/home/anton/work/fitMate/datasets/squats_non_compound'

    download_youtube_videos(database_file_path, promts_file_path, output_videos_folder, **download_parameters)


