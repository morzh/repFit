import os

from filters.steady_camera.extract_video_segmens import read_yaml
from utils.youtube_links_database.videos_download import download_youtube_videos


if __name__ == '__main__':
    folders_parameters = read_yaml('download_squats_io_folders.yaml')
    download_parameters = read_yaml('download_video_parameters.yaml')

    database_input = folders_parameters['database']
    database_file_path = os.path.join(database_input['folder'], database_input['filename'])

    promts_input = folders_parameters['video_chapters_promts']
    promts_file_path = os.path.join(promts_input['folder'], promts_input['filename'])

    output_folder = folders_parameters['output_folder']

    download_youtube_videos(database_file_path, promts_file_path, output_folder, **download_parameters)


