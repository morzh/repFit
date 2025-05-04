import os

from filters.steady_camera.steady_camera_video_segments import read_yaml
from core.utils.youtube.videos_download import download_youtube_videos


if __name__ == '__main__':
    folders_parameters = read_yaml('configs_input_output/download_squats_io_folders.yaml')
    download_config = read_yaml('configs_input_output/download_video_parameters.yaml')

    database_input = folders_parameters['database']
    database_file_path = os.path.join(database_input['folder'], database_input['filename'])

    promts_input = folders_parameters['videos_promts']
    promts_file_path = os.path.join(promts_input['folder'], promts_input['filename'])

    output_folder = folders_parameters['output']['videos_output_folder']

    download_youtube_videos(database_file_path, promts_file_path, output_folder, download_config)
