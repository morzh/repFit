import os

from filters.steady_camera.steady_camera_video_segments import read_yaml
from utils.youtube_links_database.video_chapters_download import download_video_chapters_from_youtube


if __name__ == '__main__':
    download_parameters = {
        'each_include_promt_to_separate_folder': True,
        'video_chapter_offset_seconds': 8,
        'video_format': 'mp4',
        'video_quality': 720,
        'print_chapters_links': False,
        'print_chapters_data': False,
        'print_chapters_name': False,
        'write_chapters_links': True,
        'chapters_links_filepath': 'chapters_filenames_links.txt',
        'use_proxy': True,
    }
    folders_parameters = read_yaml('configs/download_squats_io_folders.yaml')

    database_input = folders_parameters['database']
    database_file_path = str(os.path.join(database_input['folder'], database_input['filename']))

    promts_input = folders_parameters['video_chapters_promts']
    promts_file_path = str(os.path.join(promts_input['folder'], promts_input['filename']))

    output_folder = folders_parameters['output_folder']

    download_video_chapters_from_youtube(database_file_path, promts_file_path, output_folder, **download_parameters)
