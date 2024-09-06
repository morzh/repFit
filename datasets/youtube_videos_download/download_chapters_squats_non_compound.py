import os
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

    database_folder = '/home/anton/work/fitMate/datasets'
    database_filename = 'youtube_rep_fit_database.db'
    database_file_path = os.path.join(database_folder, database_filename)

    promts_folder = '/home/anton/work/fitMate/datasets/exercises_filter_promts'
    promts_filename = 'squats_non_compound.json'
    promts_file_path = os.path.join(promts_folder, promts_filename)

    output_videos_folder = '/home/anton/work/fitMate/datasets/squats_non_compound'

    download_video_chapters_from_youtube(database_file_path, promts_file_path, output_videos_folder, **download_parameters)
