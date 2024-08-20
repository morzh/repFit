import os
import pprint

from utils.youtube_links_database.database_promts import chapters_data_via_promts
from utils.youtube_links_database.download_video import download_youtube_video_chapter


def download_chapters_squats_non_compound(**kwargs) -> None:
    """
    Description:

    :key output_folder: folder for downloaded videos
    :key database_folder:
    :key promts_folder:
    """

    output_folder = kwargs.get('output_folder')
    database_folder = kwargs.get('database_folder')
    promts_folder = kwargs.get('promts_folder')

    database_filename = kwargs.get('database_filename')
    promts_filename = kwargs.get('promts_filename')

    database_filepath = os.path.join(database_folder, database_filename)
    promts_input_filepath = os.path.join(promts_folder, promts_filename)

    key_per_include_promt = kwargs['each_include_promt_to_separate_folder']
    verbose = kwargs['verbose']
    chapters_promts = chapters_data_via_promts(database_filepath, promts_input_filepath, verbose=verbose)

    for chapter_folder, chapters_data in chapters_promts.items():
        if key_per_include_promt:
            current_output_folder = os.path.join(output_folder, chapter_folder)
        else:
            current_output_folder = output_folder

        if kwargs.get('print_chapters', False):
            pprint.pprint(chapters_data, width=120)

        for chapter in chapters_data:
            current_video_id = chapter[5]
            current_time_start = int(chapter[2])
            current_time_end = int(chapter[3])
            current_output_file_basename = f'{current_video_id}__{current_time_start}-{current_time_end}__'
            current_output_filepath = os.path.join(current_output_folder, current_output_file_basename)
            download_youtube_video_chapter(current_video_id,
                                           current_output_filepath,
                                           (current_time_start, current_time_end),
                                           **kwargs)


if __name__ == '__main__':
    download_parameters = {
        'output_folder': '/home/anton/work/fitMate/datasets/squats_non_compound',
        'database_folder': '/home/anton/work/fitMate/datasets',
        'promts_folder': '/home/anton/work/fitMate/datasets/exercises_filter_promts',
        'database_filename': 'youtube_rep_fit_database.db',
        'promts_filename': 'squats_non_compound.json',
        'each_include_promt_to_separate_folder': False,
        'video_chapter_offset_seconds': 8,
        'video_format': 'mp4',
        'video_quality': 720,
        'print_links': True,
        'print_chapters': False,
        'verbose': True,
        'use_proxy': True,
    }

    download_chapters_squats_non_compound(**download_parameters)
