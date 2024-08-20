import os
import pprint

from utils.youtube_links_database.database_promts import chapters_data_via_promts
from utils.youtube_links_database.download_video import download_youtube_video_chapter


def download_chapters_squats_non_compound(database_filepath: str, promts_filepath: str, output_folder: str, **kwargs) -> None:
    """
    Description:

    :param database_filepath:
    :param promts_filepath:
    :param output_folder: folder for downloaded videos

    :key each_include_promt_to_separate_folder:
    :key video_chapter_offset_seconds:
    :key video_format:
    :key video_quality:
    :key video_format:
    :key print_links:
    :key print_chapters_data:
    :key print_chapters_name:
    :key use_proxy:
    """

    key_per_include_promt = kwargs['each_include_promt_to_separate_folder']
    chapters_promts = chapters_data_via_promts(database_filepath, promts_filepath)
    total_chapters_number = sum(len(l) for l in chapters_promts.values())
    print('Overall number of chapters is', total_chapters_number)

    offset = kwargs.get('video_chapter_offset_seconds', 1)
    for chapter_folder, chapters_data in chapters_promts.items():
        current_output_folder = os.path.join(output_folder, chapter_folder) if key_per_include_promt else output_folder
        os.makedirs(current_output_folder, exist_ok=True)

        if kwargs['print_chapters_name']: print(chapter_folder, 'where chapters number are', len(chapters_data))
        if kwargs.get('print_chapters_data', False): pprint.pprint(chapters_data, indent=6, width=150)

        for chapter in chapters_data:
            current_video_id = chapter[5]
            current_time_start = max(0, int(chapter[2]) - offset)
            current_time_end = int(chapter[3]) + offset
            current_output_file_basename = f'{current_video_id}__{current_time_start}-{current_time_end}__'
            current_output_filepath = os.path.join(current_output_folder, current_output_file_basename)

            download_youtube_video_chapter(current_video_id,
                                           current_output_filepath,
                                           (current_time_start, current_time_end),
                                           **kwargs)


if __name__ == '__main__':
    download_parameters = {
        'each_include_promt_to_separate_folder': True,
        'video_chapter_offset_seconds': 8,
        'video_format': 'mp4',
        'video_quality': 720,
        'print_links': True,
        'print_chapters_data': True,
        'print_chapters_name': True,
        'use_proxy': True,
    }

    database_folder ='/home/anton/work/fitMate/datasets'
    database_filename = 'youtube_rep_fit_database.db'
    database_file_path = os.path.join(database_folder, database_filename)

    promts_folder = '/home/anton/work/fitMate/datasets/exercises_filter_promts'
    promts_filename = 'squats_non_compound.json'
    promts_file_path = os.path.join(promts_folder, promts_filename)

    output_videos_folder = '/home/anton/work/fitMate/datasets/squats_non_compound'

    download_chapters_squats_non_compound(database_file_path, promts_file_path, output_videos_folder, **download_parameters)
