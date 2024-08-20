import os
import pprint

from utils.youtube_links_database.database_promts import chapters_data_via_promts
from utils.youtube_links_database.download_video import download_youtube_video_chapter


def download_youtube_chapters_videos(database_filepath: str, promts_filepath: str, output_folder: str, **kwargs) -> None:
    """
    Description:
        This functions downloads chapters from YouTube.

    :param database_filepath: SQLite3 links database filepath
    :param promts_filepath: File path to .json file with include and exclude tokens for database requests
    :param output_folder: folder for downloaded videos

    :key each_include_promt_to_separate_folder: use include promt tokens to sort downloaded videos by subfolders
    :key video_chapter_offset_seconds: offsets YouTube chapters start and end time in seconds ([seconds_start - offset, seconds_end + offset])
    :key video_format: video format, e.g. 'mp4'
    :key video_quality: video quality, e.g. 720
    :key print_links: if True prints links to YouTube chapters
    :key print_chapters_data: if it is True prints chapters data
    :key print_chapters_name: if True prints chapters names
    :key use_proxy: proxy for connection to 'youtube.com'
    """

    key_per_include_promt = kwargs['each_include_promt_to_separate_folder']
    chapters_promts = chapters_data_via_promts(database_filepath, promts_filepath)
    video_format = kwargs.get('video_format','mp4')
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
            current_output_filename = f'{current_video_id}__{current_time_start}-{current_time_end}__.{video_format}'
            current_output_filepath = os.path.join(current_output_folder, current_output_filename)

            if os.path.exists(current_output_filepath) and os.stat(current_output_filepath).st_size > 1024:
                continue

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

    download_youtube_chapters_videos(database_file_path, promts_file_path, output_videos_folder, **download_parameters)
