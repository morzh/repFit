import os
import pprint
from utils.youtube_links_database.database_promts import chapters_links_via_promt
from utils.youtube_links_database.download_video import download_youtube_video_chapter


def download_chapters_squats_non_compound(output_folder: str, print_chapters=False):
    database_folder = '/home/anton/work/fitMate/datasets'
    promts_folder = '/home/anton/work/fitMate/datasets/exercises_filter_promts'

    database_filename = 'youtube_rep_fit_database.db'
    promts_input_filename = 'squats_non_compound.json'

    database_filepath = os.path.join(database_folder, database_filename)
    promts_input_filepath = os.path.join(promts_folder, promts_input_filename)

    chapters_data = chapters_links_via_promt(database_filepath, promts_input_filepath, return_links=False, verbose=True)

    if print_chapters:
        pprint.pprint(chapters_data, width=120)

    for chapter in chapters_data:
        current_video_id = chapter[5]
        current_chapter_name = chapter[1]
        current_time_start = chapter[2]
        current_time_end = chapter[3]
        current_output_file_basename = f'{current_video_id}__s{current_time_start}-{current_time_end}__'
        current_output_filepath = os.path.join(output_folder, current_output_file_basename)

        download_youtube_video_chapter(current_video_id, current_output_filepath, current_chapter_name, (current_time_start, current_time_end))


if __name__ == '__main__':
    output_folder = ''
    download_chapters_squats_non_compound(output_folder, print_chapters=False)
