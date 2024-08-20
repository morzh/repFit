import os
import pprint
from utils.youtube_links_database.database_promts import chapters_data_via_promts, convert_chapters_data_to_links

class Color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def links_squats_non_compound(database_filepath, promts_filepath, **kwargs):
    """
    Description:

    :key dataset_folder: dataset folder
    :key promts_folder: promts folder (with .json file)
    :key database_filename: SQLite3 database filename
    :key promts_filename: promts filename
    :key print_links: print chapter links
    :key verbose:
    """
    verbose = kwargs.get('verbose', True)
    print_links = kwargs.get('print_links', False)

    chapters_promts = chapters_data_via_promts(database_filepath, promts_filepath)
    chapters_links = {}
    total_chapter_number = 0

    for chapter_folder, chapters_data in chapters_promts.items():
        print(Color.YELLOW + chapter_folder + Color.END)
        current_chapters_links = convert_chapters_data_to_links(chapters_data)
        chapters_links[chapter_folder] = current_chapters_links
        if print_links:
            pprint.pprint(current_chapters_links, indent=8, width=150)

        if verbose:
            current_number_chapters = len(chapters_data)
            total_chapter_number += current_number_chapters
            print(f'\t Number of chapters is {current_number_chapters}')

    if verbose:
        print('-' * 30)
        print(Color.GREEN +  f'Total number of chapters is {total_chapter_number}' + Color.END)

    return chapters_links


if __name__ == '__main__':
    database_folder = '/home/anton/work/fitMate/datasets'
    promts_folder = '/home/anton/work/fitMate/datasets/exercises_filter_promts'
    database_filename = 'youtube_rep_fit_database.db'
    promts_filename = 'squats_non_compound.json'

    database_file_path = os.path.join(database_folder, database_filename)
    promts_file_path = os.path.join(promts_folder, promts_filename)

    parameters = {
        'print_links': False,
        'verbose': True,
    }

    chapters_links = links_squats_non_compound(database_file_path, promts_file_path, **parameters)

    # pprint.pprint(chapters_links)