import os
import pprint
from utils.youtube_links_database.database_promts import chapters_data_via_promts


def links_squats_non_compound(**kwargs):
    """
    Description:

    :key dataset_folder: dataset folder
    :key promts_folder: promts folder (with .json file)
    :key database_filename: SQLite3 database filename
    :key promts_filename: promts filename
    :key print_links: print chapter links
    :key verbose:
    :key return_links: if False, return chapters data as list of tuples. If True return YouTube links with start timestamp
    """
    verbose = kwargs.get('verbose', True)
    print_links = kwargs.get('print_links', False)
    return_links = kwargs.get('return_links', True)
    database_folder = kwargs.get('dataset_folder')
    promts_folder = kwargs.get('promts_folder')
    database_filename = kwargs.get('database_filename', 'youtube_rep_fit_database.db')
    promts_filename = kwargs.get('promts_filename', 'squats_non_compound.json')

    database_filepath = os.path.join(database_folder, database_filename)
    promts_input_filepath = os.path.join(promts_folder, promts_filename)

    chapters_links = chapters_data_via_promts(database_filepath, promts_input_filepath, return_links=return_links, verbose=verbose)

    if print_links:
        pprint.pprint(chapters_links, width=120)


if __name__ == '__main__':

    parameters = {
        'dataset_folder': '/home/anton/work/fitMate/datasets',
        'promts_folder': '/home/anton/work/fitMate/datasets/exercises_filter_promts',
        'database_filename': 'youtube_rep_fit_database.db',
        'promts_filename': 'squats_non_compound.json',
        'print_links': True,
        'verbose': True,
        'return_links': False,
    }

    links_squats_non_compound(**parameters)
