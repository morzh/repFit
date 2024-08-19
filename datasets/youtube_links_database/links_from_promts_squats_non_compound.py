import os
import pprint
from utils.youtube_links_database.database_promts import chapters_links_via_promt


def links_squats_non_compound(**kwargs):
    verbose = kwargs.get('verbose', True)
    print_links = kwargs.get('print_links', False)
    return_links = kwargs.get('return_links', True)
    database_folder = kwargs.get('dataset_folder')
    promts_folder = kwargs.get('promts_folder')
    database_filename = kwargs.get('database_filename', 'youtube_rep_fit_database.db')
    promts_filename = kwargs.get('promts_filename', 'squats_non_compound.json')

    database_filepath = os.path.join(database_folder, database_filename)
    promts_input_filepath = os.path.join(promts_folder, promts_filename)

    chapters_links = chapters_links_via_promt(database_filepath, promts_input_filepath, return_links=return_links, verbose=verbose)

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
