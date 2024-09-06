import os
from utils.youtube_links_database.database_promts import links_from_database_promts


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

    chapters_links = links_from_database_promts(database_file_path, promts_file_path, **parameters)
