import os
from utils.youtube_links_database.database_promts import chapters_links_via_promt


def links_squats_non_compound():
    database_folder = '/home/anton/work/fitMate/datasets'
    promts_folder = '/home/anton/work/fitMate/datasets/exercises_filter_promts'

    database_filename = 'youtube_rep_fit_database.db'
    promts_input_filename = 'squats_non_compound.json'

    database_filepath = os.path.join(database_folder, database_filename)
    promts_input_filepath = os.path.join(promts_folder, promts_input_filename)

    chapters_links_via_promt(database_filepath, promts_input_filepath, verbose=1)


if __name__ == '__main__':
    links_squats_non_compound()
