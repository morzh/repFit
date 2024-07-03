import os
import pandas as pd


def run():
    root_filepath = '/home/anton/work/fitMate/datasets/youtube_channels_links'
    excel_filenames = [f for f in os.listdir(root_filepath) if os.path.isfile(os.path.join(root_filepath, f)) and f.endswith('.xlsx')]

    links_number = 0
    excel_files_number = 0
    for excel_filename in excel_filenames:
        print(excel_filename)
        excel_filepath = os.path.join(root_filepath, excel_filename)
        try:
            current_data = pd.read_excel(excel_filepath)
        except OSError:
            print(excel_filename)
            continue

        current_links_number = len(current_data)
        links_number += current_links_number
        excel_files_number += 1

    print('\n')
    print('STATISTICS:')
    print('-' * 35)
    print(f'YouTube links to videos number: {links_number}')
    print(f'YouTube channels number: {excel_files_number}')
    print(f'Average videos per channel: {links_number / excel_files_number}')


if __name__ == '__main__':
    run()
