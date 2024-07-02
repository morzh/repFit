import os
import sqlite3
import pandas as pd
from loguru import logger

from utils.youtube_database.add_data_to_database_tables import add_channel_data, add_channel_video_data, add_video_chapters_data, define_database_tables


print(f'{sqlite3.sqlite_version=}')

root_filepath = '/home/anton/work/fitMate/datasets'
excel_links_folder = 'youtube_channels_links'
database_filename = 'youtube_rep_fit_database.db'

excel_links_path = os.path.join(root_filepath, excel_links_folder)
database_filepath = os.path.join(root_filepath, database_filename)
connection = sqlite3.connect(database_filepath)
cursor = connection.cursor()
excel_filenames = [f for f in os.listdir(excel_links_path) if os.path.isfile(os.path.join(excel_links_path, f)) and f.endswith('xlsx')]

logger.add('fetch_youtube_database.log', format="{time} {level} {message}", level="DEBUG", retention="10 days", compression="zip")
define_database_tables(connection)

for excel_file_index, excel_filename in enumerate(excel_filenames):
    excel_basename, _ = os.path.splitext(excel_filename)
    current_channel_id = '@' + excel_basename
    cursor.execute("""SELECT id FROM YoutubeChannel""")
    channels_ids = cursor.fetchall()
    if (current_channel_id,) not in channels_ids:
        add_channel_data(current_channel_id, connection)

    excel_filepath = os.path.join(excel_links_path, excel_filename)
    try:
        current_excel_data = pd.read_excel(excel_filepath)
    except OSError:
        logger.debug(f'Read of {excel_filepath} failed.')
        continue

    for video_url_index, video_url in enumerate(current_excel_data.values[:, 1]):
        cursor.execute("""SELECT id FROM YoutubeVideo""")
        database_video_ids = cursor.fetchall()
        current_video_id = video_url.split('&t=')[0][-11:]
        if (current_video_id,) not in database_video_ids:
            current_chapters = add_channel_video_data(current_video_id, current_channel_id, connection)
            add_video_chapters_data(current_chapters, current_video_id, connection)

connection.close()
