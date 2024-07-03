import os
import sqlite3
import pandas as pd
from pathlib import Path
from loguru import logger

from utils.youtube_database.add_data_to_database_tables import (
    add_channel_data,
    add_channel_video_data,
    add_video_chapters_data,
    define_database_tables
)


@logger.catch
def main(root_folder: str, excel_files_folder: str, database_file: str) -> None:
    """
    Description:
        Adding information to SQLite3 database from Excel files. Script was designed in such a way to continue
        processing data regardless of any kinds of errors. All errors are logged.
        Naming convention: YouTube channel's name is the Excel filename (without '.xlsx' extension) with '@' prefix added.
    """

    excel_links_path = os.path.join(root_folder, excel_files_folder)
    database_filepath = os.path.join(root_folder, database_file)
    connection = sqlite3.connect(database_filepath)
    excel_filepaths = Path(excel_links_path).glob("*.xlsx")
    cursor = connection.cursor()

    define_database_tables(connection)

    for excel_file_index, excel_filepath in enumerate(excel_filepaths):
        excel_filename = os.path.basename(excel_filepath)
        logger.info(f'Excel filename: {excel_filename}')
        excel_basename, _ = os.path.splitext(excel_filename)
        current_channel_id = '@' + excel_basename
        cursor.execute("""SELECT id FROM YoutubeChannel""")
        channels_ids = cursor.fetchall()
        if (current_channel_id,) not in channels_ids:
            add_channel_data(current_channel_id, connection)

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


root_dataset_filepath = '/home/anton/work/fitMate/datasets'
excel_files_youtube_links_folder = 'youtube_channels_links'
database_filename = 'youtube_rep_fit_database.db'

logger.add('fetch_youtube_database.log', format="{time} {level} {message}", level="DEBUG", retention="11 days", compression="zip")
logger.info(f'{sqlite3.sqlite_version=}')

main(root_dataset_filepath, excel_files_youtube_links_folder, database_filename)
