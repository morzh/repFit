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
def main(excel_files_path: str, database_folder: str, database_file: str) -> None:
    """
    Description:
        Adding information to SQLite3 database from Excel files. Script was designed in such a way to continue
        processing data regardless of any kinds of errors. All errors are logged.
        Naming convention: YouTube channel's name is the Excel filename (without '.xlsx' extension) with '@' prefix added.
    :param excel_files_path: Excel files folder path
    :param database_folder: database file folder
    :param database_file: output database filename
    """

    database_filepath = os.path.join(database_folder, database_file)
    connection = sqlite3.connect(database_filepath)
    excel_filepaths = Path(excel_files_path).glob("*.xlsx")

    define_database_tables(connection)
    with connection.cursor() as cursor:
        cursor.execute(f"""SELECT id FROM YoutubeChannel""")
        existing_channels_ids = cursor.fetchall()

    for excel_file_index, excel_filepath in enumerate(excel_filepaths):
        excel_filename = os.path.basename(excel_filepath)
        logger.info(f'Excel filename: {excel_filename}')
        excel_file_basename, _ = os.path.splitext(excel_filename)
        current_channel_id = '@' + excel_file_basename
        if (current_channel_id,) not in existing_channels_ids:
            add_channel_data(current_channel_id, connection)

        try:
            current_excel_data = pd.read_excel(excel_filepath)
        except OSError:
            logger.debug(f'Read of {excel_filepath} failed.')
            continue

        for video_url_index, video_url in enumerate(current_excel_data.values[:, 1]):
            current_video_id = video_url.split('&t=')[0][-11:]
            cursor.execute(f"""SELECT id FROM YoutubeVideo WHERE id={current_video_id}""")
            database_existing_video_ids = cursor.fetchone()
            if database_existing_video_ids is None:
                current_chapters = add_channel_video_data(current_video_id, current_channel_id, connection)
                add_video_chapters_data(current_chapters, current_video_id, connection)

    connection.close()


if __name__ == '__main__':
    dataset_filepath = '/home/anton/work/fitMate/datasets/youtube_channels_links'
    output_folder = 'youtube_channels_links'
    data_filename = 'youtube_rep_fit_database.db'

    logger.add('fetch_youtube_database.log', format="{time} {level} {message}", level="DEBUG", retention="11 days", compression="zip")
    logger.info(f'{sqlite3.sqlite_version=}')

    main(dataset_filepath, output_folder, data_filename)
