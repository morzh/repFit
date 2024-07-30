import os
import sqlite3
from loguru import logger


def chapters_selection(database_folder: str, database_file: str, occurrence: str) -> None:
    database_filepath = os.path.join(database_folder, database_file)
    connection = sqlite3.connect(database_filepath)
    cursor = connection.cursor()
    cursor.execute(f"""SELECT * FROM VideosChapter WHERE title LIKE '%{occurrence}%' """)
    chapters_titles = cursor.fetchall()
    print(len(chapters_titles))


if __name__ == '__main__':
    
    output_folder = '/home/anton/work/fitMate/datasets'
    data_filename = 'youtube_rep_fit_database.db'
    chapter_occurrence_word = 'squat'

    logger.add('fetch_youtube_database.log', format="{time} {level} {message}", level="DEBUG", retention="1 days", compression="zip")
    logger.info(f'{sqlite3.sqlite_version=}')

    chapters_selection(output_folder, data_filename, chapter_occurrence_word)
