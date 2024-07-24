import os
import pprint
import sqlite3
import pandas as pd
from pathlib import Path
from loguru import logger


def filter_chapters(connection: sqlite3.Connection, like_entries: list[str], not_like_entries: list[str]) -> list[tuple]:
    """
    TODO: make request non case sensitive
    """
    execute_command = f"""SELECT * FROM VideosChapter WHERE ("""

    like_patterns = [f"""title LIKE '%{entry}%' OR""" for entry in like_entries]
    execute_command = execute_command + ' '.join(like_patterns)
    execute_command = execute_command[:-3]

    if len(not_like_entries):
        execute_command += ') INTERSECT SELECT * FROM VideosChapter WHERE ('
        not_like_patterns = [f"""title NOT LIKE '%{entry}%' OR """ for entry in not_like_entries]
        execute_command = execute_command + ' '.join(not_like_patterns)
        execute_command = execute_command[:-4]
        execute_command += ');'
    else:
        execute_command += ');'

    cursor = connection.cursor()
    cursor.execute(execute_command)
    chapters_with_entry = cursor.fetchall()

    return chapters_with_entry


def convert_chapter_data_to_link(chapters_data: list[tuple]) -> list[str]:
    youtube_link_base = 'https://www.youtube.com/watch?v='
    youtube_links = [f"{youtube_link_base}{data[5]}&t={int(data[2])}" for data in chapters_data]
    return youtube_links


def filter_videos():
    pass


def filter_channels():
    pass


def main():
    database_folder = '/home/anton/work/fitMate/datasets'
    database_filename = 'youtube_rep_fit_database.db'
    database_filepath = os.path.join(database_folder, database_filename)
    connection = sqlite3.connect(database_filepath)

    chapter_like_pattern = ['goblet squat', 'air squat', 'side squat', 'pistol squat', 'prisoner squat']
    chapter_not_like_patterns = ['why', 'answer', 'with', 'thruster', 'pulse', 'to', 'jump']

    chapters = filter_chapters(connection, chapter_like_pattern, chapter_not_like_patterns)
    chapters_links = convert_chapter_data_to_link(chapters)

    print(f'Chapters found: {len(chapters)}')
    # pprint.pprint(chapters[:550], width=120)
    pprint.pprint(chapters_links, width=120)


if __name__ == '__main__':
    main()
