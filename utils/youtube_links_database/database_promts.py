import os
import pprint
import sqlite3
import json
import numpy as np


def filter_chapters(database_filepath: str, like_entries: list[str], not_like_entries: list[str]) -> list[tuple]:
    """
    """
    connection = sqlite3.connect(database_filepath)

    execute_command = f"""SELECT * FROM VideosChapter WHERE ("""

    like_patterns = [f"""title LIKE '%{entry}%' OR""" for entry in like_entries]
    execute_command = execute_command + ' '.join(like_patterns)
    execute_command = execute_command[:-3]
    execute_command += ')'

    if len(not_like_entries):
        execute_command += ' AND NOT ('
        not_like_patterns = [f"""title LIKE '%{entry}%' OR """ for entry in not_like_entries]
        execute_command = execute_command + ' '.join(not_like_patterns)
        execute_command = execute_command[:-4]
        execute_command += ');'
    else:
        execute_command += ';'

    cursor = connection.cursor()
    cursor.execute(execute_command)
    chapters_with_entry = cursor.fetchall()

    return chapters_with_entry


def convert_chapter_data_to_link(chapters_data: list[tuple]) -> list[str]:
    """

    """
    youtube_link_base = 'https://www.youtube.com/watch?v='
    youtube_links = [f"{youtube_link_base}{data[5]}&t={int(data[2])}" for data in chapters_data]
    return youtube_links


def chapters_statistics(chapters: list[tuple]) -> None:
    """

    """
    durations = np.empty(len(chapters))
    for chapter_index, chapter in enumerate(chapters):
        current_duration = chapter[3] - chapter[2]
        durations[chapter_index] = current_duration

    durations_mean = np.mean(durations)
    durations_std = np.std(durations)
    print(f'Chapters number is {len(chapters)}. Chapters durations mean is {durations_mean:.2f} seconds and σ is {durations_std:.2f} seconds.')


def filter_videos():
    pass


def filter_channels():
    pass


def chapters_links_via_promt(database_filepath: str, promts_filepath: str, verbose=1):
    with open(promts_filepath) as f:
        squats_tokens = json.load(f)

    squats_types = squats_tokens['include_tokens']
    chapter_not_like_patterns = squats_tokens['exclude_tokens']

    chapters = filter_chapters(database_filepath, squats_types, chapter_not_like_patterns)
    chapters_links = convert_chapter_data_to_link(chapters)

    if verbose == 1:
        chapters_statistics(chapters)
    if verbose == 2:
        pprint.pprint(chapters_links, width=120)
