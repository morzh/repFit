import os
import pprint
import sqlite3
import json
import numpy as np


class Color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARK_CYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def filter_chapters(database_filepath: str, like_entries: list[str], not_like_entries: list[str]) -> list[tuple]:
    """
    Description:

    :param database_filepath:
    :param like_entries:
    :param not_like_entries:

    :return: list with chapters data
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


def convert_chapters_data_to_links(chapters_data: list[tuple]) -> list[str]:
    """
    Description:
        Convert chapters data to YouTube links

    :param chapters_data: input chapters data

    :return: list of YouTube links with chapters timestamps
    """
    youtube_link_base = 'https://www.youtube.com/watch?v='
    youtube_links = [f"{youtube_link_base}{data[5]}&t={int(data[2])}" for data in chapters_data]
    return youtube_links

# def convert_chapter_data_to_


def chapters_statistics(chapters_data: list[tuple]) -> tuple[float, float]:
    """
    Description:
        Calculate chapters statistics
        
    :param chapters_data: input chapters data

    :return: mean and standard deviation of durations of the chapters
    """
    durations = np.empty(len(chapters_data))
    for chapter_index, chapter in enumerate(chapters_data):
        current_duration = chapter[3] - chapter[2]
        durations[chapter_index] = current_duration

    durations_mean = np.mean(durations)
    durations_std = np.std(durations)

    return durations_mean, durations_std

def filter_videos():
    raise NotImplementedError


def filter_channels():
    raise NotImplementedError


def chapters_data_via_promts(database_filepath: str, promts_filepath: str) -> dict[str, list]:
    """
    Description:
        Get chapters data from promt (represented by file)

    :param database_filepath:
    :param promts_filepath:

    :return: dictionary with keys representing each promt from include_promts and value is a list of chapters data
    """
    with open(promts_filepath) as f:
        squats_tokens = json.load(f)

    exercises_types = squats_tokens['include_tokens']
    chapter_not_like_patterns = squats_tokens['exclude_tokens']

    chapters_data = {}
    for exercise in exercises_types:
        current_chapters = filter_chapters(database_filepath, [exercise], chapter_not_like_patterns)
        chapters_data[exercise] = current_chapters

    return chapters_data


def links_from_database_promts(database_filepath, promts_filepath, **kwargs) -> dict[str, list]:
    """
    Description:

    :param database_filepath:
    :param promts_filepath:

    :key print_links: print chapter links
    :key verbose:

    :return: dictionary, where each key is a squat type and value is list of links to chapters
    """
    verbose = kwargs.get('verbose', True)
    print_links = kwargs.get('print_links', False)

    chapters_promts = chapters_data_via_promts(database_filepath, promts_filepath)
    links = {}
    total_chapter_number = 0

    for chapter_folder, chapters_data in chapters_promts.items():
        print(Color.BOLD + Color.YELLOW + chapter_folder + Color.END)
        current_chapters_links = convert_chapters_data_to_links(chapters_data)
        links[chapter_folder] = current_chapters_links
        if print_links:
            pprint.pprint(current_chapters_links, indent=8, width=150)

        if verbose:
            current_number_chapters = len(chapters_data)
            total_chapter_number += current_number_chapters
            print(f'\t Number of chapters is {current_number_chapters}')

    if verbose:
        print('-' * 40)
        print(Color.BOLD + Color.GREEN +  f'Total number of chapters is {total_chapter_number}' + Color.END)

    return links