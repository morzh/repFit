from dataclasses import dataclass
import pprint
import sqlite3
import json
import numpy as np


@dataclass(frozen=True, slots=True)
class PrintColor:
    """
    Description:
        Class for print function options
    """
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


def filter_videos(database_filepath: str, like_entries: list[str], not_like_entries: list[str]) -> list[tuple]:
    """
    Description:

    :param database_filepath:
    :param like_entries:
    :param not_like_entries:

    :return: list with chapters data
    """
    raise NotImplemented

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
        Convert chapters data to YouTube links (with start and end times)

    :param chapters_data: input chapters data

    :return: list of YouTube links with chapters timestamps
    """
    youtube_link_base = 'https://www.youtube.com/watch?v='
    youtube_links = [f"{youtube_link_base}{data[5]}?start={int(data[2])}&end={int(data[3])}" for data in chapters_data]
    return youtube_links


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


def chapters_data_via_promts(database_filepath: str, promts_filepath: str) -> dict[str, list]:
    """
    Description:
        Get chapters data from database using promts (promts represented by JSON file, database bby SQLITE3 .db file).

    :param database_filepath: SQLite3 database filepath
    :param promts_filepath: promts dictionary JSON file

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


def videos_data_via_promts(database_filepath: str, promts_filepath: str) -> dict[str, list]:
    """
    Description:
        Get videos data from database using promts (promts represented by JSON file, database bby SQLITE3 .db file).

    :param database_filepath: SQLite3 database filepath
    :param promts_filepath: promts dictionary JSON file

    :return: dictionary with keys representing each promt from include_promts and value is a list of chapters data
    """
    with open(promts_filepath) as f:
        squats_tokens = json.load(f)

    exercises_types = squats_tokens['include_tokens']
    chapter_not_like_patterns = squats_tokens['exclude_tokens']

    videos_data = {}
    for exercise in exercises_types:
        current_videos = filter_videos(database_filepath, [exercise], chapter_not_like_patterns)
        videos_data[exercise] = current_videos

    return videos_data


def links_from_database_promts(database_filepath, promts_filepath, **kwargs) -> dict[str, list]:
    """
    Description:
        Get links to YouTube chapters using promts from SQLite3 database (promts represented by JSON file, database bby SQLITE3 .db file).

    :param database_filepath: SQLite3 database filepath
    :param promts_filepath: promts dictionary JSON file

    :key print_links: print chapter links
    :key verbose: if True print chapter number

    :return: dictionary, where each key is a squat type and value is list of links to chapters
    """
    verbose = kwargs.get('verbose', True)
    print_links = kwargs.get('print_links', False)

    chapters_promts = chapters_data_via_promts(database_filepath, promts_filepath)
    links = {}
    total_chapter_number = 0

    for chapter_folder, chapters_data in chapters_promts.items():
        print(PrintColor.BOLD + PrintColor.YELLOW + chapter_folder + PrintColor.END)
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
        print(PrintColor.BOLD + PrintColor.GREEN + f'Total number of chapters is {total_chapter_number}' + PrintColor.END)

    if print_links: pprint.pprint(links)

    return links