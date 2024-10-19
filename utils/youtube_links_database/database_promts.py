from dataclasses import dataclass
import pprint
import sqlite3
import json
import numpy as np
from libpasteurize.fixes.fix_imports import all_patterns
from shapely.ops import orient


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

def database_logical_element(command: str, tokens: list, operation: str, wildcard : str = '') -> str | None:
    """
    Description:

    :param command:
    :param tokens:
    :param operation:
    :param wildcard:

    :return: '(token1 OP token2 OP ... OP token_n)' like string if tokens is not empty, empty string otherwise
    """

    if len(tokens):
        patterns = [f"{command} '{wildcard}{token}{wildcard}' {operation}" for token in tokens if len(token)]
        command_pattern = " ".join(patterns)
        operation_length = len(operation) + 1
        command_pattern = command_pattern[:-operation_length]
        command_pattern = "".join(["(", command_pattern, ")"])
        return command_pattern

    return ""

def select_videos_from_database(database_filepath: str, promts: dict[str, tuple]) -> list[tuple]:
    """
    Description:
        Select videos from SQLite3 database using (... AND ...) AND (... OR ...) AND NOT (... OR ... ) promt  for videos title

    """
    #  (... AND ... AND ...) AND ( ... OR ... OR ...) AND NOT (... OR ... OR ...)
    and_tokens = list(promts['and_tokens'])
    or_tokens = list(promts['or_tokens'])
    exclude_tokens = list(promts['and_not_tokens'])
    video_minimum_duration = promts['video_options']['minimum_duration']
    video_maximum_duration = promts['video_options']['maximum_duration']

    execute_command = f"SELECT * FROM YoutubeVideo WHERE"
    and_pattern = database_logical_element('title LIKE', and_tokens, 'AND', wildcard='%')
    or_pattern = database_logical_element('title LIKE', or_tokens, 'OR', wildcard='%')
    not_pattern = database_logical_element('title LIKE', exclude_tokens, 'OR', wildcard='%')

    if len(and_pattern): execute_command = " ".join([execute_command, and_pattern])
    if len(or_pattern): execute_command = " ".join(['AND', execute_command, or_pattern])
    if len(not_pattern): execute_command = " ".join([execute_command, 'AND NOT', not_pattern])

    if video_minimum_duration > 0: execute_command = " ".join([execute_command, f' AND duration > {video_minimum_duration}'])
    if video_maximum_duration > 0: execute_command = " ".join([execute_command, f' AND duration < {video_maximum_duration}'])

    # print(execute_command)

    connection = sqlite3.connect(database_filepath)
    cursor = connection.cursor()
    cursor.execute(execute_command)
    selected_videos = cursor.fetchall()
    return selected_videos



def select_chapters_from_database(database_filepath: str, or_tokens: list[str], exclude_tokens: list[str]) -> list[tuple]:
    """
    Description:
        Select chapters from SQLite3 database using (... OR ...) AND NOT (... OR ... ) promt for chapters title

    :param database_filepath: filepath to SQLite2 database
    :param or_tokens: (... OR ...) tokens in database SELECT promt
    :param exclude_tokens: AND NOT (... OR ...) tokens in database SELECT promt

    :return: list with chapters data
    """
    connection = sqlite3.connect(database_filepath)

    execute_command = f"""SELECT * FROM VideosChapter WHERE ("""

    or_pattern = [f"""title LIKE '%{entry}%' OR""" for entry in or_tokens]
    execute_command = execute_command + ' '.join(or_pattern)
    execute_command = execute_command[:-3]
    execute_command += ')'

    if len(exclude_tokens):
        execute_command += ' AND NOT ('
        exclude_patterns = [f"""title LIKE '%{entry}%' OR """ for entry in exclude_tokens]
        execute_command = execute_command + ' '.join(exclude_patterns)
        execute_command = execute_command[:-4]
        execute_command += ');'
    else:
        execute_command += ';'

    cursor = connection.cursor()
    cursor.execute(execute_command)
    selected_chapters = cursor.fetchall()

    return selected_chapters


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


def videos_data_from_database_promts(database_filepath: str, promts_filepath: str) -> list[tuple]:
    """
    Description:
        Get videos data from database using promts (promts represented by JSON file, database bby SQLITE3 .db file).

    :param database_filepath: SQLite3 database filepath
    :param promts_filepath: promts dictionary JSON file

    :return: dictionary with keys representing each promt from include_promts and value is a list of chapters data
    """
    with open(promts_filepath) as f:
        promt_tokens = json.load(f)

    selected_videos_data = select_videos_from_database(database_filepath, promt_tokens)
    return selected_videos_data


def chapters_data_from_database_promts(database_filepath: str, promts_filepath: str) -> dict[str, list]:
    """
    Description:
        Get chapters data from database using promts (promts represented by JSON file, database bby SQLITE3 .db file).

    :param database_filepath: SQLite3 database filepath
    :param promts_filepath: promts dictionary JSON file

    :return: dictionary with keys representing each promt from include_promts and value is a list of chapters data
    """
    with open(promts_filepath) as f:
        promt_tokens = json.load(f)

    exercises_types = promt_tokens['or_tokens']
    exclude_tokens = promt_tokens['and_not_tokens']

    chapters_data = {}
    for exercise in exercises_types:
        current_chapters = select_chapters_from_database(database_filepath, [exercise], exclude_tokens)
        chapters_data[exercise] = current_chapters

    return chapters_data



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

    chapters_promts = chapters_data_from_database_promts(database_filepath, promts_filepath)
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