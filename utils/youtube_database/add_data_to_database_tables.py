import json
import sqlite3
import yt_dlp
from deep_translator import GoogleTranslator, YandexTranslator
import deep_translator.exceptions

from utils.youtube_database.fetch_information import (fetch_youtube_channel_information,
                                                      fetch_youtube_video_information,
                                                      delete_redundant_video_keys)


def add_channel_data(channel_id: str, connection: sqlite3.Connection) -> None:
    """
    Given YouTube channel_id fetch information about this channel and store it in YoutubeChannel database table
    :param channel_id: YouTube channel id
    :param connection: connection to SQLite3 database
    """
    cursor = connection.cursor()
    youtube_channel_url = 'https://www.youtube.com/' + channel_id

    channel_information = fetch_youtube_channel_information(youtube_channel_url, verbose=False)
    current_channel_name = channel_information['channel']
    del channel_information['id']
    del channel_information['channel_id']
    del channel_information['channel']

    if len(channel_information['description']):
        try:
            channel_information['description'] = GoogleTranslator(source='auto', target='en').translate(channel_information['description'])
        except deep_translator.exceptions.RequestError as error:
            print(f'Translating channel description to English error::{error.message}')

    if len(channel_information['tags']):
        try:
            channel_information['tags'] = GoogleTranslator(source='auto', target='en').translate_batch(channel_information['tags'])
        except deep_translator.exceptions.RequestError as error:
            print(f'Translating channel tags to English error::{error.message}')

    current_information_json = json.dumps(channel_information)
    cursor.execute("""INSERT INTO YoutubeChannel (id, name, info) VALUES (?, ?, ?)""",
                   (channel_id, current_channel_name, current_information_json)
                   )
    connection.commit()


def add_channel_video_data(video_id: str, channel_id: str, connection: sqlite3.Connection) -> list | None:
    """
    Description:
        Given YouTube video_id and channel_id fetch video information and add it to YoutubeVideo database table.
        Returned chapters will be used later to add them to the VideosChapter table.
    :param video_id: YouTube video id
    :param channel_id: YouTube channel id
    :param connection: connection to SQLite3 database
    :return: list of chapters of the YouTube video or None
    """
    video_url = f'https://www.youtube.com/watch?v={video_id}'
    try:
        video_information = fetch_youtube_video_information(video_url, verbose=False)
    except yt_dlp.utils.UnsupportedError as error:
        print(error.msg)
        return
    except yt_dlp.utils.DownloadError as error:
        print(error.msg)
        return

    delete_redundant_video_keys(video_information)

    video_title = video_information['title']
    video_duration = video_information['duration']
    video_chapters = video_information['chapters']

    del video_information['id']
    del video_information['title']
    del video_information['duration']
    del video_information['chapters']

    try:
        video_title = GoogleTranslator(source='auto', target='en').translate(video_title)
    except deep_translator.exceptions.RequestError as error:
        print(f'Translating video title to English error::{error.message}')

    if len(video_information['description']):
        try:
            video_information['description'] = GoogleTranslator(source='auto', target='en').translate(video_information['description'])
        except deep_translator.exceptions.RequestError as error:
            print(f'Translating video description to English error::{error.message}')

    if len(video_information['categories']):
        try:
            video_information['categories'] = GoogleTranslator(source='auto', target='en').translate_batch(video_information['categories'])
        except deep_translator.exceptions.RequestError as error:
            print(f'Translating video categories to English error::{error.message}')

    information_json = json.dumps(video_information)
    cursor = connection.cursor()
    cursor.execute("""INSERT INTO YoutubeVideo (id, title, duration, info, channel_id_fk) VALUES (?, ?, ?, ?, ?)""",
                   (video_id, video_title, video_duration, information_json, channel_id)
                   )
    connection.commit()
    return video_chapters


def add_video_chapters_data(chapters: list[dict] | None, video_id: str, connection: sqlite3.Connection) -> None:
    """
    Description:
        Given YouTube video chapters list (or None), add chapters to the VideosChapter database table.
    Remarks:
        VideosChapter.source is always 'youtube' for fetched video information.
        It will be used later to signal, from which source video segments annotation came.
    :param chapters: list of dicts with video chapters information
    :param video_id: YouTube video id
    :param connection: connection to SQLite3 database
    """
    if chapters is None:
        return
    cursor = connection.cursor()
    for chapter in chapters:
        try:
            chapter['title'] = GoogleTranslator(source='auto', target='en').translate(chapter['title'])
        except deep_translator.exceptions.RequestError as error:
            print(f'Translating chapter title ot English error::{error.message}')
        cursor.execute("""INSERT INTO VideosChapter (title, start_time, end_time, source, video_id_fk) VALUES (?, ?, ?, ?, ?)""",
                       (chapter['title'], float(chapter['start_time']), float(chapter['end_time']), 'youtube', video_id)
                       )
    connection.commit()
