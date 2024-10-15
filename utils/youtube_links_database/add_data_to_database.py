import json
import sqlite3
import yt_dlp
from deep_translator import GoogleTranslator
import deep_translator.exceptions
from loguru import logger
from retry import retry
import requests, http.client
from sympy.integrals.meijerint_doc import category
from torch import NoneType

from utils.youtube_links_database.fetch_information_from_youtube import (
    fetch_youtube_channel_information,
    fetch_youtube_video_information,
    delete_keys_from_dictionary
)


def define_database_tables(connection: sqlite3.Connection) -> None:
    """
    Description:
        Tables for YouTube database. There are three tables, first for channels, second for videos and third for video chapters.

    :param connection: connection to SQLite3 database
    """
    cursor = connection.cursor()
    cursor.execute('''PRAGMA foreign_keys = ON;''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS YoutubeChannel (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            info TEXT
            )
        ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS YoutubeVideo (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            duration INTEGER NOT NULL,
            info TEXT,
            channel_id_fk TEXT,
            FOREIGN KEY (channel_id_fk) REFERENCES YoutubeChannel(id) ON DELETE CASCADE
            )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS VideosChapter (
            chapter_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            start_time REAL,
            end_time REAL,
            source TEXT,
            video_id_fk TEXT,
            FOREIGN KEY (video_id_fk) REFERENCES YoutubeVideo(id) ON DELETE CASCADE
            )
    ''')
    connection.commit()


def add_channel_data(channel_id: str, connection: sqlite3.Connection) -> None:
    """
    Description:
        Given YouTube channel_id fetch information about this channel and store it in YoutubeChannel database table.

    :param channel_id: YouTube channel id
    :param connection: connection to SQLite3 database
    """
    youtube_channel_url = f'https://www.youtube.com/{channel_id}'

    try:
        channel_information = fetch_youtube_channel_information(youtube_channel_url, verbose=False)
    except (yt_dlp.utils.UnsupportedError, yt_dlp.utils.DownloadError) as error:
        logger.error(error.msg)
        return

    current_channel_name = channel_information['channel']
    channel_information_delete_keys_list = ['id', 'channel_id', 'channel']
    delete_keys_from_dictionary(channel_information, channel_information_delete_keys_list)

    if len(channel_information['description']):
        try:
            channel_information['description'] = GoogleTranslator(source='auto', target='en').translate(channel_information['description'])
        except deep_translator.exceptions.RequestError as error:
            logger.info(f'Translating channel {channel_id} description to English error::{error.message}')
        except deep_translator.exceptions.NotValidLength:
            logger.info(f'Channel {channel_id} description is too long.')

    if len(channel_information['tags']):
        try:
            channel_information['tags'] = GoogleTranslator(source='auto', target='en').translate_batch(channel_information['tags'])
        except deep_translator.exceptions.RequestError as error:
            logger.info(f'Translating channel {channel_id} tags to English error::{error.message}')
        except requests.exceptions.ConnectionError as error:
            logger.info(f'Translating channel {channel_id} tags to English error::{error.errno}')

    current_information_json = json.dumps(channel_information)
    try:
        cursor = connection.cursor()
        cursor.execute("""INSERT INTO YoutubeChannel (id, name, info) VALUES (?, ?, ?)""",
                       (channel_id, current_channel_name, current_information_json)
                       )
        connection.commit()
    except sqlite3.Error as error:
        logger.error(error.sqlite_errorname)


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
    except (yt_dlp.utils.UnsupportedError, yt_dlp.utils.DownloadError, yt_dlp.networking.exceptions.TransportError) as error:
        logger.error(error.msg)
        return

    video_title = video_information.get('title', '')
    video_duration = video_information.get('duration', -1)
    video_chapters = video_information.get('chapters', [])

    video_information_delete_keys_list = ['id', 'title', 'duration', 'chapters']
    delete_keys_from_dictionary(video_information, video_information_delete_keys_list)

    try:
        video_title = GoogleTranslator(source='auto', target='en').translate(video_title)
    except deep_translator.exceptions.RequestError as error:
        logger.info(f'Translating {video_id} video title to English error::{error.message}')
    except requests.exceptions.ConnectionError as error:
        logger.info(f'Translating {video_id} video title to English error::{error.errno}')

    if len(video_information.get('description', '')):
        try:
            video_information['description'] = GoogleTranslator(source='auto', target='en', proxies={"socks5://": "127.0.0.1"}).translate(video_information['description'])
        except deep_translator.exceptions.RequestError as error:
            logger.info(f'Translating video {video_id} description to English request error::{error.message}')
        except deep_translator.exceptions.NotValidLength:
            logger.info(f'{video_id} video description is too long.')
        except requests.exceptions.ConnectionError as error:
            logger.info(f'{video_id} video connection error {error.errno}.')

    categories = video_information.get('categories')
    if categories is not None and len(categories):
        try:
            video_information['categories'] = GoogleTranslator(source='auto', target='en').translate_batch(video_information['categories'])
        except deep_translator.exceptions.RequestError as error:
            logger.info(f'Translating {video_id} video categories to English request error::{error.message}')
        except deep_translator.exceptions.TranslationNotFound as error:
            logger.info(f'Translating {video_id} video error: {error.message}')
        except requests.exceptions.ConnectionError as error:
            logger.info(f'Translating {video_id} video error: {error.errno}')

    information_json = json.dumps(video_information)

    try:
        cursor = connection.cursor()
        cursor.execute("""INSERT INTO YoutubeVideo (id, title, duration, info, channel_id_fk) VALUES (?, ?, ?, ?, ?)""",
                       (video_id, video_title, video_duration, information_json, channel_id)
                       )
        connection.commit()
    except sqlite3.Error as error:
        if error.sqlite_errorname == 'SQLITE_CONSTRAINT_FOREIGNKEY':
            logger.debug(f'Error inserting foreign key {channel_id=} in table YoutubeVideo with {video_id=}')
        logger.error(error.sqlite_errorname)
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

    for chapter in chapters:
        try:
            chapter['title'] = GoogleTranslator(source='auto', target='en').translate(chapter['title'])
        except deep_translator.exceptions.RequestError as error:
            logger.info(f'Translating chapter title to English request error::{error.message}')
        except http.client.RemoteDisconnected:
            logger.info(f'Translating chapter title to English remote disconnected.')
        except requests.exceptions.ConnectionError:
            logger.info(f'Translating chapter title to English connection aborted.')

        try:
            cursor = connection.cursor()
            cursor.execute("""INSERT INTO VideosChapter (title, start_time, end_time, source, video_id_fk) VALUES (?, ?, ?, ?, ?)""",
                           (chapter['title'], float(chapter['start_time']), float(chapter['end_time']), 'youtube', video_id)
                           )
            connection.commit()
        except sqlite3.Error as error:
            logger.error(error.sqlite_errorname)
