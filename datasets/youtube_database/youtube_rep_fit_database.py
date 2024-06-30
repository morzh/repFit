import os
from loguru import logger
import sqlite3
import json
import pandas as pd
from deep_translator import GoogleTranslator

from utils.youtube.add_data_to_database_tables import add_channel_data, add_channel_video_data, add_video_chapters_data
from utils.youtube.fetch_information import fetch_youtube_channel_information

print(f'{sqlite3.sqlite_version=}')

root_filepath = '/home/anton/work/fitMate/datasets'

excel_links_folder = 'youtube_channels_links'
excel_links_path = os.path.join(root_filepath, excel_links_folder)

database_filename = 'youtube_rep_fit_database.db'
database_filepath = os.path.join(root_filepath, database_filename)

connection = sqlite3.connect(database_filepath)
cursor = connection.cursor()

"""
Tables for YouTube database. There are three tables, one of them for channels, one for videos and the last for video chapters.
"""
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
        start_time INTEGER,
        end_time INTEGER,
        is_youtube_chapter INTEGER,
        video_id TEXT,
        FOREIGN KEY (video_id) REFERENCES YoutubeVideo(id) ON DELETE CASCADE
        )
''')
connection.commit()

excel_filenames = [f for f in os.listdir(excel_links_path) if os.path.isfile(os.path.join(excel_links_path, f)) and f.endswith('xlsx')]
for excel_file_index, excel_filename in enumerate(excel_filenames):
    excel_basename, _ = os.path.splitext(excel_filename)
    current_channel_id = '@' + excel_basename
    cursor.execute("""SELECT id FROM YoutubeChannel""")
    channels_ids = cursor.fetchall()
    if current_channel_id not in channels_ids:
        add_channel_data(current_channel_id, connection)

    excel_filepath = os.path.join(excel_links_path, excel_filename)
    try:
        current_excel_data = pd.read_excel(excel_filepath)
    except OSError:
        print(f'Can\'t read {excel_filename} file')
        continue

    for video_url_index, video_url in enumerate(current_excel_data.values[:, 1]):
        cursor.execute("""SELECT id FROM YoutubeVideo""")
        database_video_ids = cursor.fetchall()
        current_video_id = video_url.split('&t=')[0][-11:]
        if current_video_id not in database_video_ids:
            current_chapters = add_channel_video_data(current_video_id, current_channel_id, connection)
            add_video_chapters_data(current_chapters, current_video_id, connection)

connection.close()


