import json
import os
import sqlite3
import pandas as pd
from deep_translator import GoogleTranslator

from utils.youtube.fetch_information import (fetch_youtube_channel_information,
                                             fetch_youtube_video_information,
                                             delete_redundant_video_keys)


def add_channel_data(current_channel_id: str, connection) -> None:
    cursor = connection.cursor()
    youtube_channel_url = 'https://www.youtube.com/' + current_channel_id

    channel_information = fetch_youtube_channel_information(youtube_channel_url, print_info=False)

    current_youtube_id = channel_information['id']
    current_channel_name = channel_information['channel']
    del channel_information['id']
    del channel_information['channel_id']
    del channel_information['channel']
    # channel_information['excel_filename'] = excel_filename

    channel_information['description'] = GoogleTranslator(source='auto', target='en').translate(channel_information['description'])
    channel_information['tags'] = GoogleTranslator(source='auto', target='en').translate_batch(channel_information['tags'])
    current_information_json = json.dumps(channel_information)
    cursor.execute("""INSERT INTO YoutubeChannel (id, name, info) VALUES (?, ?, ?)""",
                   (current_youtube_id, current_channel_name, current_information_json)
                   )
    connection.commit()


def add_channel_video_data(video_id: str, channel_id: str, connection: any) -> dict | None:
    video_url = f'https://www.youtube.com/watch?v={video_id}'
    video_information = fetch_youtube_video_information(video_url, print_info=False)
    delete_redundant_video_keys(video_information)
    video_title = GoogleTranslator(source='auto', target='en').translate(video_information['title'])
    video_duration = video_information['duration']
    video_chapters = video_information['chapters']
    del video_information['title']
    del video_information['duration']
    del video_information['chapters']
    information_json = json.dumps(video_information)
    cursor = connection.cursor()
    cursor.execute("""INSERT INTO YoutubeVideo (id, title, duration, info, channel_id_fk) VALUES (?, ?, ?, ?, ?)""",
                   (video_id, video_title, video_duration, information_json, channel_id)
                   )
    connection.commit()
    return video_chapters


def add_video_chapters_data(chapters: dict, video_id: str, connection: any) -> None:
    if chapters is None:
        return
    cursor = connection.cursor()
    for chapter in chapters:
        cursor.execute("""INSERT INTO VideosChapter (title, start_time, end_time, is_youtube_chapter, video_id_fk) VALUES (?, ?, ?, ?, ?)""",
                       (chapter['title'], chapter['start_time'], chapter['end_time'], 0, video_id)
                       )
    connection.commit()