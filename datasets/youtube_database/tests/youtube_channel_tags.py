import os
import pprint
import sqlite3
from deep_translator import GoogleTranslator

from utils.youtube.youtube_information import extract_youtube_channel_information


root_filepath = '/home/anton/work/fitMate/datasets'
excel_links_folder = 'youtube_channels_links'
excel_links_path = os.path.join(root_filepath, excel_links_folder)
database_filename = 'youtube_channels_information.db'
database_filepath = os.path.join(root_filepath, database_filename)

connection = sqlite3.connect(database_filepath)
cursor = connection.cursor()

excel_filenames = [f for f in os.listdir(excel_links_path) if os.path.isfile(os.path.join(excel_links_path, f)) and f.endswith('xlsx')]
# excel_filenames.sort()

database_channel_tags = []

cursor.execute("""
    SELECT youtube_id, GROUP_CONCAT(tag,'\n') AS tags_for_this_object 
    FROM ChannelToTagsMapping 
    JOIN YoutubeChannels ON from_channel_id = YoutubeChannels.id
    JOIN ChannelTags ON to_channel_tag_id = ChannelTags.tag_id
    GROUP BY youtube_id
""")
database_tags_data = cursor.fetchall()
database_tags_data = {entry[0]: entry[1].split('\n') for entry in database_tags_data}

for excel_index, current_excel_filename in enumerate(excel_filenames):
    if excel_index == 10:
        break
    current_excel_basename, _ = os.path.splitext(current_excel_filename)
    current_youtube_channel_url = 'https://www.youtube.com/@' + current_excel_basename
    current_information = extract_youtube_channel_information(current_youtube_channel_url, print_info=False)

    current_channel_tags = current_information.get('tags', [])
    current_youtube_id = current_information.get('id', None)
    current_channel_fetched_tags = []
    for tag in current_channel_tags:
        translated_tag = GoogleTranslator(source='auto', target='en').translate(tag)
        translated_tag = translated_tag.lower()
        current_channel_fetched_tags.append(translated_tag)

    current_database_tags = database_tags_data[current_youtube_id]
    print(current_database_tags)
    print(current_channel_fetched_tags)
    print(current_database_tags == current_channel_fetched_tags)
