import os
import sqlite3
from deep_translator import GoogleTranslator

from utils.youtube.youtube_information import extract_youtube_channel_information


root_filepath = '/home/anton/work/fitMate/datasets'
excel_links_folder = 'youtube_channels_links'
excel_links_path = os.path.join(root_filepath, excel_links_folder)
database_filename = 'youtube_channels_information.db'
database_filepath = os.path.join(root_filepath, database_filename)
if os.path.exists(database_filepath):
    os.remove(database_filepath)

connection = sqlite3.connect(database_filepath)
cursor = connection.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS YoutubeChannels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        youtube_id TEXT NOT NULL,
        channel_id TEXT NOT NULL,
        channel_name TEXT NOT NULL,
        description TEXT,
        original_url TEXT,
        excel_file TEXT
        )
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS ChannelTags (
        tag_id INTEGER PRIMARY KEY,
        tag TEXT
        )
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS ChannelToTagsMapping (
        from_youtube_channel_id INTEGER,
        to_channel_tag_id INTEGER
        )
''')

connection.commit()

excel_filenames = [f for f in os.listdir(excel_links_path) if os.path.isfile(os.path.join(excel_links_path, f)) and f.endswith('xlsx')]
for excel_index, excel_filename in enumerate(excel_filenames):
    if excel_index == 10:
        break
    excel_basename, _ = os.path.splitext(excel_filename)
    youtube_channel_url = 'https://www.youtube.com/@' + excel_basename
    current_information = extract_youtube_channel_information(youtube_channel_url, print_info=False)
    current_description = current_information['description']
    translated_tag = GoogleTranslator(source='auto', target='en').translate(current_description)
    cursor.execute("""INSERT INTO 
                        YoutubeChannels (youtube_id, channel_id, channel_name, description, original_url, excel_file) 
                        VALUES (?, ?, ?, ?, ?, ?)""",
                       (current_information['id'],
                        current_information['channel_id'],
                        current_information['channel'],
                        current_description,
                        current_information['original_url'],
                        excel_filename))
    connection.commit()

    cursor.execute("""SELECT last_insert_rowid()""")
    last_youtube_channel_id = cursor.fetchone()[0]
    current_channel_tags = current_information.get('tags', [])

    for tag in current_channel_tags:
        translated_tag = GoogleTranslator(source='auto', target='en').translate(tag)
        translated_tag = translated_tag.lower().rstrip()
        # Find tag_id of a tag, if tag already exists in ChannelTags table.
        # And if it is not, add this tag to the ChannelTags table
        cursor.execute(f"""SELECT tag_id FROM ChannelTags WHERE tag = '{translated_tag}';""")
        current_tag_id = cursor.fetchone()
        if current_tag_id is None:
            cursor.execute(f"""INSERT INTO ChannelTags (tag) VALUES ('{translated_tag}')""")
            cursor.execute("""SELECT last_insert_rowid()""")
            current_tag_id = cursor.fetchone()[0]
        if isinstance(current_tag_id, tuple):
            current_tag_id = current_tag_id[0]

        cursor.execute(f"""
            INSERT INTO 
            ChannelToTagsMapping (from_youtube_channel_id, to_channel_tag_id) 
            VALUES ({last_youtube_channel_id}, {current_tag_id})
            """)
        connection.commit()

connection.close()


