from utils.youtube.fetch_information import fetch_youtube_channel_information


# youtube_channel_url = 'https://www.youtube.com/@eylemabaci'
youtube_channel_url = 'https://www.youtube.com/@drakestephens3447'

current_info = fetch_youtube_channel_information(youtube_channel_url, verbose=True)

print(len(current_info['tags']))
