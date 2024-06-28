from utils.youtube.youtube_information import extract_youtube_channel_information


# youtube_channel_url = 'https://www.youtube.com/@eylemabaci'
youtube_channel_url = 'https://www.youtube.com/@drakestephens3447'

current_info = extract_youtube_channel_information(youtube_channel_url, print_info=True)

print(len(current_info['tags']))
