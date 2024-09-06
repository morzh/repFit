import os.path
import yt_dlp
import pprint
from retry import retry


def download_youtube_videos(database_file_path, promts_file_path, output_videos_folder, **parameters):
    pprint.pprint(parameters)
