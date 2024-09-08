import os.path
import yt_dlp
import pprint
from retry import retry

from utils.youtube_links_database.database_promts import videos_data_via_promts


def download_youtube_videos(database_filepath, promts_filepath, output_folder, **kwargs):
    """
    Description:
        Downloads videos from YouTube using filepath to SQLIte3 YouTube links database and filepath to .JSON with database promts.

    :param database_filepath: SQLite3 links database filepath
    :param promts_filepath: File path to .json file with include and exclude tokens for database requests
    :param output_folder: folder for downloaded videos
    """
    pprint.pprint(kwargs)

    key_per_include_promt = kwargs['output_options']['each_include_promt_to_separate_folder']
    chapters_promts = videos_data_via_promts(database_filepath, promts_filepath)
    video_format = kwargs.get('video_format','mp4')

    total_chapters_number = sum(len(l) for l in chapters_promts.values())
    print('Overall number of chapters is', total_chapters_number)