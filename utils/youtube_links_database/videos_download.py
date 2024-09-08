import os.path
import yt_dlp
import pprint
from retry import retry

from utils.youtube_links_database.database_promts import videos_data_from_database_promts


def download_youtube_videos(database_filepath, promts_filepath, output_folder, config) -> None:
    """
    Description:
        Downloads videos from YouTube using filepath to SQLIte3 YouTube links database and filepath to .JSON with database promts.

    :param database_filepath: SQLite3 links database filepath
    :param promts_filepath: File path to .json file with include and exclude tokens for database requests
    :param output_folder: folder for downloaded videos
    :param config: configuration dictionary
    """
    # pprint.pprint(config)
    # key_per_include_promt = config['output_options']['each_include_promt_to_separate_folder']
    videos_data = videos_data_from_database_promts(database_filepath, promts_filepath)
    video_options = config['video_options']
    debug_options = config['debug_videos_options']
    print('Overall number of chapters is', len(videos_data))

    for video_data in videos_data:
        current_video_id = video_data[0]
        current_video_link = f'https://www.youtube.com/watch?v={current_video_id}'
        current_output_filename = ''.join([video_data[0], '.', video_options['video_format']])
        current_output_filepath = os.path.join(output_folder, current_output_filename)

        try:
            download_single_video_from_youtube(current_video_id, current_output_filepath, **kwargs)
            if debug_options['print_chapters_links']:
                print(f'{current_video_link} -> {current_output_filename}')

            if debug_options.get('write_chapters_links', False) and debug_options['videos_links_filepath'] is not None:
                with open(debug_options['videos_links_filepath'], 'a') as f:
                    f.write(f'{current_video_link} -> {current_output_filename}\n')
        except Exception as e:
            print(e)
