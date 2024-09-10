import os.path
import yt_dlp
import pprint
from retry import retry

from utils.youtube_links_database.database_promts import videos_data_from_database_promts


def check_file(directory, filename_base) -> bool:
    """
    Description:

    :param directory:
    :param filename_base:
    """
    for s in os.listdir(directory):
        if os.path.splitext(s)[0] == filename_base and os.path.isfile(os.path.join(directory, s)):
            return True

    return False


@retry(yt_dlp.utils.DownloadError, delay=2, backoff=2, max_delay=4, tries=5)
def download_single_video_from_youtube(video_id: str, output_filepath: str, use_proxy: bool, **kwargs) -> None:
    """
    Description
        Download YouTube video chapter.

    :param video_id: YouTube video ID.
    :param output_filepath: output filepath for downloaded video.
    :param use_proxy: use proxy socks5://127.0.0.1 to mitigate YouTube speed restriction in Russia

    :key video_format: add offset to video segment. New segment will be [time_start - offset, time_end + offset]
    :key video_quality: video quality, e.g. 360, 720 or 1080
    """

    video_format = kwargs.get('video_format', 'mp4')

    ydl_options = {
        'verbose': True,
        'outtmpl': output_filepath,
        'prefer_ffmpeg': True,
    }
    if video_format is not None:
        video_quality = kwargs.get('video_quality', 720)
        ydl_options['format'] =  f'bestvideo[height={video_quality}][ext={video_format}]'
    if use_proxy:
        ydl_options['proxy'] = "socks5://127.0.0.1"

    youtube_video_url = f'https://www.youtube.com/watch?v={video_id}'

    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        ydl.download(youtube_video_url)


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
    use_proxy = config['connection']['use_proxy']
    video_format = video_options['video_format']
    print('Overall number of videos is', len(videos_data))

    for video_data in videos_data:
        current_video_id = video_data[0]
        current_video_link = f'https://www.youtube.com/watch?v={current_video_id}'
        current_output_filename = ''.join([video_data[0], '.', video_format]) if video_format else video_data[0]
        current_output_filepath = os.path.join(output_folder, current_output_filename)

        if check_file(output_folder, video_data[0]):
            """
            In our case, base names of video files are just YouTube ids. But format or video file (or extension) could be different.
            So we need to check only if filename base is exist (in other words ignore extension of the video file).
            """
            continue

        try:
            # if os.path.exists(current_output_filepath):
                # continue

            download_single_video_from_youtube(current_video_id, current_output_filepath, use_proxy, **video_options)
            if debug_options['print_chapters_links']:
                print(f'{current_video_link} -> {current_output_filename}')

            if debug_options.get('write_chapters_links', False) and debug_options['videos_links_filepath'] is not None:
                with open(debug_options['videos_links_filepath'], 'a') as f:
                    f.write(f'{current_video_link} -> {current_output_filename}\n')
        except Exception as e:
            print(e)
