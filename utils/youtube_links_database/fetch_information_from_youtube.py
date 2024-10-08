import yt_dlp
import pprint
from loguru import logger
from retry import retry

"""Video information which is redundant for repFit database"""
redundant_video_keys_list = [
    'abr',
    'acodec',
    'aspect_ratio',
    'automatic_captions',
    'audio_channels',
    'asr',
    'average_rating',
    'age_limit',
    'availability',
    'channel_is_verified',
    'channel_follower_count',
    'channel',
    'channel_id',
    'channel_url',
    'comment_count',
    'display_id',
    'dynamic_range',
    'duration_string',
    'extractor',
    'extractor_key',
    'epoch',
    'formats',
    'format',
    'format_id',
    'format_note',
    'heatmap',
    'like_count',
    'live_status',
    'is_live',
    'requested_formats',
    'playlist_index',
    'playlist',
    'protocol',
    'playable_in_embed',
    'release_year',
    'resolution',
    'requested_subtitles',
    'release_timestamp',
    'stretched_ratio',
    'subtitles',
    'tbr',
    'thumbnail',
    'thumbnails',
    'timestamp',
    'uploader',
    'uploader_url',
    'uploader_id',
    'vbr',
    'view_count',
    'vcodec',
    'webpage_url_domain',
    'webpage_url_basename',
    'webpage_url',
    'was_live',
    '_has_drm',
    '_format_sort_fields',
]

"""Video information which is redundant for repFit database"""
redundant_channel_keys_list = [
    'entries',
    'thumbnails',
    'view_count',
    'webpage_url_basename',
    'webpage_url_domain',
    '_type',
    'availability',
    'uploader',
    'uploader_id',
    'uploader_url',
    'webpage_url',
    'epoch',
    'extractor',
    'extractor_key',
    'modified_date',
    'playlist_count',
    'release_year',
    'requested_entries',
    'title',
    'channel_follower_count',
    'channel_url',
    'channel_is_verified',

] 


def delete_keys_from_dictionary(video_information: dict, keys: list) -> None:
    """
    Description:
        Delete given keys from dictionary
    :param video_information: video information dict
    :param keys: keys to delete from dictionary
    """
    for key in keys:
        video_information.pop(key, None)


@retry((yt_dlp.utils.UnsupportedError, yt_dlp.utils.DownloadError), delay=1, backoff=2, max_delay=4, tries=5, logger=logger)
def fetch_youtube_video_information(video_url: str, use_proxy: bool = True, verbose: bool = False) -> dict:
    """
    Description:
        Fetch YouTube video information

    :param video_url: URL to YouTube video
    :param use_proxy: use proxy
    :param verbose: print fetched information

    :return: video information
    """
    ydl_options = {
        'socket_timeout': 40
    }
    if use_proxy:
        ydl_options['proxy'] = "socks5://127.0.0.1"

    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        video_info = ydl.extract_info(video_url, download=False)
        delete_keys_from_dictionary(video_info, redundant_video_keys_list)
        if verbose:
            pprint.pprint(video_info)
    return video_info


@retry((yt_dlp.utils.UnsupportedError, yt_dlp.utils.DownloadError), delay=1, backoff=2, max_delay=4, tries=3, logger=logger)
def fetch_youtube_channel_information(youtube_channel_url: str, use_proxy: bool = True, verbose: bool = False) -> dict:
    """
    Description:
        Fetch YouTube channel information
    :param youtube_channel_url: URL to YouTube channel
    :param use_proxy: use proxy
    :param verbose: print fetched information

    :return: channel information
    """
    ydl_options = {
        'playlist_items': '1',
        'extract_flat': 'in_playlist',
        'socket_timeout': 40
    }
    if use_proxy:
        ydl_options['proxy'] = "socks5://127.0.0.1"

    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        channel_info = ydl.extract_info(youtube_channel_url, download=False)
        delete_keys_from_dictionary(channel_info, redundant_channel_keys_list)
        if verbose:
            pprint.pprint(channel_info)
    return channel_info
