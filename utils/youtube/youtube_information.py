import yt_dlp
import pprint


def delete_redundant_video_keys(info: dict) -> None:
    info.pop('abr', None)
    info.pop('acodec', None)
    info.pop('aspect_ratio', None)
    info.pop('automatic_captions', None)
    info.pop('audio_channels', None)
    info.pop('asr', None)
    info.pop('average_rating', None)
    info.pop('age_limit', None)
    info.pop('availability', None)
    info.pop('channel_is_verified', None)
    info.pop('channel_follower_count', None)
    info.pop('channel', None)
    info.pop('channel_id', None)
    info.pop('channel_url', None)
    info.pop('comment_count', None)
    info.pop('display_id', None)
    info.pop('dynamic_range', None)
    info.pop('duration_string', None)
    info.pop('extractor', None)
    info.pop('extractor_key', None)
    info.pop('epoch', None)
    info.pop('formats', None)
    info.pop('format', None)
    info.pop('format_id', None)
    info.pop('format_note', None)
    info.pop('heatmap', None)
    info.pop('like_count', None)
    info.pop('live_status', None)
    info.pop('is_live', None)
    info.pop('requested_formats', None)
    info.pop('playlist_index', None)
    info.pop('playlist', None)
    info.pop('protocol', None)
    info.pop('playable_in_embed', None)
    info.pop('release_year', None)
    info.pop('resolution', None)
    info.pop('requested_subtitles', None)
    info.pop('release_timestamp', None)
    info.pop('stretched_ratio', None)
    info.pop('subtitles', None)
    info.pop('tbr', None)
    info.pop('thumbnail', None)
    info.pop('thumbnails', None)
    info.pop('timestamp', None)
    info.pop('uploader', None)
    info.pop('uploader_url', None)
    info.pop('uploader_id', None)
    info.pop('vbr', None)
    info.pop('view_count', None)
    info.pop('vcodec', None)
    info.pop('webpage_url_domain', None)
    info.pop('webpage_url_basename', None)
    info.pop('webpage_url', None)
    info.pop('was_live', None)
    info.pop('_has_drm', None)
    info.pop('_format_sort_fields', None)


def delete_redundant_channel_keys(info: dict) -> None:
    info.pop('entries', None)
    info.pop('thumbnails', None)
    info.pop('view_count', None)
    info.pop('webpage_url_basename', None)
    info.pop('webpage_url_domain', None)
    info.pop('_type', None)
    info.pop('availability', None)
    info.pop('uploader', None)
    info.pop('uploader_id', None)
    info.pop('uploader_url', None)
    info.pop('webpage_url', None)
    info.pop('epoch', None)
    info.pop('extractor', None)
    info.pop('extractor_key', None)
    info.pop('modified_date', None)
    info.pop('playlist_count', None)
    info.pop('release_year', None)
    info.pop('requested_entries', None)
    info.pop('title', None)
    info.pop('channel_follower_count', None)
    info.pop('channel_url', None)
    info.pop('channel_is_verified', None)


def extract_youtube_video_information(video_url, print_info=False):
    with yt_dlp.YoutubeDL({}) as ydl:
        video_info = ydl.extract_info(video_url, download=False)
        delete_redundant_video_keys(video_info)
        if print_info:
            pprint.pprint(video_info)
    return video_info


def extract_youtube_channel_information(youtube_channel_url, print_info=False):
    ydl_opts = {
        'playlist_items': '1',
        'extract_flat': 'in_playlist',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        channel_info = ydl.extract_info(youtube_channel_url, download=False)
        delete_redundant_channel_keys(channel_info)
        if print_info:
            pprint.pprint(channel_info)
    return channel_info
