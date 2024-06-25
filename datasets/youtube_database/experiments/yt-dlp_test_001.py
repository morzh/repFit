import yt_dlp
import pickle
import pprint


def delete_redundant_keys(data):
    del data['abr']
    del data['acodec']
    del data['aspect_ratio']
    del data['automatic_captions']
    del data['audio_channels']
    del data['asr']
    del data['average_rating']
    del data['age_limit']
    del data['availability']
    del data['channel_is_verified']
    del data['channel_follower_count']
    del data['comment_count']
    del data['display_id']
    del data['dynamic_range']
    del data['duration_string']
    del data['extractor']
    del data['extractor_key']
    del data['epoch']
    del data['formats']
    del data['format']
    del data['format_id']
    del data['format_note']
    del data['heatmap']
    del data['like_count']
    del data['live_status']
    del data['is_live']
    del data['requested_formats']
    del data['playlist_index']
    del data['playlist']
    del data['protocol']
    del data['playable_in_embed']
    del data['release_year']
    del data['resolution']
    del data['requested_subtitles']
    del data['release_timestamp']
    del data['stretched_ratio']
    del data['subtitles']
    del data['tbr']
    del data['thumbnail']
    del data['thumbnails']
    del data['timestamp']
    del data['uploader']
    del data['uploader_url']
    del data['uploader_id']
    del data['vbr']
    del data['view_count']
    del data['vcodec']
    del data['webpage_url_domain']
    del data['webpage_url_basename']
    del data['was_live']
    del data['_has_drm']
    del data['_format_sort_fields']


def yt_dlp_extract_info(video_url, print_info=False):
    with yt_dlp.YoutubeDL({}) as ydl:
        video_info = ydl.extract_info(video_url, download=False)
        delete_redundant_keys(video_info)
        if print_info:
            pprint.pprint(video_info)
    return video_info


'''
yt_opts = {
    'verbose': True,
    'skip_download': True,
    'dump_single_json': True,
    'write_json': True,
    'write_info_json': True,
    'write_annotations': True,
    'write_thumbnail': True,
}
'''

url = "https://www.youtube.com/watch?v=b1Fo_M_tj6w"
current_info = yt_dlp_extract_info(url)
with open(current_info['id'] + '.pickle', 'wb') as f:
    pickle.dump(current_info, f, pickle.HIGHEST_PROTOCOL)
