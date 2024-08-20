import os.path
import ffmpeg
import yt_dlp
from retry import retry


@retry(yt_dlp.utils.DownloadError, delay=2, backoff=2, max_delay=4, tries=5)
def download_youtube_video_chapter(video_id: str,
                                   output_filepath: str,
                                   video_segment: tuple[int, int] | None,
                                   **kwargs) -> None:
    """
    Description
        Download YouTube video chapter.

    :param video_id: YouTube video ID.
    :param output_filepath: output filepath for downloaded video.
    :param video_segment: video segment [time_start, time_end] in seconds.

    :key video_format: add offset to video segment. New segment will be [time_start - offset, time_end + offset]
    :key video_quality: video quality, e.g. 360, 720 or 1080
    """
    video_format = kwargs.get('video_format','mp4')
    video_quality = kwargs.get('video_quality', 720)

    video_temporal_filepath = f'__tmp__.{video_format}'

    if os.path.exists(video_temporal_filepath):
        os.remove(video_temporal_filepath)

    ydl_options = {
        'verbose': True,
        'format': f'{video_format}[height={video_quality}]',
        # 'download_ranges': download_range_func([chapter_name], [video_segment]),
        'force_keyframes_at_cuts': True,
        'outtmpl': video_temporal_filepath,
        'prefer_ffmpeg': True,
    }
    if kwargs.get('use_proxy', False):
        ydl_options['proxy'] = "socks5://127.0.0.1"

    youtube_link_base = 'https://www.youtube.com/watch?v='
    youtube_video_url = youtube_link_base + video_id

    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        ydl.download(youtube_video_url)

    pts = 'PTS-STARTPTS'
    input_stream = ffmpeg.input(video_temporal_filepath)
    video_cut = input_stream.trim(start=video_segment[0], end=video_segment[1]).setpts(pts)
    output_video = ffmpeg.output(video_cut, output_filepath, format='mp4')
    output_video.run()
