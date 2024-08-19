import os.path

import ffmpeg
import yt_dlp
from retry import retry


@retry(yt_dlp.utils.DownloadError, delay=2, backoff=2, max_delay=4, tries=5)
def download_youtube_video_chapter(video_id: str, output_file_path_basename: str, video_segment: tuple[float, float] | None = None) -> None:
    """
    Description
        Download YouTube video chapter.

    :param video_id: YouTube video ID
    :param output_file_path_basename: output video file path base name. Video file extension will be added automatically.
    :param chapter_name: video chapter to download
    :param video_segment: video segment [time_start, time_end] in seconds
    """
    video_format = 'mp4'
    video_height = 720

    video_temporal_filepath = f'__tmp__.{video_format}'
    video_output_filepath = f'{output_file_path_basename}.{video_format}'

    if os.path.exists(video_temporal_filepath):
        os.remove(video_temporal_filepath)

    ydl_opts = {
        'verbose': True,
        'format': f'{video_format}[height={video_height}]',
        # 'download_ranges': download_range_func([chapter_name], [video_segment]),
        'force_keyframes_at_cuts': True,
        'outtmpl': video_temporal_filepath,
        'prefer_ffmpeg': True,
        'proxy': "socks5://127.0.0.1",  # just to avoid YouTube restrictions
    }
    youtube_link_base = 'https://www.youtube.com/watch?v='
    youtube_video_url = youtube_link_base + video_id

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(youtube_video_url)

    pts = 'PTS-STARTPTS'
    input_stream = ffmpeg.input(video_temporal_filepath)
    video_cut = input_stream.trim(start=video_segment[0] - 1, end=video_segment[1] + 1).setpts(pts)
    output_video = ffmpeg.output(video_cut, video_output_filepath, format='mp4')
    output_video.run()





