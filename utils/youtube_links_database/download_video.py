import yt_dlp
from yt_dlp.utils import download_range_func


def download_youtube_video_chapter(video_id: str,
                                   output_file_path_basename: str,
                                   chapter_name: str | None = None,
                                   video_segment: tuple[float, float] | None = None) -> None:
    """
    Description
        Download YouTube video chapter.

    :param video_id: YouTube video ID
    :param output_file_path_basename: output video file path base name. Video file extension will be added automatically.
    :param chapter_name: video chapter to download
    :param video_segment: video segment [time_start, time_end] in seconds
    """
    ydl_opts = {
        'verbose': True,
        'format': 'best',
        'download_ranges': download_range_func([chapter_name], [video_segment]),
        'force_keyframes_at_cuts': True,
        'outtmpl': f'{output_file_path_basename}.%(ext)s'
    }
    youtube_link_base = 'https://www.youtube.com/watch?v='
    youtube_video_url = youtube_link_base + video_id

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(youtube_video_url)
