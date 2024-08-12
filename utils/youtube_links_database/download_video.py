import yt_dlp
from yt_dlp.utils import download_range_func

# yt-dlp https://www.youtube.com/watch?v=xMOHrl_is4Y --downloader ffmpeg --downloader-args "ffmpeg_i:-ss 997 -to 2512"


def download_video_chapters(youtube_video_url: str, chapters: list | None = None, video_segments: list[tuple] | None = None):
    ydl_opts = {
        'verbose': True,
        'format': 'best[ext=mp4]',
        'download_ranges': download_range_func(chapters, video_segments),
        'force_keyframes_at_cuts': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(youtube_video_url)
