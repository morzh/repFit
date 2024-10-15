import os.path
import ffmpeg
import yt_dlp
import pprint
from retry import retry

from utils.youtube_links_database.database_promts import chapters_data_from_database_promts


@retry(yt_dlp.utils.DownloadError, delay=2, backoff=2, max_delay=4, tries=5)
def download_single_video_chapter_from_youtube(video_id: str, output_filepath: str, video_segment: tuple[int, int] | None, **kwargs) -> None:
    """
    Description
        Download YouTube video chapter.

    :param video_id: YouTube video ID.
    :param output_filepath: output filepath for downloaded video.
    :param video_segment: video segment [time_start, time_end] in seconds.

    :key video_format: add offset to video segment. New segment will be [time_start - offset, time_end + offset]
    :key video_quality: video quality, e.g. 360, 720 or 1080
    :key use_proxy: use proxy socks5://127.0.0.1 to mitigate YouTube speed restriction in Russia
    """
    video_format = kwargs.get('video_format','mp4')
    video_quality = kwargs.get('video_quality', 720)

    video_temporal_filepath = f'__tmp__.{video_format}'
    video_partial_filepath = '__tmp__.mp4.part'

    if os.path.exists(video_temporal_filepath): os.remove(video_temporal_filepath)
    if os.path.exists(video_partial_filepath): os.remove(video_partial_filepath)

    ydl_options = {
        'verbose': True,
        'format': f'bestvideo[height={video_quality}]',
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


def download_video_chapters_from_youtube(database_filepath: str, promts_filepath: str, output_folder: str, **kwargs) -> None:
    """
    Description:
        This functions downloads chapters from YouTube.

    :param database_filepath: SQLite3 links database filepath
    :param promts_filepath: File path to .json file with include and exclude tokens for database requests
    :param output_folder: folder for downloaded videos

    :key each_include_promt_to_separate_folder: use include promt tokens to sort downloaded videos by subfolders
    :key video_chapter_offset_seconds: offsets YouTube chapters start and end time in seconds ([seconds_start - offset, seconds_end + offset])
    :key video_format: video format, e.g. 'mp4'
    :key video_quality: video quality, e.g. 720
    :key print_links: if True prints links to YouTube chapters
    :key print_chapters_data: if it is True prints chapters data
    :key print_chapters_name: if True prints chapters names
    :key use_proxy: proxy for connection to 'youtube.com'
    """

    key_per_include_promt = kwargs['each_include_promt_to_separate_folder']
    chapters_promts = chapters_data_from_database_promts(database_filepath, promts_filepath)
    video_format = kwargs.get('video_format','mp4')
    total_chapters_number = sum(len(l) for l in chapters_promts.values())
    print('Overall number of chapters is', total_chapters_number)

    chapters_links_filepath = kwargs.get('chapters_links_filepath', None)

    offset = kwargs.get('video_chapter_offset_seconds', 1)
    for chapter_folder, chapters_data in chapters_promts.items():
        current_output_folder = os.path.join(output_folder, chapter_folder) if key_per_include_promt else output_folder
        os.makedirs(current_output_folder, exist_ok=True)

        if kwargs['print_chapters_name']: print(chapter_folder, 'where chapters number are', len(chapters_data))
        if kwargs.get('print_chapters_data', False): pprint.pprint(chapters_data, indent=6, width=150)

        for chapter in chapters_data:
            current_video_id = chapter[5]
            current_time_start = max(0, int(chapter[2]) - offset)
            current_time_end = int(chapter[3]) + offset
            current_output_filename = f'{current_video_id}__{current_time_start}-{current_time_end}__.{video_format}'
            current_output_filepath = os.path.join(current_output_folder, current_output_filename)

            if os.path.exists(current_output_filepath) and os.stat(current_output_filepath).st_size > 1024:
                continue

            try:
                current_video_segment = (current_time_start, current_time_end)
                download_single_video_chapter_from_youtube(current_video_id, current_output_filepath, current_video_segment, **kwargs)

                if kwargs['print_chapters_links']:
                    current_chapter_link = f'https://www.youtube.com/watch?v={current_video_id}?start={current_time_start}&end={current_time_end}'
                    print(f'{current_chapter_link} -> {current_output_filename}')

                if kwargs.get('write_chapters_links', False) and chapters_links_filepath is not None:
                    current_chapter_link = f'https://www.youtube.com/embed/{current_video_id}?start={current_time_start}&end={current_time_end}'
                    with open(chapters_links_filepath, 'a') as f:
                        f.write(f'{current_chapter_link} -> {current_output_filename}\n')
            except Exception as e:
                print(e)

