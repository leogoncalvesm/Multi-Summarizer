import time
import argparse
from os import listdir, mkdir
from datetime import datetime, timedelta
from os.path import join, exists, normpath, basename, dirname


def log(message: str, log_type: str = "INFO", log_file: str = "logs.txt") -> None:
    """Application logging function"""

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{now}] {log_type}: {message}"
    print(log_message)
    # with open(log_file, "a") as f:
    #     f.write(f"{log_message}\n")


def custom_cosine(v1, v2):
    cos = 0
    for i in range(len(v1)):
        cos = cos + v1[i] * v2[i]

    return cos


def process_arguments() -> tuple[str, str, list[str]]:
    """Parses project call parameters"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-vp",
        "--videos-path",
        required=True,
        help="Path of the folder containing the videos content folders",
    )
    parser.add_argument(
        "-n",
        "--name",
        default="",
        help="Name of the video set. Default is the video-set folder name",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="Output file name of the summarized video",
    )
    parser.add_argument("-v", "--videos", action="append", nargs="+")
    args = parser.parse_args()

    video_set_name = args.name if args.name else basename(normpath(args.videos_path))
    output = args.output if args.output else join("results", f"{video_set_name}.mp4")

    # Creating output directory if not exists
    output_directory = dirname(output)
    if not exists(output_directory):
        mkdir(output_directory)

    # Checking for videos dataset path
    if not exists(args.videos_path):
        raise Exception(f"Video dataset '{args.videos_path}' not found")

    # Using all videos in directory if none passed by parameter
    videos_list = args.videos[0] if args.videos else listdir(args.videos_path)

    # Validating if there're at least two videos to summarize
    if len(videos_list) < 2:
        raise Exception(
            f"Not enough videos to summarize. At least two videos are required"
        )

    # Validating if all videos passed exist in directory
    if any(
        not exists(join(args.videos_path, video, f"{video}.mp4"))
        for video in videos_list
    ):
        raise Exception(f"Non existing video passed")

    return {
        "name": video_set_name,
        "path": (args.videos_path),
        "videos": videos_list,
        "output": output,
    }


def get_seconds_from_time(time_str: str, time_format="%H:%M:%S") -> int:
    """Converts time in given format to seconds"""
    time_obj = time.strptime(time_str, time_format)
    return int(
        timedelta(
            hours=time_obj.tm_hour, minutes=time_obj.tm_min, seconds=time_obj.tm_sec
        ).total_seconds()
    )
