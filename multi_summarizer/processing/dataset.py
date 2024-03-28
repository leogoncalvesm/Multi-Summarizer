import json
from os.path import join, exists
from dataclasses import dataclass
from os import system, listdir, makedirs

from components.video import Video, Segment
from processing.utils import get_seconds_from_time


@dataclass
class Dataset:
    name: str
    path: str
    videos: list[str]


@dataclass
class DatasetVideo:
    name: str
    video_file: str
    content_file: str


class DatasetLoader:
    def __init__(
        self, dataset: Dataset, video_format: str = "mp4", content_format: str = "json"
    ) -> None:
        self.__dataset = dataset
        self.__video_format = video_format
        self.__content_format = content_format
        self.__videos = self.__load_dataset_videos()

    def load_videos(self) -> Video:
        return [self.__load_video(video) for video in self.__videos]

    def __load_video(self, video: DatasetVideo) -> list[Segment]:
        """Loads video object and all of its segments from given video in dataset"""
        return Video(
            name=video.name,
            path=video.video_file,
            segments=[
                Segment(
                    begin=get_seconds_from_time(obj.get("begin")),
                    end=get_seconds_from_time(obj.get("end")),
                    content=obj.get("content"),
                )
                for obj in json.load(open(video.content_file))
            ],
        )

    def __get_video_filename(self, video_name: str, file_ext: str) -> str:
        return join(self.__dataset.path, video_name, f"{video_name}.{file_ext}")

    def __load_dataset_videos(self) -> list[DatasetVideo]:
        return [
            DatasetVideo(
                name=video,
                video_file=self.__get_video_filename(video, self.__video_format),
                content_file=self.__get_video_filename(video, self.__content_format),
            )
            for video in self.__dataset.videos
        ]

    def save_video_frames(self) -> str:
        """
        Runs ffmpeg to save 1 frame per second for every video in the dataset.
        If video directory exists and is not empty, does nothing.
        """
        dataset_frames = join("video_frames", self.__dataset.name)

        for video in self.__videos:
            frames_dir = join(dataset_frames, video.name)

            # Checking whether rames have been extracted before
            if exists(frames_dir) and listdir(frames_dir):
                continue

            if not exists(frames_dir):
                makedirs(frames_dir)

            frames_path = join(frames_dir, "image-%d.jpg")
            ffmpeg_cmd = f"ffmpeg -i '{video.video_file}' -hide_banner -loglevel error -vf fps=1 -start_number 0 '{frames_path}'"
            system(ffmpeg_cmd)

        return dataset_frames
