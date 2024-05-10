from __future__ import annotations

from abc import abstractmethod

from components.video import Video
from components.segment import Segment


class BaseSummarizer:
    def __init__(
        self,
        videos: list[Video] = [],
        summary_name: str = "",
        frames_path: str = "",
        output_path: str = "output.mp4",
    ) -> None:
        self.__videos = videos
        self.__summary_name = summary_name
        self.__frames_path = frames_path
        self.__output_path = output_path
        self.__summary_video = None

    @abstractmethod
    def summarize(self) -> Video:
        pass

    def get_frames_path(self) -> str:
        return self.__frames_path

    def get_videos(self) -> list[Video]:
        return self.__videos

    def get_video_at(self, index: int) -> Video:
        return self.__videos[index]

    def start_summary_video(self) -> None:
        self.__summary_video = Video(name=self.__summary_name, path=self.__output_path)

    def append_segment_to_summary(self, segment: Segment) -> None:
        self.__summary_video.append_segment(segment)

    def append_segments_to_summary(self, segments: list[Segment]) -> None:
        for segment in segments:
            self.append_segment_to_summary(segment)

    def adjust_summary_segments_seconds(self) -> None:
        new_begin = 0
        for segment in self.__summary_video.get_segments():
            new_end = new_begin + segment.get_duration()
            segment.set_begin(new_begin)
            segment.set_end(new_end)
            new_begin = new_end

    def get_summary_video(self) -> Video:
        return self.__summary_video

    def print_summary(self) -> None:
        self.adjust_summary_segments_seconds()
        for segment in self.__summary_video.get_segments():
            print(segment)
