from __future__ import annotations

from summarizer.components.frame import Frame

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from summarizer.components.video import Video


class Segment:
    def __init__(
        self, begin: int, end: int, content: str = "", video: Video = None
    ) -> None:
        self.__begin = begin
        self.__end = end
        self.__content = content
        self.__video: video

    def get_duration(self) -> int:
        """Gets segment's duration in seconds"""
        return self.__end - self.__begin

    def __ge__(self, other: Segment) -> bool:
        return self.get_duration() >= other.get_duration()

    def __gt__(self, other: Segment) -> bool:
        return self.get_duration() > other.get_duration()

    def get_begin(self) -> int:
        return self.__begin

    def get_end(self) -> int:
        return self.__end

    def set_end(self, end: int) -> None:
        self.__end = end

    def get_video(self) -> Video:
        return self.__video

    def set_video(self, video: Video) -> None:
        self.__video = video

    def get_content(self) -> str:
        return self.__content

    def load_frames(self, frames_path: str, sort: bool = False) -> list[Frame]:
        frames = self.__video.load_frames(frames_path, sort=sort)
        segment_range = range(self.__begin, self.__end)
        return list(filter(lambda f: f.get_video_second() in segment_range, frames))
