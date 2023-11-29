from __future__ import annotations

from abc import abstractmethod

from summarizer.components.video import Video
from summarizer.components.segment import Segment

from summarizer.modules.quality import Quality
from summarizer.modules.redundancy import Redundancy
from summarizer.modules.introduction import Introduction
from summarizer.modules.subjectivity import Subjectivity


class BaseSummarizer:
    def __init__(
        self,
        videos: list[Video] = [],
        summary_name: str = "",
        frames_path: str = "",
        output_path: str = "output.mp4",
    ) -> None:
        self.__videos = videos
        self.__frames_path = frames_path
        self.__summary_video = Video(name=summary_name, path=output_path)

    @abstractmethod
    def summarize(self) -> Video:
        pass

    def get_frames_path(self) -> str:
        return self.__frames_path

    def get_videos(self) -> list[Video]:
        return self.__videos

    def get_video_at(self, index: int) -> None:
        return self.__videos[index]

    def append_segment_to_summary(self, segment: Segment) -> None:
        self.__summary_video.append_segment(segment)

    def get_summary_video(self) -> Video:
        return self.__summary_video

    def introduction(self, include: bool = True) -> BaseSummarizer:
        if include:
            return Introduction(self).include()
        return Introduction(self).exclude()

    def subjectivity(self, include: bool = True) -> BaseSummarizer:
        if include:
            return Subjectivity(self).include()
        return Subjectivity(self).exclude()

    def redundancy(self, include: bool = True) -> BaseSummarizer:
        if include:
            return Redundancy(self).include()
        return Redundancy(self).exclude()

    def quality(self, include: bool = True) -> BaseSummarizer:
        if include:
            return Quality(self).include()
        return Quality(self).exclude()


class HSMVideoSumm(BaseSummarizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def summarize(self) -> Video:
        return (
            self.introduction(include=True)
            .subjectivity(include=False)
            .redundancy(include=True)
            .get_summary_video()
        )
