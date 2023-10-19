from __future__ import annotations

from abc import ABC, abstractmethod

from summarizer.components.video import Video
from summarizer.modules.introduction.introduction import Introduction
from summarizer.modules.subjectivity.subjectivity import Subjectivity
from summarizer.modules.redundancy.redundancy import Redundancy


class BaseSummarizer(ABC):
    pass


class HSMVideoSumm(BaseSummarizer):
    def __init__(
        self, summary_name: str, frames_path: str, output_path: str, videos: list[Video]
    ) -> None:
        self.videos = videos
        self.frames_path = frames_path
        self.summary_video = Video(name=summary_name, path=output_path)

    def save_summary(self):
        self.summary_video.write_from_segments()

    def introduction(self, include: bool = True) -> HSMVideoSumm:
        if include:
            return Introduction(self).include()
        return Introduction(self).exclude()

    def subjectivity(self, include: bool = True) -> HSMVideoSumm:
        if include:
            return Subjectivity(self).include()
        return Subjectivity(self).exclude()

    def redundancy(self, include: bool = True) -> HSMVideoSumm:
        if include:
            return Redundancy(self).include()
        return Redundancy(self).exclude()
