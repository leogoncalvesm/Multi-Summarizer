from __future__ import annotations

from summarizer.components.video import Video
from summarizer.modules.quality import Quality
from summarizer.modules.redundancy import Redundancy
from summarizer.modules.introduction import Introduction
from summarizer.modules.subjectivity import Subjectivity
from summarizer.summarizers.base_summarizer import BaseSummarizer


class HSMVideoSumm(BaseSummarizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def summarize(self) -> Video:
        return (
            self.__introduction(include=True)
            .__subjectivity(include=False)
            .__redundancy(include=True)
            .get_summary_video()
        )

    def __introduction(self, include: bool = True) -> HSMVideoSumm:
        if include:
            return Introduction(self).include()
        return Introduction(self).exclude()

    def __subjectivity(self, include: bool = True) -> HSMVideoSumm:
        if include:
            return Subjectivity(self).include()
        return Subjectivity(self).exclude()

    def __redundancy(self, include: bool = True) -> HSMVideoSumm:
        if include:
            return Redundancy(self).include()
        return Redundancy(self).exclude()

    def __quality(self, include: bool = True) -> HSMVideoSumm:
        if include:
            return Quality(self).include()
        return Quality(self).exclude()
