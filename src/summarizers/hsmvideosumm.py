from __future__ import annotations

from components.video import Video
from modules.quality import Quality
from modules.redundancy import Redundancy
from modules.introduction import Introduction
from modules.subjectivity import Subjectivity
from summarizers.base_summarizer import BaseSummarizer


class HSMVideoSumm(BaseSummarizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def summarize(self) -> Video:
        return (
            self.start_summary_video()
            .__introduction(include=True)
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
