from __future__ import annotations

from summarizer.modules.modules_base import SelectionCriteria

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from summarizer.components.summarizer import BaseSummarizer


class Quality(SelectionCriteria):
    def __init__(self, summarizer: BaseSummarizer) -> None:
        self.__summarizer = summarizer

    def include(self) -> BaseSummarizer:
        # TODO
        return self.__summarizer

    def exclude(self) -> BaseSummarizer:
        # TODO
        return self.__summarizer
