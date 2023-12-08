from __future__ import annotations

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from summarizer.summarizers.base_summarizer import BaseSummarizer


class SelectionCriteria(ABC):
    @abstractmethod
    def include(self) -> BaseSummarizer:
        pass

    @abstractmethod
    def exclude(self) -> BaseSummarizer:
        pass
