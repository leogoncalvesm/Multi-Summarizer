from __future__ import annotations

from itertools import chain

from summarizer.components.segment import Segment
from summarizer.modules.modules_base import SelectionCriteria
from summarizer.processing.image import BagOfVisualWords, ImageProcessing

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from summarizer.summarizers.base_summarizer import BaseSummarizer


class Quality(SelectionCriteria):
    def __init__(self, summarizer: BaseSummarizer) -> None:
        self.__summarizer = summarizer

    def include(self) -> BaseSummarizer:
        # TODO
        return self.__summarizer

    def exclude(self) -> BaseSummarizer:
        # TODO
        return self.__summarizer

    def best_segments_for_videos(
        self, n_segments: int, flatten: bool = False
    ) -> list[Segment] | list[list[Segment]]:
        videos_best_segs = []
        for video in self.__summarizer.get_videos():
            # Getting Bag of Visual Words for the segments in the video
            bovw = BagOfVisualWords(
                items={
                    segment: ImageProcessing.ks_sift(
                        segment, self.__summarizer.get_frames_path()
                    )
                    for segment in video.get_segments()
                },
                dict_size=300,
            )
            bovw.fit_kmeans()
            df = bovw.generate_bovw_dataframe()

            # Summing the histogram features for each segment
            df["histogram_sum"] = df.sum(axis=1)

            videos_best_segs.append(
                df.nlargest(n_segments, columns="histogram_sum").index.to_list()
            )
        if flatten:
            videos_best_segs = list(chain.from_iterable(videos_best_segs))

        return videos_best_segs

    def get_segment_quality(self, segment: Segment) -> float:
        return ImageProcessing.ks_sift(segment, self.__summarizer.__frames_path)
