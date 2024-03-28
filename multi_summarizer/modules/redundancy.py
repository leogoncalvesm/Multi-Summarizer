from __future__ import annotations

from pandas import DataFrame
from numpy import equal, tril
from itertools import chain

from components.video import Video
from components.segment import Segment
from processing.text import BagOfWords
from processing.utils import custom_cosine
from modules.quality import Quality
from modules.chronology import Chronology
from modules.modules_base import SelectionCriteria
from summarizers.base_summarizer import BaseSummarizer


class Redundancy(SelectionCriteria):
    def __init__(self, summarizer: BaseSummarizer) -> None:
        self.__summarizer = summarizer

    def include(self) -> BaseSummarizer:
        cluster_redundancies = self.__get_redundancy_clusters()

        # Mapping each clustered segment for the videos to summarize as {segment: (video index, segment index)}
        segment_index = {
            self.__segment_from_indexes(item[0], item[1]): item
            for item in set(chain(*cluster_redundancies))
        }

        # Applying Quality selection criteria to retrieve the 1 segment with best quality for each cluster
        quality = Quality(
            summarizer=BaseSummarizer(
                videos=[
                    Video(
                        segments=list(map(self.__segment_from_indexes, *zip(*cluster))),
                        assign_to_segments=False,
                    )
                    for cluster in cluster_redundancies
                ],
                frames_path=self.__summarizer.get_frames_path(),
            )
        )

        # Retrieving best segments as their corresponding (video index, segment index) tuple
        best_segments = [
            segment_index.get(best)
            for best in quality.best_segments_for_videos(n_segments=1, flatten=True)
        ]

        # Ordering by chronology strategy and mapping (video index, segment index) tuple back to segment object
        segments_to_include = [
            self.__segment_from_indexes(seg_item[0], seg_item[1])
            for seg_item in Chronology.order_by_similarity_cluster(
                items_cluster={
                    item: cluster
                    for item, cluster in zip(best_segments, cluster_redundancies)
                }
            )
        ]

        # Including segments in video summary
        self.__summarizer.append_segments_to_summary(segments_to_include)

        return self.__summarizer

    def exclude(self) -> BaseSummarizer:
        cluster_redundancies = self.__get_redundancy_clusters()
        for cluster in cluster_redundancies:
            for video_index, segment_index in cluster:
                self.__summarizer.get_video_at(video_index).delete_segment_at(
                    segment_index
                )

        return self.__summarizer

    def __get_redundancy_clusters(self) -> list[set[tuple[int, int]]]:
        bow_df = self.__generate_bow_df()
        correlations = self.__calculate_bow_correlations(bow_df)
        redundancies = self.__find_redundancies(correlations)
        cluster_redundancies = self.__cluster_redundancies(redundancies)

        return cluster_redundancies

    def __segment_from_indexes(self, video_index: int, segment_index: int) -> Segment:
        return self.__summarizer.get_video_at(video_index).get_segment(segment_index)

    def __generate_bow_df(self) -> DataFrame:
        bow = BagOfWords(
            {
                (vid_index, seg_index): segment.get_content()
                for vid_index, video in enumerate(self.__summarizer.get_videos())
                for seg_index, segment in enumerate(video.get_segments())
            }
        )
        bow.items_preprocessing()
        return bow.generate_bow_dataframe(["video_index", "segment_index"])

    def __calculate_bow_correlations(self, bow_df: DataFrame) -> DataFrame:
        # Generating BoW correlations matrix
        correlations = bow_df.T.corr(custom_cosine)

        # Disregarding same-video comparisons
        is_same_video = equal.outer(
            correlations.index.get_level_values("video_index"),
            correlations.columns.get_level_values("video_index"),
        )
        # Keeping only the upper diagonal of the pairwise comparisons
        is_upper_diagonal = tril(correlations) > 0
        # Keeping only similarities greater than treshold
        is_gt_threshold = correlations.gt(self.__calc_minimum_threshold())

        # Masking correlations matrix
        correlations = (
            correlations.mask(is_same_video | is_upper_diagonal | ~is_gt_threshold)
            .dropna(axis="index", how="all")
            .dropna(axis="columns", how="all")
        )

        # Finding text-pair matches
        correlations = correlations.melt(ignore_index=False).dropna(
            "index", subset=["value"]
        )
        correlations.columns = ["video_index_col", "segment_index_col", "value"]
        correlations.reset_index(inplace=True)
        return correlations

    def __find_redundancies(self, correlations: DataFrame) -> DataFrame:
        redundancies = correlations[
            correlations.groupby(["video_index", "video_index_col"])["value"].transform(
                max
            )
            == correlations["value"]
        ]

        # Setting video index and segment index as one tuple object
        redundancies["video"] = tuple(
            zip(redundancies["video_index"], redundancies["segment_index"])
        )
        redundancies["match"] = tuple(
            zip(redundancies["video_index_col"], redundancies["segment_index_col"])
        )
        return redundancies

    def __cluster_redundancies(
        self, redundancies: DataFrame
    ) -> list[set[tuple[int, int]]]:
        # Retrieving all matches from DataFrame and clustering where elements intersect
        all_matches = redundancies[["video", "match"]].values.tolist()
        return self.__cluster_matches(all_matches)

    def __cluster_matches(
        self, matches: list[list[tuple[int, int]]]
    ) -> list[set[tuple[int, int]]]:
        clusters, locations = [], {}  # hashtable of which cluster each element is in

        for item_a, item_b in matches:
            if item_a in locations:
                clusters[locations.get(item_a)].add(item_b)
            elif item_b in locations:
                clusters[locations.get(item_b)].add(item_a)
            else:
                clusters.append({item_a, item_b})
                locations[item_a] = len(clusters) - 1
                locations[item_b] = len(clusters) - 1

        return clusters

    def __calc_minimum_threshold(self):
        set_time = sum(
            video.get_segments()[-1].get_end()
            for video in self.__summarizer.get_videos()
        )
        dif = (set_time - 785) / 785
        return 0.17 + 0.17 * dif
