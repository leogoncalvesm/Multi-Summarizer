from __future__ import annotations

from modules.modules_base import SelectionCriteria

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from summarizers.base_summarizer import BaseSummarizer


class Chronology(SelectionCriteria):
    def __init__(self, summarizer: BaseSummarizer) -> None:
        self.__summarizer = summarizer

    def include(self) -> BaseSummarizer:
        # TODO
        return self.__summarizer

    def exclude(self) -> BaseSummarizer:
        # TODO
        return self.__summarizer

    @staticmethod
    def order_by_similarity_cluster(
        items_cluster: dict[tuple[int, int], set[tuple[int, int]]]
    ) -> list[int, int]:
        result = []
        for item, cluster in items_cluster.items():
            ind = Chronology.find_insert_position(item, cluster, result)
            result.insert(ind, item)

    @staticmethod
    def find_same_video_in_cluster(video_index, cluster):
        for cluster_item in cluster:
            if video_index == cluster_item[0]:
                return cluster_item
        return {}

    @staticmethod
    def find_insert_position(new_item, cluster, result_items):
        insert_position = 0
        for res_vid, res_seg in result_items:
            same_video_item = Chronology.find_same_video_in_cluster(res_vid, cluster)
            is_later = (
                same_video_item[1] > res_seg
                if same_video_item
                else new_item[1] > res_seg
            )
            if not is_later:
                return insert_position
            insert_position += 1
        return insert_position
