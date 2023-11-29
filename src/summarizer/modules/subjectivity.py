from __future__ import annotations

from summarizer.components.video import Video
from summarizer.components.segment import Segment
from summarizer.processing.image import FaceDetector
from summarizer.modules.modules_base import SelectionCriteria
from summarizer.processing.text import SubjectivityGoogleAPI
from summarizer.processing.models import (
    FACE_CLASSIFIER,
    SENTIMENT_API_RESULTS,
    SENTILEX_DATA_PT,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from summarizer.components.summarizer import BaseSummarizer


class Subjectivity(SelectionCriteria):
    def __init__(self, summarizer: BaseSummarizer) -> None:
        self.__summarizer = summarizer
        self.__face_classifier = FaceDetector(FACE_CLASSIFIER)

    def include(self) -> BaseSummarizer:
        self.__clear_videos_segments(False)
        return self.__summarizer

    def exclude(self) -> BaseSummarizer:
        self.__clear_videos_segments(True)
        return self.__summarizer

    def __clear_videos_segments(self, remove_subjective: bool) -> None:
        for video in self.__summarizer.get_videos():
            self.__remove_segments(video, remove_subjective)

    def __remove_segments(self, video: Video, remove_subjective: bool) -> None:
        video_segments, segs_delete = video.get_segments(), []

        segments_to_delete = [
            i
            for i, segment in enumerate(video_segments)
            if self.__is_segment_subjective(segment) == remove_subjective
        ]

        # Deleting objective/subjective segments from last to first
        for i in sorted(segments_to_delete, reverse=True):
            video.delete_segment(i)

    def __is_segment_subjective(self, segment: Segment) -> bool:
        if not self.__segment_contains_faces(segment):
            return False
        text_subj_classifier = SubjectivityGoogleAPI(
            sentilex_path=SENTILEX_DATA_PT,
            sentiment_data_path=SENTIMENT_API_RESULTS,
        )

        return text_subj_classifier.is_subjective(segment.get_content())

    def __segment_contains_faces(self, segment: Segment) -> bool:
        for frame in segment.load_frames(self.__summarizer.get_frames_path()):
            if self.__face_classifier.frame_contains_face(frame):
                return True
        return False
