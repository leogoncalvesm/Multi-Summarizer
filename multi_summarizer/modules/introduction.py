from __future__ import annotations

from components.video import Video
from components.segment import Segment
from processing.image import ImageProcessing

from modules.modules_base import SelectionCriteria

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from summarizers.base_summarizer import BaseSummarizer


class Introduction(SelectionCriteria):
    def __init__(self, summarizer: BaseSummarizer) -> None:
        self.__summarizer = summarizer

    def include(self) -> BaseSummarizer:
        shortest_intro = self.__get_shortest_introduction()
        if shortest_intro:
            self.__summarizer.append_segments_to_summary(shortest_intro)
        return self.__summarizer

    def exclude(self) -> BaseSummarizer:
        self.__remove_introductions()
        return self.__summarizer

    def __remove_introductions(self) -> None:
        for video in self.__summarizer.get_videos():
            intro_end_sec = self.__find_introduction_end_second(video)
            intro_segments = video.get_segments_in_window(end_second=intro_end_sec)
            for _ in intro_segments:
                video.delete_segment_at(0)

    def __get_shortest_introduction(self) -> list[Segment]:
        min_intro_segments = None
        for video in self.__summarizer.get_videos():
            # Detecting introduction in video and keeping the smallest one
            intro_end_sec = self.__find_introduction_end_second(video)
            intro_segments = video.get_segments_in_window(end_second=intro_end_sec)

            if min_end_sec is None or intro_end_sec < min_end_sec:
                min_end_sec = intro_end_sec
                min_intro_segments = intro_segments

            # Delete intro segments from vieo
            for _ in intro_segments:
                video.delete_segment_at(0)

        return min_intro_segments

    def __find_introduction_end_second(
        self, video: Video, threshold: float = 0.7
    ) -> int:
        """
        Finds and returns the end second of the video's introduction.
        Introduction's end is detected by calculating the histogram intersection betweeen consecutive frames.
        """
        # Flattened list of frames in video
        all_frames = video.load_frames(self.__summarizer.get_frames_path(), sort=True)

        # Getting current frame and the frame on it's right
        for curr_frame, next_frame in zip(all_frames, all_frames[1:]):
            # Calculating histogram intersection between frames
            hist_intersec = ImageProcessing.compare_histogram_intersection(
                curr_frame, next_frame
            )

            # If matches threshold rule, returns the frame second
            if hist_intersec < threshold:
                return curr_frame.get_video_second()

        return curr_frame.get_video_second()
