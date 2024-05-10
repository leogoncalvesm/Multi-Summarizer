from __future__ import annotations

from processing.utils import log
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
        log("Including introduction in summarized video")
        shortest_intro = self.__get_shortest_introduction()
        if shortest_intro:
            self.__summarizer.append_segments_to_summary(shortest_intro)
        return self.__summarizer

    def exclude(self) -> BaseSummarizer:
        log("Excluding introductions for summarized video")
        self.__remove_introductions()
        return self.__summarizer

    def __remove_introductions(self) -> None:
        for video in self.__summarizer.get_videos():
            intro_end_sec = self.__find_introduction_end_second(video)
            intro_segments = video.get_segments_until(intro_end_sec)
            for _ in intro_segments:
                video.delete_segment_at(0)

    def __get_shortest_introduction(self) -> list[Segment]:
        min_end_sec, min_intro_segments = None, None
        for video in self.__summarizer.get_videos():
            # Detecting introduction in video and keeping the smallest one
            intro_end_sec = self.__find_introduction_end_second(video)
            intro_segments = video.get_segments_until(intro_end_sec)

            if intro_segments and (min_end_sec is None or intro_end_sec < min_end_sec):
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

        curr_frame = all_frames.pop(0)
        hist_curr = ImageProcessing.get_frame_histogram(curr_frame)

        for next_frame in all_frames:
            # Calculating histogram intersection between frames
            hist_next = ImageProcessing.get_frame_histogram(next_frame)
            hist_intersec = ImageProcessing.compare_histograms(hist_curr, hist_next)

            # If matches threshold rule, returns the frame second
            if hist_intersec < threshold:
                return curr_frame.get_video_second()

            curr_frame, hist_curr = next_frame, hist_next

        return curr_frame.get_video_second()
