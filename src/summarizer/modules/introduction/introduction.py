from __future__ import annotations

from summarizer.components.video import Video
from summarizer.components.segment import Segment
from summarizer.modules.module import SelectionCriteria
from summarizer.processing.image import ImageProcessing

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from summarizer.components.summarizer import HSMVideoSumm


class Introduction(SelectionCriteria):
    def __init__(self, summarizer: HSMVideoSumm) -> None:
        self.__summarizer = summarizer

    def include(self) -> HSMVideoSumm:
        shortest_intro = self.__get_shortest_introduction()
        if shortest_intro:
            self.__summarizer.summary_video.append_segment(shortest_intro)
        return self.__summarizer

    def exclude(self) -> HSMVideoSumm:
        self.__remove_introductions()
        return self.__summarizer

    def __remove_introductions(self) -> None:
        for video in self.__summarizer.videos:
            intro = self.__find_introduction(video)
            self.__delete_introduction(video, intro)

    def __get_shortest_introduction(self) -> Segment:
        min_intro = None
        for video in self.__summarizer.videos:
            # Detecting introduction in video and keeping the smallest one
            video_intro = self.__find_introduction(video)

            min_intro = (
                video_intro if min_intro is None else min(min_intro, video_intro)
            )

            # Delete intro segments from vieo
            self.__delete_introduction(video, video_intro)

        return min_intro

    def __find_introduction(self, video: Video, threshold: float = 0.7) -> Segment:
        """
        Creates a segment with the frames found as composing the introduction.
        Introduction's end is detected by calculating the histogram intersection betweeen consecutive frames.
        """
        # Flattened list of frames in video
        all_frames = video.load_frames(self.__summarizer.frames_path, sort=True)

        # Creating an empty segment
        intro_segment = Segment(0, 0)
        intro_segment.set_video(video)

        # Getting current frame and the frame on it's right
        for curr_frame, next_frame in zip(all_frames, all_frames[1:]):
            intro_segment.set_end(intro_segment.get_end() + 1)

            # Calculating histogram intersection between frames
            hist_intersec = ImageProcessing.compare_histogram_intersection(
                curr_frame, next_frame
            )

            # If matches threshold rule, returns Segment
            if hist_intersec < threshold:
                return intro_segment

        return intro_segment

    def __delete_introduction(self, video: Video, intro_segment: Segment) -> None:
        """
        Gets final frame from detected introduction and removes all segments
        where introduction's final second is bigger than or equal to the
        segment's final second
        """
        while intro_segment.get_end() >= video.get_segments()[0].get_end():
            video.delete_segment(0)
