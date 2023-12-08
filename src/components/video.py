from os import listdir
from os.path import join
from moviepy.editor import VideoFileClip, vfx, concatenate_videoclips

from components.frame import Frame
from components.segment import Segment


class Video:
    def __init__(
        self,
        name: str = "",
        path: str = "",
        segments: list[Segment] = [],
        assign_to_segments: bool = True,
    ) -> None:
        self.__name = name
        self.__path = path
        self.__segments = segments

        if assign_to_segments:
            self.__assign_to_segments()

    def __assign_to_segments(self) -> None:
        for segment in self.__segments:
            segment.set_video(self)

    def append_segment(self, new_segment: Segment) -> None:
        self.__segments.append(new_segment)

    def save(self, fadein: float = 0.5, fadeout: float = 0.5) -> None:
        clips_list = []
        for segment in self.__segments:
            clip = VideoFileClip(segment.get_video().get_video_path()).subclip(
                segment.get_begin(), segment.get_end()
            )
            clip = vfx.fadein(clip, fadein)
            clip = vfx.fadeout(clip, fadeout)
            clips_list.append(clip)

        final_clip = concatenate_videoclips(clips_list)
        final_clip.write_videofile(self.__path)

    def get_name(self) -> str:
        return self.__name

    def get_video_path(self) -> str:
        return self.__path

    def get_segment(self, seg_index: int) -> Segment:
        return self.__segments[seg_index]

    def get_segments(self) -> list[Segment]:
        return self.__segments

    def get_content(self, separator: str = " ") -> str:
        return separator.join(seg.get_content() for seg in self.__segments)

    def delete_segment(self, segment_index: int) -> None:
        del self.__segments[segment_index]

    def load_frames(self, frames_path: str, sort=False) -> list[Frame]:
        frames = [
            Frame(join(frames_path, self.__name, frame_path))
            for frame_path in listdir(join(frames_path, self.__name))
        ]
        if sort:
            frames.sort(key=lambda frame: frame.get_video_second())

        return frames
