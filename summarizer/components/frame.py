from re import search
from os.path import basename
from cv2 import imread, IMREAD_GRAYSCALE


class Frame:
    def __init__(self, path: str) -> None:
        self.__path = path
        self.__video_second = self.__process_frame_second()

    def get_video_second(self) -> int:
        return self.__video_second

    def get_path(self) -> str:
        return self.__path

    def __process_frame_second(self) -> int:
        """Gets the correspondeing video second for the frame based on the file name"""
        filename = basename(self.__path)
        return int(search("image-(\d+)\.jpg", filename).groups()[0])

    def load_image(self) -> None:
        """Executes OpenCV imread function and loads to object `image` attribute"""
        return imread(self.__path, IMREAD_GRAYSCALE)
