from summarizer.components.frame import Frame
from cv2 import (
    CascadeClassifier,
    calcHist,
    normalize,
    compareHist,
    NORM_L1,
    HISTCMP_INTERSECT,
)


class FaceDetector:
    def __init__(self, classifier_path: str) -> None:
        self.__face_classifier = CascadeClassifier(classifier_path)

    def frame_contains_face(self, frame: Frame) -> bool:
        faces = self.__face_classifier.detectMultiScale(frame.load_image(), 1.1, 5)
        return bool(len(faces))


class ImageProcessing:
    @staticmethod
    def compare_histogram_intersection(frame_1: Frame, frame_2: Frame) -> float:
        """Calculates histograms for the two frames passed and returns the histogram intersection between the frames"""
        # Calculating and normalizing histogram for frame 1
        histogram_1 = calcHist([frame_1.load_image()], [0], None, [256], [0, 256])
        normalize(histogram_1, histogram_1, norm_type=NORM_L1)

        # Calculating and normalizing histogram for frame 2
        histogram_2 = calcHist([frame_2.load_image()], [0], None, [256], [0, 256])
        normalize(histogram_2, histogram_2, norm_type=NORM_L1)

        return compareHist(histogram_1, histogram_2, HISTCMP_INTERSECT)
