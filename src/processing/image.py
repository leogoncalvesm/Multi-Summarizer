from __future__ import annotations

from math import log10
from typing import Any
from pandas import DataFrame
from collections import Counter
from sklearn.cluster import KMeans
from dataclasses import dataclass, field
from numpy import ndarray, dot, zeros, argsort, transpose, concatenate
from cv2 import (
    NORM_L1,
    calcHist,
    normalize,
    xfeatures2d,
    compareHist,
    CascadeClassifier,
    HISTCMP_INTERSECT,
)

from components.frame import Frame
from components.segment import Segment


class FaceDetector:
    def __init__(self, classifier_path: str) -> None:
        self.__face_classifier = CascadeClassifier(classifier_path)

    def frame_contains_face(self, frame: Frame) -> bool:
        faces = self.__face_classifier.detectMultiScale(frame.load_image(), 1.1, 5)
        return bool(len(faces))


class BagOfVisualWords:
    def __init__(self, items: dict[Any, ndarray], dict_size: int) -> None:
        self.__items = items
        self.__dict_size = dict_size
        self.__kmeans = None
        self.__bovw_df = None

    def fit_kmeans(self, **kwargs) -> None:
        self.__kmeans = KMeans(n_clusters=self.__dict_size, **kwargs)
        self.__kmeans.fit(concatenate(list(self.__items.values())))

    def generate_bovw_dataframe(self) -> DataFrame:
        self.__bovw_df = DataFrame(
            self.__items.items(), columns=["segment", "features"]
        )

        tfs = self.__bovw_df.features.apply(self.__kmeans.predict).apply(Counter)
        doc_freq = Counter(key for a in tfs for key in a.keys())

        for key in doc_freq.keys():
            term_freq = tfs.apply(lambda x: x.get(key))
            term_idf = log10(self.__dict_size / doc_freq.get(key))

            self.__bovw_df[key] = term_freq * term_idf

        self.__bovw_df.drop(columns=["features"], inplace=True)
        self.__bovw_df.set_index("segment", drop=True, inplace=True)

        return self.__bovw_df


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

    @staticmethod
    def ks_sift(segment: Segment, frames_path: str):
        segment_keyframes = []
        for frame in segment.load_frames(frames_path)[1:-1]:
            _, descriptor = xfeatures2d.SIFT_create().detectAndCompute(
                frame.load_image(), None
            )

            if descriptor is None:
                continue

            keyframe = Keyframe(descriptor=descriptor)
            if keyframe.is_keyframe(segment_keyframes):
                segment_keyframes.append(keyframe)
        return concatenate([kf.descriptor for kf in segment_keyframes])


@dataclass
class Keyframe:
    descriptor: ndarray
    descriptor_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.descriptor_size = len(self.descriptor)

    def num_matches(self, other: Keyframe, threshold: float = 0.95) -> int:
        num_match = 0
        d1_t, d2_t = map(transpose, (self.descriptor, other.descriptor))

        for i, desc in enumerate(self.descriptor):
            sim = dot(desc, d2_t)
            self_match = argsort(-sim)[0]

            if sim[self_match] >= threshold:
                match_feature = other.descriptor[self_match]
                sim_check = dot(match_feature, d1_t)
                other_match = argsort(-sim_check)[0]

                num_match += (sim_check[other_match] >= threshold) and (
                    other_match == i
                )

        return num_match

    def is_keyframe(
        self,
        keyframes: list[Keyframe],
        min_keypoints_diff_ratio: float = 0.6,
        min_descriptors_diff_ratio: float = 0.1,
    ) -> bool:
        if not keyframes:
            return True

        return sum(
            (
                abs(self.descriptor_size - kf.descriptor_size)
                >= kf.descriptor_size * min_keypoints_diff_ratio
            )
            or (
                self.num_matches(kf) < (min_descriptors_diff_ratio * kf.descriptor_size)
            )
            for kf in keyframes
        ) == len(keyframes)
