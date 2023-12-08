from __future__ import annotations

import nltk
from re import sub
from math import log10
from typing import Any
from itertools import chain
from abc import ABC, abstractmethod
from pandas import read_pickle, DataFrame, MultiIndex
from sklearn.feature_extraction.text import TfidfVectorizer
from google.cloud.language_v1.types import AnalyzeSentimentResponse


class BagOfWordsProcessing:
    def __init__(self, text: str) -> None:
        self.__text = text
        self.__stopwords = set()
        self.__stemmer = None

    def __load_nltk_stopwords(self, language="portuguese") -> None:
        self.__stopwords = nltk.corpus.stopwords.words(language)

    def __load_nltk_stemmer(self) -> None:
        self.__stemmer = nltk.stem.RSLPStemmer()

    def __replace_pattern(self, pattern: str = "", replace: str = "") -> None:
        self.__text = sub(pattern, replace, self.__text)

    def __remove_stopwords(self) -> None:
        self.__load_nltk_stopwords()
        self.__text = " ".join(
            [
                word
                for word in self.__text.split()
                if word.lower() not in self.__stopwords
            ]
        )

    def __stemming(self) -> None:
        self.__load_nltk_stemmer()
        self.__text = " ".join(
            [self.__stemmer.stem(word) for word in self.__text.split()]
        )

    def base_text_processing(self) -> str:
        self.__replace_pattern("[-./?!,\":;()']")  # Remove ponctuation
        self.__replace_pattern("[0-9]")  # Remove numbers
        self.__remove_stopwords()
        self.__stemming()

        return self.__text


class BagOfWords:
    def __init__(self, items: dict[Any, str], language: str = "portuguese") -> None:
        self.__items = items
        self.__word_list = []
        self.__bow_df = None

    def get_bag(self) -> str:
        return self.__items

    def __calc_words_list(self) -> None:
        full_text = " ".join(chain(self.__items.values()))
        self.__word_list = set(full_text.split())

    def items_preprocessing(self) -> BagOfWords:
        for key, text in self.__items.items():
            self.__items[key] = BagOfWordsProcessing(text).base_text_processing()
        return self

    def generate_bow_dataframe(self, index_names: list) -> BagOfWords:
        # Generating list of words among all sentences
        self.__calc_words_list()

        # Creating initial DataFrame with the processed sentences for every (video, segment)
        self.__bow_df = DataFrame.from_dict(
            self.__items, orient="index", columns=["sentence"]
        )

        # Calculating TF-IDF weight
        n_sentences = len(self.__bow_df)
        for w in self.__word_list:
            term_freq = self.__bow_df.sentence.str.count(w)

            # Calculating IDF
            doc_freq = term_freq.gt(0).sum()
            term_idf = log10(n_sentences / doc_freq)

            # Applying TF-IDF
            self.__bow_df[w] = term_freq * term_idf

        self.__bow_df.drop(columns=["sentence"], inplace=True)

        # Normalizing values
        sentences_magnitude = (
            self.__bow_df.apply(pow, exp=2).sum(axis=1) ** 0.5
        ).replace(
            0, 1
        )  # Replacing 0 with 1 to avoid 0 division below

        self.__bow_df = self.__bow_df.div(sentences_magnitude, axis=0)

        # Reordering columns in alphabetical order
        self.__bow_df = self.__bow_df.reindex(sorted(self.__bow_df.columns), axis=1)

        # Changing tuple index to multi-index
        self.__bow_df.index = MultiIndex.from_tuples(
            self.__bow_df.index, names=index_names
        )

        return self.__bow_df

    def generate_bow_dataframe_tfidfvectorizer(self, index_names: list) -> BagOfWords:
        vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False)

        sentences = list(chain(self.__items.values()))
        tfidf_data = vectorizer.fit_transform(sentences)

        index_df = DataFrame(self.__items.keys(), columns=index_names)
        tfidf_df = DataFrame(
            tfidf_data.toarray(), columns=vectorizer.get_feature_names_out()
        )

        self.__bow_df = index_df.join(tfidf_df).set_index(index_names)
        return self


class SubjectivityClassificator(ABC):
    @abstractmethod
    def is_subjective(self, text: str) -> bool:
        pass


class SubjectivityGoogleAPI(SubjectivityClassificator):
    def __init__(self, sentiment_data_path: str, sentilex_path: str) -> None:
        self.__sentiment_df = read_pickle(sentiment_data_path)
        self.__adjectives = self.__load_adjectives(sentilex_path)

    def is_subjective(self, text: str) -> bool:
        sentiment_data = self.__load_text_sentiment(text)
        if sentiment_data is None:
            raise Exception(f"No sentiment data for text: {text}")

        text_tokens = sub("[-./?!,\":;()']", " ", text).lower().split()

        word_count = len(text_tokens)
        adjectives_count = sum(word in self.__adjectives for word in text_tokens)

        return self.__calculate_subjectivity(
            word_count, adjectives_count, sentiment_data
        )

    def __calculate_subjectivity(
        self, words: int, adjectives: int, sentiment: AnalyzeSentimentResponse
    ) -> bool:
        if words < 0:
            raise Exception("Negative amount of tokens in text")

        magnitude = sentiment.document_sentiment.magnitude

        # Small segments
        if words in range(36):
            return magnitude > 0.6 or adjectives >= 3
        # Medium segments
        if words in range(36, 71):
            return magnitude > 1.2 or adjectives >= 4

        return adjectives >= 4 or (
            sum((abs(s.sentiment.score) > 0.3) for s in sentiment.sentences)
            >= len(sentiment.sentences) * 0.4
        )

    def __load_text_sentiment(self, text: str) -> AnalyzeSentimentResponse:
        """
        Retorna os resultados de sentimentos da Google Natural Language API previamente salvos.
        Estes dados de sentimentos resultantes da API foram coletados e salvos em Outubro/2021.
        """
        df_filtered = self.__sentiment_df[self.__sentiment_df.content == text]
        if df_filtered.empty:
            return None
        return df_filtered.to_dict("records")[0].get("sentiment_response")

    def __load_adjectives(self, sentilex_path: str) -> set[str]:
        with open(sentilex_path) as f:
            adjectives = {line.split(",")[0] for line in f if "PoS=Adj" in line}
        return adjectives
