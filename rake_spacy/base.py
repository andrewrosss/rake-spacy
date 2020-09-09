# -*- coding: utf-8 -*-
"""Implementation of Rapid Automatic Keyword Extraction algorithm.

As described in the paper `Automatic keyword extraction from individual
documents` by Stuart Rose, Dave Engel, Nick Cramer and Wendy Cowley.
"""
import string
from collections import Counter
from collections import defaultdict
from enum import Enum
from itertools import chain
from itertools import groupby
from itertools import product


class Metric(Enum):
    """Different metrics that can be used for ranking."""

    DEGREE_TO_FREQUENCY_RATIO = 0  # Uses d(w)/f(w) as the metric
    WORD_DEGREE = 1  # Uses d(w) alone as the metric
    WORD_FREQUENCY = 2  # Uses f(w) alone as the metric


class Rake(object):
    """Rapid Automatic Keyword Extraction Algorithm."""

    def __init__(
        self,
        stopwords=None,
        punctuations=None,
        language="english",
        ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO,
        max_length=100000,
        min_length=1,
    ):
        """Constructor.

        :param stopwords: List of Words to be ignored for keyword extraction.
        :param punctuations: Punctuations to be ignored for keyword extraction.
        :param language: Language to be used for stopwords
        :param max_length: Maximum limit on the number of words in a phrase
                           (Inclusive. Defaults to 100000)
        :param min_length: Minimum limit on the number of words in a phrase
                           (Inclusive. Defaults to 1)
        """
        ...

    def extract_keywords_from_text(self, text):
        """Method to extract keywords from the text provided.

        :param text: Text to extract keywords from, provided as a string.
        """
        ...

    def extract_keywords_from_sentences(self, sentences):
        """Method to extract keywords from the list of sentences provided.

        :param sentences: Text to extraxt keywords from, provided as a list
                          of strings, where each string is a sentence.
        """
        ...

    def get_ranked_phrases(self):
        """Method to fetch ranked keyword strings.

        :return: List of strings where each string represents an extracted
                 keyword string.
        """
        ...

    def get_ranked_phrases_with_scores(self):
        """Method to fetch ranked keyword strings along with their scores.

        :return: List of tuples where each tuple is formed of an extracted
                 keyword string and its score. Ex: (5.68, 'Four Scoures')
        """
        ...

    def get_word_frequency_distribution(self):
        """Method to fetch the word frequency distribution in the given text.

        :return: Dictionary (defaultdict) of the format `word -> frequency`.
        """
        ...

    def get_word_degrees(self):
        """Method to fetch the degree of words in the given text. Degree can be
        defined as sum of co-occurances of the word with other words in the
        given text.

        :return: Dictionary (defaultdict) of the format `word -> degree`.
        """
        ...
