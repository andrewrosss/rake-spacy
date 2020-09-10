import collections
import itertools
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Tuple

import spacy

from . import aggregators
from . import cog
from . import mappers
from . import phrasers
from . import scorers
from . import stop_tokens


class Rake(object):
    """Rapid Automatic Keyword Extraction Algorithm."""

    def __init__(
        self,
        min_length=1,
        max_length=100_000,
        nlp=None,
        stop_token_class: stop_tokens.BaseStopTokenIndicator = None,
        phraser_class: phrasers.BasePhraser = None,
        token_mapper_class: mappers.BaseTokenMapper = None,
        word_scorer_class: scorers.BaseScorer = None,
        aggregator_class: aggregators.BaseAggregator = None,
        # stopwords=None,
        # punctuations=None,
        # language="english",
        # ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.nlp = spacy.load("en_core_web_sm") if nlp is None else nlp
        self.stop_token_class = (
            stop_tokens.BasicStopTokenIndicator()
            if stop_token_class is None
            else stop_token_class
        )
        self.phraser_class = (
            phrasers.ContiguousNonStopTokenPhraser(self.stop_token_class)
            if phraser_class is None
            else phraser_class
        )
        self.token_mapper_class = (
            mappers.LemmaMapper() if token_mapper_class is None else token_mapper_class
        )
        self.word_scorer_class = (
            scorers.FrequencyScorer()
            if word_scorer_class is None
            else word_scorer_class
        )
        self.aggregator_class = (
            aggregators.SumAggregator()
            if aggregator_class is None
            else aggregator_class
        )

    def extract_keywords_from_text(
        self, text: str
    ) -> List[Tuple[float, spacy.tokens.Span]]:
        """Method to extract keywords from the text provided.

        :param text: Text to extract keywords from, provided as a string.
        """
        doc = self.nlp(text)
        phrases = self.phraser_class(doc)
        self.co_occurance_graph = self.generate_word_co_occurance_graph(phrases)
        self.frequency_dist = self.generate_frequency_dist(phrases)
        self.degree = self.generate_degree(phrases)
        return self.generate_ranklist(phrases)

    # def extract_keywords_from_sentences(self, sentences):
    #     ...

    # def get_ranked_phrases(self):
    #     ...

    # def get_ranked_phrases_with_scores(self):
    #     ...

    # def get_word_frequency_distribution(self):
    #     ...

    # def get_word_degrees(self):
    #     ...

    def generate_word_co_occurance_graph(
        self, phrases: List[spacy.tokens.Span]
    ) -> DefaultDict[str, DefaultDict[str, int]]:
        cograph = cog.co_occurange_graph_factory(self.token_mapper_class)
        for phrase in phrases:
            for word, coword in itertools.product(phrase, phrase):
                if not (self.stop_token_class(word) or self.stop_token_class(coword)):
                    cograph[word][coword] += 1
        return cograph

    def generate_frequency_dist(
        self, phrases: List[spacy.tokens.Span]
    ) -> Dict[str, int]:
        # recall self.stop_token_class returns True if the token is a "stop token"
        words = [t for t in itertools.chain(*phrases) if not self.stop_token_class(t)]
        word_strings = [self.token_mapper_class(t) for t in words]
        fd = collections.Counter(word_strings)
        return fd

    def generate_degree(
        self, phrases: List[spacy.tokens.Span]
    ) -> DefaultDict[str, int]:
        cograph = self.co_occurance_graph
        degree = collections.defaultdict(int)
        for word in cograph:
            degree[word] = sum(cograph[word].values())
        return degree

    def generate_ranklist(
        self, phrases: List[spacy.tokens.Span]
    ) -> List[Tuple[float, spacy.tokens.Span]]:
        cograph = self.co_occurance_graph
        scores = [([self.word_scorer_class(cograph, t) for t in p], p) for p in phrases]
        rank_list = [
            (self.aggregator_class(word_scores), p) for word_scores, p in scores
        ]
        self.rank_list = sorted(rank_list, key=lambda t: (-t[0], t[1].text))
        self.ranked_phrases = [p for _, p in self.rank_list]
        return self.rank_list
