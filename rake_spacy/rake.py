# stdlib
import enum
import itertools
import collections
from typing import DefaultDict, Dict, List, Tuple

# 3rd party
import spacy

# local
# -----


class Metric(enum.Enum):
    """Different metrics that can be used for ranking."""

    DEGREE_TO_FREQUENCY_RATIO = 0  # uses d(w)/f(w) as the metric
    WORD_DEGREE = 1  # uses d(w) as the metric
    WORD_FREQUENCY = 2  # uses f(w)


class Rake:
    """Rapid Automatic Keyword Extraction Algorithm."""

    def __init__(
        self,
        nlp=None,
        ranking_metric: Metric = Metric.DEGREE_TO_FREQUENCY_RATIO,
        max_length: int = 100_000,
        min_length: int = 1,
    ) -> None:
        self.ranking_metric = ranking_metric
        self.max_length = max_length
        self.min_length = min_length
        self.nlp = nlp

        # stuff to be extracted from the provided text
        self.frequency_dist = None
        self.degree = None
        self.co_occurance_graph = None
        self.rank_list = None
        self.ranked_phrases = None

    def extract_keywords_from_text(
        self, text: str
    ) -> List[Tuple[float, spacy.tokens.Span]]:
        self.nlp = spacy.load("en_core_web_sm") if self.nlp is None else self.nlp
        doc = self.nlp(text)
        phrases = self.generate_phrases(doc)
        self.generate_frequency_dist(phrases)
        self.generate_degree(phrases)
        return self.generate_ranklist(phrases)

    def generate_phrases(self, doc: spacy.tokens.Doc) -> List[spacy.tokens.Span]:
        groups = [
            list(g)
            for k, g in itertools.groupby(doc, lambda t: not self.to_ignore(t))
            if k
        ]
        return [doc[g[0].i : g[-1].i + 1] for g in groups]

    @staticmethod
    def to_ignore(t: spacy.tokens.Token) -> bool:
        return (t.is_stop or t.is_space or t.is_punct) and not t.like_num

    def generate_frequency_dist(
        self, phrases: List[spacy.tokens.Span]
    ) -> Dict[str, int]:
        fd = collections.Counter(t.text for t in itertools.chain(*phrases))
        self.frequency_dist = fd
        return fd

    def generate_degree(self, phrases: List[spacy.tokens.Span]) -> Dict[str, int]:
        cog = self.generate_word_co_occurance_graph(phrases)
        degree = collections.defaultdict(int)
        for word in cog:
            degree[word] = sum(cog[word].values())
        self.degree = degree
        return degree

    def generate_word_co_occurance_graph(
        self, phrases: List[spacy.tokens.Span]
    ) -> Dict[str, DefaultDict[str, int]]:
        cog = collections.defaultdict(lambda: collections.defaultdict(int))
        for phrase in phrases:
            for word, coword in itertools.product(phrase, phrase):
                cog[word.text][coword.text] += 1
        self.co_occurance_graph = cog
        return cog

    def generate_ranklist(
        self, phrases: List[spacy.tokens.Span]
    ) -> List[Tuple[float, spacy.tokens.Span]]:
        rank_list = [(sum(self.word_score(t.text) for t in p), p) for p in phrases]
        self.rank_list = sorted(rank_list, reverse=True)
        self.ranked_phrases = [p for _, p in self.rank_list]
        return self.rank_list

    def word_score(self, word: str) -> float:
        if self.ranking_metric == Metric.DEGREE_TO_FREQUENCY_RATIO:
            score = self.degree[word] / self.frequency_dist[word]
        elif self.ranking_metric == Metric.WORD_DEGREE:
            score = self.degree[word]
        else:
            score = self.frequency_dist[word]
        return 1.0 * score
