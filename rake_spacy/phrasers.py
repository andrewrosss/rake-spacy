from __future__ import annotations

import itertools
from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import List

import spacy

from .stop_tokens import BasicStopTokenIndicator


class BasePhraser(ABC):
    @abstractmethod
    def __call__(self, doc: spacy.tokens.Doc) -> List[spacy.tokens.Span]:
        """Produces a list of spans. These spans will be used to construct the
        co-occurance graph.

        Args:
            doc (spacy.tokens.Doc): The parsed spacy Doc object.

        Returns:
            List[spacy.tokens.Span]: A list of candidate phrases.
        """
        pass


class EntityAndNounChunkPhraser(BasePhraser):
    def __call__(self, doc: spacy.tokens.Doc) -> List[spacy.tokens.Span]:
        """Produces a list of spans where each span is either an named entity, a
        noun chunk, or if an entity and noun-chunk intersect then the "larger"
        of the two.

        Args:
            doc (spacy.tokens.Doc): The document from which to generate the phrases.

        Returns:
            List[spacy.tokens.Span]: A list of candidate phrases.
        """
        return spacy.util.filter_spans(list(doc.ents) + list(doc.noun_chunks))


class ContiguousNonStopTokenPhraser(BasePhraser):
    def __init__(
        self, stop_token_indicator_fn: Callable[[spacy.tokens.Token], bool] = None
    ):
        self.stop_token_indicator_fn = (
            BasicStopTokenIndicator()
            if stop_token_indicator_fn is None
            else stop_token_indicator_fn
        )

    def __call__(self, doc: spacy.tokens.Doc) -> List[spacy.tokens.Span]:
        """Produces a list of spans where each span is a list of contiguous
        non-stopwords.

        Note:
            This is (essentially) the phrase-generation technique proposed
            in the original RAKE paper (DOI: 10.1002/9780470689646.ch1)

        Args:
            doc (spacy.tokens.Doc): The document from which to generate the
            phrases.

        Returns:
            List[spacy.tokens.Span]: A list of candidate phrases.
        """
        groups = [
            list(g)
            for k, g in itertools.groupby(
                doc, lambda t: not self.stop_token_indicator_fn(t)
            )
            if k
        ]
        return [doc[g[0].i : g[-1].i + 1] for g in groups]  # noqa: E203
