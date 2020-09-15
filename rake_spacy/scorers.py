from abc import ABC
from abc import abstractmethod

import spacy


class BaseScorer(ABC):
    @abstractmethod
    def __call__(self, cog, token: spacy.tokens.Token) -> float:
        """Extracts a numeric score for a token from a co-oocurange graph.

        Args:
            cog (CoOccuranceGraph): The co-occurance graph.
            token (spacy.tokens.Token): The token to score.

        Returns:
            float: The token's score extracted from the co-occurance graph.
        """
        pass


class DegreeScorer(BaseScorer):
    def __call__(self, cog, token: spacy.tokens.Token) -> float:
        return float(sum(cog[token].values()))


class FrequencyScorer(BaseScorer):
    def __call__(self, cog, token: spacy.tokens.Token) -> float:
        return float(cog[token][token])


class DegreeToFrequencyScorer(BaseScorer):
    def __call__(self, cog, token: spacy.tokens.Token) -> float:
        freq = cog[token][token]
        deg = sum(cog[token].values())
        return float(0 if freq == 0 else deg / freq)


class LocationPenalizedFrequencyScorer(BaseScorer):
    def __call__(self, cog, token: spacy.tokens.Token) -> float:
        freq = cog[token][token]
        location_penalty = (len(token.doc) - token.i) / len(token.doc)
        return 1.0 * freq * location_penalty
