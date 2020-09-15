from abc import ABC
from abc import abstractmethod

import spacy


class BaseTokenMapper(ABC):
    @abstractmethod
    def __call__(self, token: spacy.tokens.Token) -> str:
        """Returns a string representation of a spacy Token.

        Args:
            token (spacy.tokens.Token): The token to coerce to a string.

        Returns:
            str: The resulting string representation.
        """
        pass


class LemmaMapper(BaseTokenMapper):
    def __call__(self, token: spacy.tokens.Token) -> str:
        return token if isinstance(token, str) else token.lemma_


class LemmaLowerMapper(BaseTokenMapper):
    def __call__(self, token: spacy.tokens.Token) -> str:
        return token if isinstance(token, str) else token.lemma_.lower()


class TextMapper(BaseTokenMapper):
    def __call__(self, token: spacy.tokens.Token) -> str:
        return token if isinstance(token, str) else token.text
