from abc import ABC
from abc import abstractmethod

import spacy


class BaseTokenMapper(ABC):
    @abstractmethod
    def __call__(self, token: spacy.tokens.Token) -> str:
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
