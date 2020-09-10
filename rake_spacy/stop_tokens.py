from abc import ABC
from abc import abstractmethod

import spacy


class BaseStopTokenIndicator(ABC):
    @abstractmethod
    def __call__(self, token: spacy.tokens.Token) -> bool:
        """Returns True if the token is considered a stop word/token.

        Args:
            token (spacy.tokens.Token): The token under consideration.

        Returns:
            bool: Wether the token is a stop word.
        """
        pass


class BasicStopTokenIndicator(BaseStopTokenIndicator):
    def __call__(self, token: spacy.tokens.Token) -> bool:
        return (
            token.is_stop or token.is_space or token.is_punct
        ) and not token.like_num
