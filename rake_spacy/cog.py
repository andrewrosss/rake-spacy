from collections import defaultdict
from typing import Callable

import spacy


class WrappedDefaultDict(defaultdict):
    def __init__(
        self, token_mapper: Callable[[spacy.tokens.Token], str], *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.token_mapper = token_mapper

    def __getitem__(self, k):
        if isinstance(k, str):
            return super().__getitem__(k)
        return super().__getitem__(self.token_mapper(k))

    def __setitem__(self, k, v) -> None:
        if isinstance(k, str):
            super().__setitem__(k, v)
            return
        super().__setitem__(self.token_mapper(k), v)


def co_occurange_graph_factory(token_mapper: Callable[[spacy.tokens.Token], str]):
    # adjacency matrix
    return WrappedDefaultDict(
        token_mapper, lambda: WrappedDefaultDict(token_mapper, int)
    )
