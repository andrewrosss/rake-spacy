from collections import defaultdict
from typing import Any
from typing import Callable
from typing import DefaultDict

import spacy


class ProxiedDefaultDict(defaultdict):
    """A thin wrapper around collections.defaultdict.

    This class slides in a transformation between objects passed to __getitem__
    and __setitem__, using that transformation to convert the given item to a
    string before using it as a key. This allows for two things:

      1. Basically any object can be used as a key in a ProxiedDefaultDict, and
      2. The user can specify how ProxiedDefaultDict converts a given object to
         a string by providing a callable which ProxiedDefaultDict will use to
         perform the conversions.

    Args:
        proxy (Callable[[Any], str]): This callable will be used to convert
            objects passed to __getitem__ and __setitem__ to a string. The
            resulting string representation will be used as the key in the
            underlying defaultdict. If proxy is None then the builtin str
            function is used. Defaults to None.

    Note:
        Any additional args/kwargs will be passed directly to the defaultdict
        superclass.

    Warning:
        The callable, ``proxy``, should be idempotent, that is, proxy should
        have the following property: ``proxy(x) == proxy(proxy(x))``

    Examples:
        >>> from pprint import pprint
        >>> import datetime
        >>> def isoformat(t: datetime.datetime) -> str:
        ...     return t if isinstance(t, str) else t.isoformat()
        ...
        >>> p = ProxiedDefaultDict(isoformat, int)
        >>> t1 = datetime.datetime(2020, 9, 13, 15, 36, 23, 447108)
        >>> t2 = datetime.datetime(2020, 1, 2, 3, 4, 5)
        >>> p[t1] = 1
        >>> p[t2] += -1
        >>> pprint(p)
        ProxiedDefaultDict(<class 'int'>,
                           {'2020-01-02T03:04:05': -1,
                            '2020-09-13T15:36:23.447108': 1})
        >>> p[t2] += 4
        >>> pprint(p)
        ProxiedDefaultDict(<class 'int'>,
                           {'2020-01-02T03:04:05': 3,
                            '2020-09-13T15:36:23.447108': 1})
        >>> p[t2]
        3
        >>> p.keys()
        dict_keys(['2020-09-13T15:36:23.447108', '2020-01-02T03:04:05'])
    """

    def __init__(self, proxy: Callable[[Any], str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proxy = str if proxy is None else proxy

    def __getitem__(self, k):
        return super().__getitem__(self.proxy(k))

    def __setitem__(self, k, v) -> None:
        super().__setitem__(self.proxy(k), v)


def co_occurange_graph_factory(
    token_mapper: Callable[[spacy.tokens.Token], str]
) -> DefaultDict[str, DefaultDict[str, int]]:
    # adjacency matrix
    return ProxiedDefaultDict(
        token_mapper, lambda: ProxiedDefaultDict(token_mapper, int)
    )
