from abc import ABC
from abc import abstractmethod
from typing import List

import numpy as np


class BaseAggregator(ABC):
    @abstractmethod
    def __call__(self, scores: List[float]) -> float:
        """Reduces a list of numbers to a single number.

        Args:
            scores (List[float]): The numbers over which to perform the reduction.

        Returns:
            float: The result.
        """
        pass


class SumAggregator(BaseAggregator):
    def __call__(self, scores: List[float]) -> float:
        return sum(scores)


class MeanAggregator(BaseAggregator):
    def __call__(self, scores: List[float]) -> float:
        return sum(scores) / len(scores)


class PenalizedNormAggregator(BaseAggregator):
    def __init__(self, max_len_before_penalization: int = 5):
        self.max_len_before_penalization = max_len_before_penalization

    def __call__(self, scores: List[float]) -> float:
        N = np.linalg.norm(scores)
        D = len(
            [s for s in scores if s != 0]
        )  # omit ignoreable words from length penalty
        # if there are 4 or fewer non-stop words, don't penalize
        D = 1 if (1 <= D <= self.max_len_before_penalization) else D
        return float(N / D) if D != 0 else 0
