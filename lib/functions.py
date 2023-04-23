import math
from typing import List


class Activation:
    def __call__(self, inputs: List[float]):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class Sigmoid(Activation):
    def __call__(self, inputs: List[float]):
        return [1.0 / (1 + math.exp(-x)) for x in inputs]
