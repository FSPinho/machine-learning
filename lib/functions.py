import math
from typing import List

import torch

from lib.cloneable import Cloneable
from lib.serializable import Serializable


class Activation(Serializable, Cloneable):
    def __call__(self, inputs: List[float]):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def serialize(self) -> dict:
        return {"type": self.__class__.__name__}

    @staticmethod
    def deserialize(instance_dict: dict):
        _classes = [Sigmoid, ReLU]
        _type = instance_dict.get("type")

        for _class in _classes:
            if _class.__name__ == _type:
                return _class()

        return None


class Sigmoid(Activation):
    def __call__(self, inputs: List[float]):
        return [1.0 / (1 + math.exp(-x)) for x in inputs]


class ReLU(Activation):
    def __call__(self, inputs: List[float]):
        return [max(0.0, x) for x in inputs]


class Normalize(Activation):
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs.shape) > 1:
            return torch.stack([self(i) for i in inputs])

        max_i = 0
        max_ = inputs[max_i]
        for i, v in enumerate(inputs):
            if v > max_:
                max_i = i
                max_ = v

        inputs = [0.0] * len(inputs)
        inputs[max_i] = 1.0
        return torch.tensor(inputs)


class Distribute(Activation):
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(inputs)


class Accuracy:
    def __call__(self, outputs: torch.Tensor, expected_outputs: torch.Tensor):
        n = Normalize()
        return (n(outputs) == n(expected_outputs)).float().sum() / len(outputs)
