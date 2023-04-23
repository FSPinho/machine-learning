import math
from typing import List

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
        _classes = [Sigmoid]
        _type = instance_dict.get("type")

        for _class in _classes:
            if _class.__name__ == _type:
                return _class()

        return None


class Sigmoid(Activation):
    def __call__(self, inputs: List[float]):
        return [1.0 / (1 + math.exp(-x)) for x in inputs]
