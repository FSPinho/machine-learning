from copy import deepcopy
from typing import List

from lib.cloneable import Cloneable
from lib.debuggable import Debuggable
from lib.serializable import Serializable


class Layer(Debuggable, Serializable, Cloneable):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self._input_size = input_size
        self._output_size = output_size

        self._biases = [0.0] * output_size
        self._weights = [[0.0] * output_size] * input_size

        self._validate_biases()
        self._validate_weights()

    def __call__(self, inputs: List[float]) -> List[float]:
        self._validate_inputs(inputs)
        return self._calculate_outputs(inputs)

    def __str__(self):
        return f"Layer {self._input_size}x{self._output_size}"

    @property
    def biases(self):
        return list(self._biases)

    @property
    def weights(self):
        return [list(row) for row in self._weights]

    def clone(self, **kwargs):
        clone = Layer(self._input_size, self._output_size)
        clone._biases = self.biases
        clone._weights = self.weights

        for key, val in kwargs.items():
            setattr(clone, key, val)

        return clone

    def _calculate_outputs(self, inputs: List[float]) -> List[float]:
        outputs = []

        for output_index in range(self._output_size):
            bias = self._biases[output_index]
            output_value = 0.0

            for input_index in range(self._input_size):
                weight = self._weights[input_index][output_index]
                output_value += weight * inputs[input_index]

            output_value += bias
            outputs.append(output_value)

        return outputs

    def _validate_inputs(self, inputs: List[float]):
        assert len(inputs) == self._input_size, "Given inputs must match the layer input size"

    def _validate_biases(self):
        assert len(self._biases) == self._output_size, "Given biases must match the layer output size"

    def _validate_weights(self):
        assert len(self._weights) == self._input_size, "Given weights must match the layer input size"

        for weights_row in self._weights:
            assert len(weights_row) == self._output_size, "Given weights row must match the layer output size"

    def serialize(self) -> dict:
        return dict(
            type=Layer.__name__,
            input_size=self._input_size,
            output_size=self._output_size,
            biases=self._biases,
            weights=self._weights,
        )

    @staticmethod
    def deserialize(instance_dict: dict):
        if instance_dict.get("type") == Layer.__name__:
            layer = Layer(0, 0)
            return layer.clone(
                _input_size=instance_dict.get("input_size"),
                _output_size=instance_dict.get("output_size"),
                _biases=instance_dict.get("biases"),
                _weights=instance_dict.get("weights"),
            )
        return None
