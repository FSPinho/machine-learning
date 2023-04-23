from copy import deepcopy
from typing import List


class Layer:
    def __init__(self, input_size: int, output_size: int):
        self._input_size = input_size
        self._output_size = output_size

        self._weights = [[0.0] * output_size] * input_size
        self._biases = [0.0] * output_size

    def __call__(self, inputs: List[float]) -> List[float]:
        self._validate_inputs(inputs)
        return self._calculate_outputs(inputs)

    def __str__(self):
        return f"Layer {self._input_size}x{self._output_size} biases={self._biases} weight={self._weights}"

    @property
    def weights(self):
        return deepcopy(self._weights)

    @weights.setter
    def weights(self, weights: List[List[float]]):
        self._validate_weights(weights)
        self._weights = weights

    @property
    def biases(self):
        return deepcopy(self._biases)

    @biases.setter
    def biases(self, biases: List[float]):
        self._validate_biases(biases)
        self._biases = biases

    def clone(self):
        clone = Layer(self._input_size, self._output_size)
        clone.weights = self._weights
        clone.biases = self._biases
        return clone

    def _validate_inputs(self, inputs: List[float]):
        assert len(inputs) == self._input_size, "Given inputs must match the layer input size"

    def _validate_weights(self, weights: List[List[float]]):
        assert len(weights) == self._input_size, "Given weights must match the layer input size"

        for weights_row in weights:
            assert len(weights_row) == self._input_size, "Given weights row must match the layer output size"

    def _validate_biases(self, biases: List[float]):
        assert len(biases) == self._output_size, "Given biases must match the layer output size"

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
