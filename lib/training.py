import random
from typing import List, Union

from lib.layer import Layer
from lib.network import Network


class Trainer:
    def train(self, network: Network, inputs: List[List[float]], expected_outputs: List[List[float]]):
        raise NotImplementedError

    @staticmethod
    def get_network_error(network: Network, inputs: List[List[float]], expected_outputs: List[List[float]]):
        return DumbTrainer.get_error(network(inputs), expected_outputs)

    @staticmethod
    def get_error(outputs: List[List[float]], expected_outputs: List[List[float]]):
        errors = []
        for _outputs, _expected_outputs in zip(outputs, expected_outputs):
            errors += [pow(_value - _expected_value, 2.0) for _value, _expected_value in
                       zip(_outputs, _expected_outputs)]
        return sum(errors) / len(errors)


class DumbTrainer(Trainer):
    def __init__(self, generations: int, variation_factor: float):
        self._generations = generations
        self._variation_factor = variation_factor

    def train(self, network: Network, inputs: List[List[float]], expected_outputs: List[List[float]]):
        return network

    def _generate_network_variation(self, network: Network):
        network_variation = Network(network.layers)

        for layer in network.layers:
            if isinstance(layer, Layer):
                layer.weights = self._mutate(layer.weights)
                layer.biases = self._mutate(layer.biases)

        return network_variation

    def _mutate(self, values: Union[List, float]):
        if isinstance(values, (int, float)):
            return values + random.uniform(-1.0, 1.0) * self._variation_factor
        return [self._mutate(val) for val in values]
