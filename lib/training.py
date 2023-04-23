import random
from typing import List, Union

from lib.debuggable import Debuggable
from lib.layer import Layer
from lib.network import Network


class Trainer(Debuggable):
    def __init__(self):
        super().__init__()

    def train(self, network: Network, inputs: List[List[float]], expected_outputs: List[List[float]]):
        raise NotImplementedError

    @staticmethod
    def get_network_error(network: Network, inputs: List[List[float]], expected_outputs: List[List[float]]):
        return Trainer.get_error(network(inputs), expected_outputs)

    @staticmethod
    def get_error(outputs: List[List[float]], expected_outputs: List[List[float]]):
        errors = []
        for _outputs, _expected_outputs in zip(outputs, expected_outputs):
            errors += [pow(_value - _expected_value, 2.0) for _value, _expected_value in
                       zip(_outputs, _expected_outputs)]
        return sum(errors) / len(errors)


class DumbTrainer(Trainer):
    def __init__(self, target_error: float, generations: int, generation_size: int, variation_factor: float):
        super().__init__()

        assert target_error > 0
        assert generations > 0
        assert generation_size > 0
        assert variation_factor > 0

        self._target_error = target_error
        self._generations = generations
        self._generation_size = generation_size
        self._variation_factor = variation_factor

    def train(self, network: Network, inputs: List[List[float]], expected_outputs: List[List[float]]):
        self._debug_log(f"Training network... generations={self._generations}")

        best_network = network

        for generation_index in range(self._generations):
            children_networks = [
                best_network,
                *[self._generate_network_variation(best_network) for _ in range(self._generation_size - 1)]
            ]
            children_networks_and_errors = [
                (self.get_network_error(network, inputs, expected_outputs), network) for network in children_networks
            ]
            children_networks_and_errors = list(sorted(children_networks_and_errors, key=lambda ne: ne[0]))
            error, best_network = children_networks_and_errors[0]
            errors = [error for error, _ in children_networks_and_errors]

            self._debug_log(
                f"Generation index={generation_index} error={error} " +
                f"children_errors={self._prepare_value_to_log(errors)}",
                end="\r",
                flush=True
            )

            if error <= self._target_error:
                break

        self._debug_log("")
        self._debug_log("Training finished")

        return best_network

    def _generate_network_variation(self, network: Network):
        layers = []

        for layer in network.layers:
            if isinstance(layer, Layer):
                layer.weights = self._mutate(layer.weights)
                layer.biases = self._mutate(layer.biases)
            layers.append(layer)

        return Network(*layers)

    def _mutate(self, values: Union[List, float]):
        if isinstance(values, (int, float)):
            return values + random.uniform(-1.0, 1.0) * self._variation_factor
        return [self._mutate(val) for val in values]
