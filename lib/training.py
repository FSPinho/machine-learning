import random
from typing import List, Union

from lib.debuggable import Debuggable
from lib.layer import Layer
from lib.model import Model


class Trainer(Debuggable):
    def __init__(self):
        super().__init__()

    def train(self, model: Model, inputs: List[List[float]], expected_outputs: List[List[float]]):
        raise NotImplementedError

    @staticmethod
    def get_model_error(model: Model, inputs: List[List[float]], expected_outputs: List[List[float]]):
        return Trainer.get_error(model(inputs), expected_outputs)

    @staticmethod
    def get_error(outputs: List[List[float]], expected_outputs: List[List[float]]):
        errors = []
        for _outputs, _expected_outputs in zip(outputs, expected_outputs):
            errors += [pow(_value - _expected_value, 2.0) for _value, _expected_value in
                       zip(_outputs, _expected_outputs)]
        return sum(errors) / len(errors)


class DumbTrainer(Trainer):
    def __init__(self, target_error: float, generations: int, generation_size: int, variation_factors: List[float]):
        super().__init__()

        assert target_error > 0
        assert generations > 0
        assert generation_size > 0
        assert len(variation_factors) > 0

        self._target_error = target_error
        self._generations = generations
        self._generation_size = generation_size
        self._variation_factors = variation_factors

    def train(self, model: Model, inputs: List[List[float]], expected_outputs: List[List[float]]):
        self._debug_log(
            f"Training model... generations={self._generations} " +
            f"error={self._prepare_value_to_log(self.get_model_error(model, inputs, expected_outputs))}"
        )

        best_models = [model]

        for generation_index in range(self._generations):
            variations = self._generate_model_variations(best_models)

            variations_and_errors = [
                (self.get_model_error(variation, inputs, expected_outputs), variation)
                for variation in variations
            ]
            variations_and_errors = list(sorted(variations_and_errors, key=lambda ne: ne[0]))
            variations_and_errors = variations_and_errors[:2]  # + variations_and_errors[-1:]

            best_models = [variation for _, variation in variations_and_errors]
            error, _ = variations_and_errors[0]
            errors = [error for error, _ in variations_and_errors]

            self._debug_log(
                f"Generation index={generation_index}/{self._generations} " +
                f"variations={len(variations)} " +
                f"errors={self._prepare_value_to_log(errors)}",
                end="\r",
                flush=True
            )

            if error <= self._target_error:
                break

        self._debug_log("")
        self._debug_log("Training finished")

        return best_models[0]

    def _generate_model_variations(self, models: List[Model]):
        variations = [*models]
        model_index = 0
        factor_index = 0
        while len(variations) < self._generation_size:
            model = models[model_index]
            factor = self._variation_factors[factor_index]

            variation = self._generate_model_variation(model, factor)
            variations.append(variation)

            model_index = (model_index + 1) % len(models)
            factor_index = (factor_index + 1) % len(self._variation_factors)
        return variations

    def _generate_model_variation(self, model: Model, factor: float):
        layers = []

        for layer in model.layers:
            if isinstance(layer, Layer):
                clone = layer.clone(
                    _biases=self._mutate(layer.biases, factor),
                    _weights=self._mutate(layer.weights, factor)
                )
            else:
                clone = layer.clone()

            layers.append(clone)

        return model.clone(_layers=layers)

    def _mutate(self, values: Union[List, float], factor: float):
        if isinstance(values, (int, float)):
            return values + random.uniform(-1.0, 1.0) * factor
        return [self._mutate(val, factor) for val in values]
