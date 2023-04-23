from typing import List, Union

from lib.functions import Activation
from lib.layer import Layer


class Network:
    def __init__(self, *layers: Union[Activation, Layer]):
        self._layers = layers
        self._debug = False

    def __call__(self, inputs: List[List[float]]):
        outputs = []

        self._debug_log(f"Executing {len(inputs)} input rows...")

        for _index, _inputs_row in enumerate(inputs):
            self._debug_log(f"Executing row {_index}")
            self._debug_log(f" -  Input:", _inputs_row)

            _outputs_row = [*_inputs_row]
            for layer in self._layers:
                _outputs_row = layer(_outputs_row)

            outputs.append(_outputs_row)
            self._debug_log(f" - Output:", _outputs_row)

        return outputs

    def __str__(self):
        output_lines = ["Network", "Layers"]

        for layer in self._layers:
            output_lines += [f" - {str(layer)}"]

        return "\n".join(output_lines)

    @property
    def layers(self):
        return [layer.clone() if isinstance(layer, Layer) else layer for layer in self._layers]

    def enable_debug(self):
        self._debug = True

    def disable_debug(self):
        self._debug = False

    def _debug_log(self, *args):
        if self._debug:
            print("Network Debug:", *self._prepare_value_to_log(args))

    def _prepare_value_to_log(self, value):
        if isinstance(value, tuple):
            return map(self._prepare_value_to_log, value)

        if isinstance(value, list):
            return f"[ {', '.join(map(self._prepare_value_to_log, value))} ]"

        if isinstance(value, (int, float)):
            return "{value:.2f}".format(value=value)

        return value
