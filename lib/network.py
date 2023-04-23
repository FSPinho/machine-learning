from typing import List, Union

from lib.debuggable import Debuggable
from lib.functions import Activation
from lib.layer import Layer


class Network(Debuggable):
    def __init__(self, *layers: Union[Activation, Layer]):
        super().__init__()
        self._layers = layers

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

