import codecs
import numpy as np
from json import loads, dumps, JSONDecodeError
from os import mkdir
from os.path import dirname, exists, join
from typing import List, Union

from lib.cloneable import Cloneable
from lib.debuggable import Debuggable
from lib.functions import Activation
from lib.layer import Layer
from lib.serializable import Serializable

MODELS_PATH = join(dirname(dirname(__file__)), "trained_models")


class Model(Debuggable, Serializable, Cloneable):
    def __init__(self, key: str, *layers: Union[Activation, Layer]):
        super().__init__()
        self._key = key
        self._layers = layers

    def __call__(self, inputs: List[List[float]]):
        outputs = []

        for _index, _inputs_row in enumerate(inputs):
            if _index % 100 == 0 or _index == len(inputs) - 1:
                self._debug_log(f"Executing input row {_index}/{len(inputs)}", end="\r", flush=True)

            _outputs_row = _inputs_row
            for layer in self._layers:
                _outputs_row = layer(_outputs_row)

            outputs.append(_outputs_row)

        self._debug_log(f"")

        return outputs

    def __str__(self):
        output_lines = ["Model", "Layers"]

        for layer in self._layers:
            output_lines += [f" - {str(layer)}"]

        return "\n".join(output_lines)

    @property
    def layers(self):
        return [layer.clone() for layer in self._layers]

    def clone(self, **kwargs):
        clone = Model(self._key, *[layer.clone() for layer in self._layers])

        for key, val in kwargs.items():
            setattr(clone, key, val)

        return clone

    def serialize(self) -> dict:
        return dict(
            type=Model.__name__,
            key=self._key,
            layers=[layer.serialize() for layer in self._layers]
        )

    @staticmethod
    def deserialize(instance_dict: dict):
        if instance_dict.get("type") == Model.__name__:
            model = Model(key="")
            return model.clone(
                _key=instance_dict.get("key"),
                _layers=[
                    Activation.deserialize(layer) or Layer.deserialize(layer)
                    for layer in instance_dict.get("layers", [])
                ]
            )
        return None

    def save(self):
        model_math = Model.get_persistence_path(self._key)

        if not exists(dirname(model_math)):
            mkdir(dirname(model_math))

        with codecs.open(model_math, mode="w", encoding="UTF-8") as file:
            data = dumps(self.serialize(), indent=4, ensure_ascii=False)
            file.write(data)

    @staticmethod
    def load(key):
        try:
            model_path = Model.get_persistence_path(key)
            if exists(model_path):
                with codecs.open(model_path, encoding="UTF-8") as file:
                    data = loads(file.read())
                    return Model.deserialize(data)
        except JSONDecodeError as e:
            print(f"Can't load network: {e.msg}")

        return None

    @staticmethod
    def load_or_create(model: 'Model'):
        return Model.load(model._key) or model

    @staticmethod
    def get_persistence_path(key: str):
        return join(MODELS_PATH, f"{key}.json")
