from lib.functions import Sigmoid, ReLU
from lib.layer import Layer
from lib.model import Model
from lib.training import DumbTrainer
from lib.util.print_table import print_table


class ModelXOR:
    @staticmethod
    def create_and_train():
        model = Model.load_or_create(Model(
            "model_xor",
            Layer(2, 2),
            # ReLU(),
            Sigmoid(),
            Layer(2, 2),
            Sigmoid(),
        ))

        inputs = [
            # A xor B
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]

        expected_outputs = [
            # TRUE, FALSE
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]

        trainer = DumbTrainer(
            target_error=0.0001,
            generations=100000,
            generation_size=10,
            variation_factors=[0.2]
        )
        trainer.enable_debug()
        model = trainer.train(model, inputs, expected_outputs)

        print("Expected")
        print_table(expected_outputs)

        print("Actual")
        print_table(model(inputs))

        model.save()
