from lib.functions import Sigmoid
from lib.layer import Layer
from lib.network import Network
from lib.training import DumbTrainer
from util.print_table import print_table

n = Network(
    Layer(2, 2),
    Sigmoid(),
    Layer(2, 2),
    Sigmoid(),
    Layer(2, 2),
)

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

trainer = DumbTrainer(1, 0.1)
n = trainer.train(n, inputs, expected_outputs)

print("Expected")
print_table(expected_outputs)

print("Actual")
print_table(n(inputs))

print("Error:", DumbTrainer.get_network_error(n, inputs, expected_outputs))

print(n)

n.enable_debug()
n(inputs)
