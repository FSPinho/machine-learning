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

trainer = DumbTrainer(target_error=0.0001, generations=100000, generation_size=2, variation_factor=0.1)
trainer.enable_debug()
n = trainer.train(n, inputs, expected_outputs)

print("Expected")
print_table(expected_outputs)

print("Actual")
print_table(n(inputs))

print("Error:", DumbTrainer.get_network_error(n, inputs, expected_outputs))

print(n)
print("Weights")

n.enable_debug()
n(inputs)
