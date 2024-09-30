import math
import numpy as np
import matplotlib.pyplot as plt
import random
from micrograd.engine import Value


class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(random.uniform(-1, 1))

    def parameters(self):
        return self.w + [self.b]

    def __call__(self, x):
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = activation.tanh()
        return out


class Layer:
    # TODO: Implement shape member function

    def __init__(self, n_inputs, n_outputs):
        """
        Initialize neurons for layer. Each neuron should have weights for
        given inputs. Number of neurons in layer should equal n_outputs for layer.
        Each neuron provides a single output from the layer to the next.
        """
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __call__(self, x):
        # why is single input x going into each neuron?
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs


class MultiLayerPerception:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
