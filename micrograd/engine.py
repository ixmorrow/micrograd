import math
import numpy as np
import matplotlib.pyplot as plt
import random


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        """_summary_

        Args:
            data (Float): Scalar variable representing the value of this object.
            grad (Float): Will keep track of the gradient/derivative of this Value w.r.t. output node
            _children (tuple, optional): Tuple of previous nodes, will be stored as a Set(). Defaults to ().
            _op (str, optional): Operation between the _children nodes that produced this node. Defaults to ''.
            label (str, optional): Label for this node in graph. Defaults to ''.
        """
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, power):  # self**power
        assert isinstance(
            power, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(self.data**power, (self,), f"**{power}")

        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad = (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data})"

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __neg__(self):
        return self * -1

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self.data * other**-1
