"""
Package for fast classical computation of the quantum
Neural Tangent Kernel (NTK) of a particular class
of QNNs.
"""

from . import utils, clifford_circuits, kernels, pauli_evolve

from clifford_ntk.utils import *
from clifford_ntk.clifford_circuits import ConvolutionalQNN, TestCircuit
from clifford_ntk.pauli_evolve import EvolvedPaulis