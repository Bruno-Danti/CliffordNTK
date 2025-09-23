"""
clifford_ntk/utils.py
"""

import numpy as np
from qiskit.quantum_info import PauliList, Pauli

def random_thetas(shape: tuple[int]) -> np.ndarray:
    """Generate random angles in {0, pi/2, pi, 3pi/2}."""
    possible_theta_values = np.pi / 2 * np.arange(4)
    return np.random.choice(possible_theta_values, shape)

def sum_Z(n_qubits: int) -> PauliList:
    """Return Σ_i Z_i as a PauliList."""
    def char(i,j): return 'Z' if i == j else 'I'
    return PauliList([
        ''.join([char(i,n_qubits - j - 1) for i in range(n_qubits)])
        for j in range(n_qubits)
    ])


def get_z_x_masks(pauli: Pauli) -> tuple[int, int]:
    arr = np.array([2**i for i in range(len(pauli))])
    z_mask = np.sum(pauli.z * arr)
    x_mask = np.sum(pauli.x * arr)
    return z_mask, x_mask