"""
clifford_ntk/utils.py
"""

import numpy as np, struct
from qiskit.quantum_info import PauliList, Pauli
from .pauli_evolve.evolved_paulis import EvolvedPaulis


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

def write_evolved_paulis(evolved_paulis: EvolvedPaulis, path: str):
    n_qubits = evolved_paulis.param_circ.n_qubits
    n_gates = len(evolved_paulis.plus_ops)

    with open(path, "wb") as f:
        f.write(struct.pack("<I", n_qubits))

        for j in range(n_gates):
            for ops in [evolved_paulis.plus_ops[str(j)], evolved_paulis.minus_ops[str(j)]]:
                for pauli in ops:  # pauli is a single Pauli string like "XIZ..."
                    #z_mask, x_mask, y_count, phase = pauli.to_masks_phase()
                    z_mask, x_mask = get_z_x_masks(pauli)
                    y_count = pauli._count_y()[0]
                    phase = pauli.phase
                    #print(z_mask, x_mask, y_count, phase)
                    # z_mask and x_mask fit in 16 bits since n_qubits=10
                    f.write(struct.pack("<HHb b", z_mask, x_mask, y_count, phase))
