from .base import ParametricCliffordCircuit
import numpy as np

class ConvolutionalQNN(ParametricCliffordCircuit):
    def __init__(self, n_qubits: int, n_layers: int, thetas: np.ndarray):
        self.n_layers = n_layers
        super().__init__(n_qubits, thetas)
    
    def _two_qubit_gate(self, thetas: np.ndarray, qubit1: int, qubit2: int) -> None:
        self._add_param_gate("rx", thetas[0], qubit1)
        self._add_param_gate("rx", thetas[1], qubit2)
        self.qc.cx(qubit1, qubit2)
        self._add_param_gate("rz", thetas[2], qubit1)
        self._add_param_gate("rz", thetas[3], qubit2)
    
    def _layer(self, thetas: np.ndarray) -> None:
        i: int = 0  # Current column of used thetas
        def gate(q1: int, q2: int) -> None:
            "Applies the elementary _two_qubit_gate sweeping through thetas' columns"
            nonlocal i
            self._two_qubit_gate(thetas[:,i], q1, q2)
            i += 1
        
        def contiguous_qubits_gates(q: int) -> None:
            '''
            Args:
                q (int): Smallest qubit the bundle of gates is applied to. \\
                         e.g. contiguous_qubits_gates(2) applies to (2,3), (4,5), and (6,7)
            '''
            if q >= 0:
                gate(q, q+1)
            gate(q+2, q+3)
            if q + 5 <= self.n_qubits - 1:
                gate(q+4, q+5)

        def alternate_qubits_gates(q: int) -> None:
            gate(q, q + 2)
            gate(q + 1, q + 3)
        
        contiguous_qubits_gates(-2)
        for q in np.arange(0, self.n_qubits-2, 2):
            alternate_qubits_gates(q)
            contiguous_qubits_gates(q)

    def _build_circuit(self):
        reshaped_thetas = self.thetas.reshape(
            (4, 5 * self.n_qubits // 2 - 4, self.n_layers)
            )
        for i in range(self.n_layers):
            self._layer(reshaped_thetas[:,:,i])
            self.qc.barrier()
