from .base import ParametricCliffordCircuit

class TestCircuit(ParametricCliffordCircuit):
    # def __init__(self, n_qubits, thetas):
    #     super().__init__(n_qubits, thetas)
    
    def _build_circuit(self) -> None:
        for i in range(self.n_qubits):
            self._add_param_gate("rx", self.thetas[i], i)
