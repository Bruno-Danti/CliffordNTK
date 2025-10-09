import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.circuit.library import RXGate, RZGate

class ParametricCliffordCircuit:
    def __init__(self, n_qubits: int, thetas: np.ndarray):
        self.n_qubits = n_qubits
        self.thetas = thetas
        self.param_gate_label: int = 0
        self.label_index_dict: dict[str, int] = {}

        self.qc = QuantumCircuit(n_qubits)
        self._build_circuit()
    
    def _build_circuit(self) -> None:
        pass

    def _add_param_gate(self, gate_name: str, *args) -> None:
        getattr(self.qc, gate_name)(*args, label = str(self.param_gate_label))
        self.label_index_dict[str(self.param_gate_label)] = len(self.qc.data) - 1
        self.param_gate_label += 1
    
    @property
    def clifford_list(self) -> list[Clifford]:
        out: list[Clifford] = []
        gate = QuantumCircuit(self.n_qubits)
        for i in range(len(self.qc.data)):
            gate.data = [self.qc.data[i]]
            out.append(Clifford(gate))
        return out
    
    def get_labeled_gate(self, label: int | str, shift: float = 0) -> QuantumCircuit:
        index = self.label_index_dict[str(label)]
        old_instr = self.qc.data[index]

        if old_instr.operation.name == "rx":
            new_operation = RXGate(old_instr.params[0] + shift)
        if old_instr.operation.name == "rz":
            new_operation = RZGate(old_instr.params[0] + shift)

        out = QuantumCircuit(self.n_qubits)
        out.append(new_operation, old_instr.qubits, old_instr.clbits)
        return out
    
    def get_labeled_clifford(self, label: int | str, shift: float = 0) -> Clifford:
        return Clifford(
            self.get_labeled_gate(label, shift)
        )

    @property
    def n_gates(self): return len(self.qc.data)

    @property
    def n_parameters(self): return np.prod(self.thetas.shape)
