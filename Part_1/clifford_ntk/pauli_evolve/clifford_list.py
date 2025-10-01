from qiskit.quantum_info import Clifford
from ..clifford_circuits.base import ParametricCliffordCircuit

class CliffordList:
    def __init__(self, param_circ: ParametricCliffordCircuit):
        self.list: list[Clifford] = param_circ.clifford_list
        self.n_qubits: int = param_circ.n_qubits
        self.I: Clifford = Clifford.from_label("I" * param_circ.n_qubits)

        self.heads = self.build_heads()
        self.tails = self.build_tails()
    
    def build_heads(self) -> list[Clifford]:
        out = []
        op = self.I
        for gate in self.list:
            op = op.compose(gate)
            out.append(op)
        return out
    
    def build_tails(self) -> list[Clifford]:
        out = []
        op = self.I
        for gate in reversed(self.list):
            op = op.compose(gate, front = True)
            out.append(op)
        out.reverse()
        return out
