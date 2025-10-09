import numpy as np
from qiskit.quantum_info import PauliList
from .clifford_list import CliffordList
from ..clifford_circuits.base import ParametricCliffordCircuit

class EvolvedPaulis:
    def __init__(self, op: PauliList, param_circ: ParametricCliffordCircuit):
        self.param_circ: ParametricCliffordCircuit = param_circ
        self.label_index_dict = self.param_circ.label_index_dict
        self.op: PauliList = op
        self.cliff_list: CliffordList = CliffordList(self.param_circ)

        self.plus_ops: dict[str, PauliList] = {}
        self.minus_ops: dict[str, PauliList] = {}
        self.evolve_op()
    
    def evolve_op(self) -> None:
        I = self.cliff_list.I
        shift = np.pi / 2
        for label, index in self.label_index_dict.items():
            head = self.cliff_list.heads[index - 1] if index > 0 else I
            gate_plus = self.param_circ.get_labeled_clifford(label, shift)
            gate_minus =  self.param_circ.get_labeled_clifford(label, -shift)
            tail = self.cliff_list.tails[index + 1] if index < self.param_circ.n_gates-1 else I

            clifford_plus = head.compose(gate_plus.compose(tail))
            clifford_minus = head.compose(gate_minus.compose(tail))

            self.plus_ops[label] = self.op.evolve(clifford_plus)
            self.minus_ops[label] = self.op.evolve(clifford_minus)
