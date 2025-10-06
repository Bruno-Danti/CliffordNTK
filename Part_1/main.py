from clifford_ntk import (
    random_thetas,
    sum_Z,
    ConvolutionalQNN,
    EvolvedPaulis,
    write_evolved_paulis
)
import sys

n_q = 10
n_l = 3
thetas = random_thetas(4 * n_l * (5*n_q//2 - 4))

op = sum_Z(n_q)
qnn = ConvolutionalQNN(n_q, n_l, thetas)
evolved_paulis = EvolvedPaulis(op, qnn)
write_evolved_paulis(evolved_paulis, sys.argv[1])