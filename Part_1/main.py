from .clifford_ntk import (
    random_thetas,
    sum_Z,
    ConvolutionalQNN,
    EvolvedPaulis,
    write_evolved_paulis
)
import sys, numpy as np


def main(n_q: int, n_l: int, out_path: str = "out/tmp_paulis.bin",
         write_thetas_log: bool = True):

    # n_q = 10
    # n_l = 3
    thetas = random_thetas(4 * n_l * (5*n_q//2 - 4))

    op = sum_Z(n_q)
    qnn = ConvolutionalQNN(n_q, n_l, thetas)
    evolved_paulis = EvolvedPaulis(op, qnn)
    write_evolved_paulis(evolved_paulis, out_path)

    if write_thetas_log:
        with open("out/thetas.csv", "a") as f:
            np.savetxt(f, thetas.reshape(1,-1), delimiter=',')