import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer

from lib.clifford_pauli_evolve import(
    ConvolutionalQNN,
    EvolvedPaulis,
    random_thetas,
    sum_Z,
    write_evolved_paulis
)
from lib.gradient_matrix.gradient_matrix import gradient_matrix
from lib.kernel_matrix.kernel_matrix import self_kernel, other_kernel

def evolve_paulis(n_qubits: int,
                  n_layers: int,
                  out_pauli_path: str,
                  log_thetas: bool = True):
    thetas = random_thetas(4 * n_layers * (5*n_qubits//2 - 4))
    if log_thetas:
        with open("./data/log/thetas.csv", "a") as f:
            np.savetxt(f, thetas.reshape(1,-1), delimiter=',')
    
    op = sum_Z(n_qubits)
    qnn = ConvolutionalQNN(n_qubits, n_layers, thetas)
    out = EvolvedPaulis(op, qnn)
    write_evolved_paulis(out, out_pauli_path)


n_samples = 1
n_images = 20010
n_images_test = 10000
n_qubits = 10
n_layers = 3
n_trainable_gates = 252

K_train_train_acc = np.zeros((n_images, n_images))
K_test_train_acc = np.zeros((n_images_test, n_images))

tmp_pauli_path = "./data/tmp/paulis.bin"
print("Computing the NTK matrices for various sampled thetas:")
for _ in tqdm(range(n_samples)):
    evolve_paulis(n_qubits,
                  n_layers,
                  tmp_pauli_path)
    G_train = gradient_matrix(
        n_images, n_qubits, n_trainable_gates,
        "./data/download/train.bin",
        tmp_pauli_path
    )
    G_test = gradient_matrix(
        n_images_test, n_qubits, n_trainable_gates,
        "./data/download/test.bin",
        tmp_pauli_path
    )

    K_train_train_acc += self_kernel(G_train)
    K_test_train_acc += other_kernel(G_test, G_train)

K_train_train_acc /= n_samples
def symmetrize(mat: np.ndarray) -> np.ndarray:
    return mat + np.triu(mat, 1).T
K_train_train_acc = symmetrize(K_train_train_acc)

K_test_train_acc /= n_samples

train_train_out_path = "./data/out/K_train_train.csv"
print(f"Saving the K_train_train to {train_train_out_path}...")
t0 = timer()
np.savetxt(train_train_out_path, K_train_train_acc, delimiter= ",")
t1 = timer()
print(f"Done. Elapsed = {t1-t0}s.")

test_train_out_path = "./data/out/K_test_train.csv"
print(f"Saving the K_test_train to {test_train_out_path}...")
t0 = timer()
np.savetxt(test_train_out_path, K_test_train_acc, delimiter= ",")
t1 = timer()
print(f"Done. Elapsed = {t1-t0}s.")