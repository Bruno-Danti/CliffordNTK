import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
import gc

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

'''
n_samples = 1
n_images = 20010
n_images_test = 10000
n_qubits = 10
n_layers = 3
n_trainable_gates = 252

tmp_pauli_path = "./data/tmp/paulis.bin"
train_train_out_path = "./data/out/K_train_train.csv"
test_train_out_path = "./data/out/K_test_train.csv"

encoded_train_set_path = "./data/download/train.bin"
encoded_test_set_path = "./data/download/test.bin"
'''


def perform_experiment(
        n_samples: int, n_train_images: int,
        n_test_images: int, n_qubits: int,
        n_layers: int, n_trainable_gates: int,
        
        tr_set_path: str,  # Encoded train set input path
        ts_set_path: str,  # Encoded test set input path
        pauli_path: str,  # Temporary path where Pauli ops are stored
        K_tr_tr_path: str,  # Output path where the K_train_train matrix is saved
        K_ts_tr_path: str  # Output path where the K_test_train matrix is saved
) -> None:
    
    PROFILE_TIME: bool = False

    K_train_train_acc = np.zeros((n_train_images, n_train_images), dtype=np.float32)
    K_test_train_acc = np.zeros((n_train_images, n_test_images), dtype=np.float32) # The dimensions are swapped due to how dgemm works.

    print("Computing the NTK matrices for various sampled thetas:")


    for _ in tqdm(range(n_samples)):
        evolve_paulis(n_qubits,
                    n_layers,
                    pauli_path)
        if PROFILE_TIME:
            t0 = timer()
        G_train = gradient_matrix(
            n_train_images, n_qubits, n_trainable_gates,
            tr_set_path,
            pauli_path
        )
        G_test = gradient_matrix(
            n_test_images, n_qubits, n_trainable_gates,
            ts_set_path,
            pauli_path
        )
        if PROFILE_TIME:
            t1 = timer()
            print(f"\nTime to compute G matrices: {t1 - t0}s.")

        if PROFILE_TIME:
            t0 = timer()
        #K_train_train_acc += self_kernel(G_train)
        #K_test_train_acc += other_kernel(G_test, G_train)        

        from scipy.linalg.blas import ssyrk, sgemm  # s for 32 bits and d for 64
        ssyrk(alpha=1.0, a=G_train, c=K_train_train_acc.T, lower=False, beta=1.0, overwrite_c=True)
        sgemm(alpha=1.0, a=G_test, b=G_train, c=K_test_train_acc.T, trans_b=True, beta=1.0, overwrite_c=True)



        if PROFILE_TIME:
            t1 = timer()
            print(f"Time for linalg ops: {t1 - t0}s.")

    del G_train, G_test
    gc.collect()

    K_train_train_acc /= np.float32(n_samples)
    # def symmetrize(mat: np.ndarray) -> np.ndarray:
    #     return mat + np.tril(mat, -1).T

    # This version requires less memory because it doesn't allocate memory for the returned variable
    def symmetrize_inplace(mat: np.ndarray, block_size: int = 5000) -> np.ndarray:
        n = mat.shape[0]
        for i in range(0, n, block_size):
            for j in range(i, n, block_size):
                if i == j:
                    # Blocks on diagonal
                    block = mat[i:i+block_size, i:i+block_size]
                    triu_idx = np.triu_indices(block.shape[0], k=1)
                    block[triu_idx] = block.T[triu_idx]
                else:
                    # Off-diag blocks
                    mat[i:i+block_size, j:j+block_size] = mat[j:j+block_size, i:i+block_size].T


    # K_train_train_acc = symmetrize(K_train_train_acc)
    print("Symmetrizing the K_train_train matrix...")
    symmetrize_inplace(K_train_train_acc)

    K_test_train_acc /= np.float32(n_samples)

    


    print(f"Saving the K_test_train to {K_ts_tr_path}...")
    t0 = timer()
    np.save(K_ts_tr_path, K_test_train_acc.T)
    t1 = timer()
    print(f"Done. Elapsed = {t1-t0}s.")

    del K_test_train_acc
    gc.collect()



    print(f"Saving the K_train_train to {K_tr_tr_path}...")
    t0 = timer()
    np.save(K_tr_tr_path, K_train_train_acc)
    t1 = timer()
    print(f"Done. Elapsed = {t1-t0}s.")