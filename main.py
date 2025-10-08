import numpy as np, subprocess
from scipy.linalg.blas import dsyrk
from timeit import default_timer as timer
from tqdm import tqdm

# def part1():
#     subprocess.run(
#         ["python3", "Part_1/main.py", "out/tmp_evolved_paulis.bin"],
#         check=True
#     )

from Part_1.main import main as part1

# def part2(n_images, n_qubits, n_trainable_gates,
#           input_path: str) -> np.ndarray:
#     subprocess.run(
#         ["Part_2/a.out", str(n_images),
#          str(n_qubits), str(2 ** n_qubits), str(n_trainable_gates),
#         #  "encode_dataset/encoded_dataset.bin",
#          input_path,
#          "out/tmp_evolved_paulis.bin",
#          "out/tmp_G.csv"]
#     )
#     G: np.ndarray = np.loadtxt(
#         "out/tmp_G.csv", float, delimiter=',', max_rows=n_images)
#     return G

from Part_2.main import main as part2


def part3(G: np.ndarray) -> np.ndarray:
    return dsyrk(alpha = 1.0, a = G, lower = False)

def part3_test(G: np.ndarray, G_test: np.ndarray) -> np.ndarray:
    # print(f"Shape of G_test = {G_test.shape}.")
    # print(f"Shape of G = {G.shape}.")
    return G_test @ G.T


#t_start = timer()

n_samples = 10
n_images = 10000
n_images_test = 157
n_qubits = 10
n_layers = 3
n_trainable_gates = 252

K_acc = np.zeros((n_images, n_images))
K_acc_test = np.zeros((n_images_test, n_images))
for _ in tqdm(range(n_samples)):

    
    # t0 = timer()
    part1(n_qubits, n_layers)
    # t1 = timer()

    # print(f"Time for part 1: {t1 - t0}")
    
    G = part2(n_images, n_qubits, n_trainable_gates,
          "encode_dataset/encoded_dataset.bin",
          "out/tmp_paulis.bin")

    G_test = part2(n_images_test, n_qubits, n_trainable_gates,
                   "encode_dataset/encoded_dataset_test.bin",
                   "out/tmp_paulis.bin")
    
    # t0 = timer()
    K_acc += part3(G)
    K_acc_test += part3_test(G, G_test)
    # t1 = timer()

    # print(f"Time to compute K = {t1 - t0}")

K_acc /= n_samples

# t_savetxt_st = timer()
np.savetxt("out/K.csv", K_acc, delimiter= ",")
np.savetxt("out/K_test.csv", K_acc_test, delimiter= ",")
# t_savetxt_end = timer()
# print(f"time to save the K matrix = {t_savetxt_end - t_savetxt_st}")


# t_end = timer()

# print(f"Total = {t_end - t_start}")