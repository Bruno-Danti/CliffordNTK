import numpy as np, subprocess
from scipy.linalg.blas import dsyrk
from timeit import default_timer as timer

def part1():
    subprocess.run(
        ["python3", "Part_1/main.py", "out/tmp_evolved_paulis.bin"],
        check=True
    )

def part2(n_images, n_qubits, n_trainable_gates):
    subprocess.run(
        ["Part_2/a.out", str(n_images),
         str(n_qubits), str(2 ** n_qubits), str(n_trainable_gates),
         "encode_dataset/encoded_dataset.bin",
         "out/tmp_evolved_paulis.bin",
         "out/tmp_G.csv"]
    )


def part3(G: np.ndarray) -> np.ndarray:
    return dsyrk(alpha = 1.0, a = G, lower = False)


#t_start = timer()

n_samples = 3
n_images = 10000
n_qubits = 10
n_trainable_gates = 252

K_acc = np.zeros((n_images, n_images))
for _ in range(n_samples):
    # t0 = timer()
    part1()
    # t1 = timer()

    # print(f"Time for part 1: {t1 - t0}")
    
    part2(n_images, n_qubits, n_trainable_gates)

    G: np.ndarray = np.loadtxt(
        "out/tmp_G.csv", float, delimiter=',', max_rows=n_images)
    
    # t0 = timer()
    K_acc += part3(G)
    # t1 = timer()

    # print(f"Time to compute K = {t1 - t0}")

K_acc /= n_samples

# t_savetxt_st = timer()
np.savetxt("out/K.csv", K_acc, delimiter= ",")
# t_savetxt_end = timer()
# print(f"time to save the K matrix = {t_savetxt_end - t_savetxt_st}")


# t_end = timer()

# print(f"Total = {t_end - t_start}")