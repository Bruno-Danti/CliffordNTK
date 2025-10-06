import numpy as np, subprocess
from scipy.linalg.blas import dsyrk

def part3(G: np.ndarray) -> np.ndarray:
    return dsyrk(alpha = 1.0, a = G, lower = False)

n_samples = 10
n_images = 1000
n_qubits = 10
n_trainable_gates = 252

K_acc = np.zeros((n_images, n_images))
for _ in range(n_samples):
    subprocess.run(
        ["python3", "Part_1/main.py", "out/tmp_evolved_paulis.bin"],
        check=True
    )
    subprocess.run(
        ["Part_2/a.out", str(n_images),
         str(n_qubits), str(2 ** n_qubits), str(n_trainable_gates),
         "encode_dataset/encoded_dataset.bin",
         "out/tmp_evolved_paulis.bin",
         "out/tmp_G.csv"]
    )
    G: np.ndarray = np.loadtxt(
        "out/tmp_G.csv", float, delimiter=',', max_rows=n_images)
    K_acc += part3(G)

K_acc /= n_samples

np.savetxt("out/K.csv", K_acc, delimiter= ",")