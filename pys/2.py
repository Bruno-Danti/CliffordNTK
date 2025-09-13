import numpy as np
from numba import njit
from torchvision.datasets import MNIST
import torch
from torchvision.transforms import ToTensor, Resize
from timeit import default_timer as timer

mnist_train = MNIST("./data",
                    train= True,
                    transform= ToTensor(),
                    download= True)

n_qubits = 10
n_qubits_cut = 8

@njit(inline='always')
def popcount(x):
    # Kernighan’s algorithm
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c

@njit
def pauli_expectation(indices, amps, delta, z_mask, y_mask, n_qubits):
    acc = 0.0 + 0.0j
    n = indices.shape[0]

    for k in range(n):
        i = indices[k]
        amp_i = amps[k]

        # partner index after bit flips
        j = i ^ delta

        # binary search (requires indices sorted ascending!)
        amp_j = 0.0 + 0.0j
        kk = np.searchsorted(indices, j)   # <-- was 'i' in your snippet; should be 'j'
        if kk < n and indices[kk] == j:
            amp_j = amps[kk]

        if amp_j != 0.0j:
            # Z phase: (-1)^{popcount(i & z_mask)}
            parity = popcount(i & z_mask) & 1
            phase = (-1.0 + 0.0j) if parity else (1.0 + 0.0j)

            # Y phase: i^{popcount(i & y_mask)}  (mod 4)
            y_par = popcount(i & y_mask) % 4
            if y_par == 1:
                phase *= 1j
            elif y_par == 2:
                phase *= -1.0
            elif y_par == 3:
                phase *= -1j

            acc += amp_i.conjugate() * phase * amp_j

    return acc

@njit
def pauli_expectation_2(
    indices, full_vector,
    delta, z_mask, y_mask, n_qubits
):
    acc = 0.0 + 0.0j
    n = indices.shape[0]

    for k in range(n):
        i = indices[k]
        amp_i = full_vector[i]

        j = i ^ delta
        amp_j = full_vector[j]
        # Z phase: (-1)^{popcount(i & z_mask)}
        parity = popcount(i & z_mask) & 1
        phase = (-1.0 + 0.0j) if parity else (1.0 + 0.0j)

        # Y phase: i^{popcount(i & y_mask)}  (mod 4)
        y_par = popcount(i & y_mask) % 4
        if y_par == 1:
            phase *= 1j
        elif y_par == 2:
            phase *= -1.0
        elif y_par == 3:
            phase *= -1j

        acc += amp_i.conjugate() * phase * amp_j
    return acc

def get_image(idx: int, dataset, n_qubits: int):
    image, label = dataset[idx]
    size = 2**(n_qubits//2)
    padded_image = torch.zeros([size, size])
    padded_image[:28, :28] = image.squeeze()[:, :]
    return padded_image

def img_to_vec(image: torch.Tensor):
    n = int(np.log2(image.shape[0]))  # Image should be a 2^n x 2^n tensor
    # Convert tensor indices from y,x to y0,y1,...,yn-1,x0,x1,...,xn-1
    image = image.reshape([2 for i in range(2*n)])
    # Permute indices from y0,y1,...,yn-1,x0,x1,...,xn-1 to y0,x0,y1,x1,...,yn-1,xn-1
    image = image.permute([i+j for i in range(n) for j in [0, n]])
    vec = image.flatten()
    return np.array([x for x in vec])


img = get_image(0, mnist_train, n_qubits)
vec = img_to_vec(img)
vec_cut = vec[:2**n_qubits_cut]
vec_indices = np.where(vec > 0.0)[0]
vec_indices_cut = np.where(vec_cut > 0.0)[0]
vec_amps = vec[vec_indices]
print(len(vec_indices), len(vec_indices_cut))
print(len(vec_cut))

delta = 1
z_mask = 1
y_mask = 0


pauli_expectation(vec_indices, vec_amps,
                  delta, z_mask, y_mask, n_qubits)
pauli_expectation_2(vec_indices, vec,
                  delta, z_mask, y_mask, n_qubits)

start_time = timer()
for _ in range(1): pauli_expectation(vec_indices, vec_amps,
                  delta, z_mask, y_mask, n_qubits)
end_time = timer()
print(end_time - start_time)

start_time = timer()
for _ in range(1000): pauli_expectation_2(vec_indices, vec,
                  delta, z_mask, y_mask, n_qubits)
end_time = timer()
print(end_time - start_time)

start_time = timer()
for _ in range(1000): pauli_expectation_2(vec_indices_cut, vec_cut,
                  delta, z_mask, y_mask, n_qubits)
end_time = timer()
print(end_time - start_time)