import clifford_ntk

n_q = 10
n_l = 3
thetas = clifford_ntk.utils.random_thetas(4 * n_l * (5*n_q//2 - 4))

op = clifford_ntk.utils.sum_Z(n_q)

qnn = clifford_ntk.clifford_circuits.ConvolutionalQNN(n_q, n_l, thetas)

evolved_paulis = clifford_ntk.pauli_evolve.EvolvedPaulis(op, qnn)

single_pauli = evolved_paulis.plus_ops["0"][0]

print(single_pauli)



import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

mnist_train = MNIST("./data",
                    train= True,
                    transform= ToTensor(),
                    download= True)

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

vec = img_to_vec(
    get_image(
        0,
        mnist_train,
        n_q
    )
)
indices = np.where(vec != 0)[0]
n_indices = len(indices)


z_mask, x_mask = clifford_ntk.utils.get_z_x_masks(
    single_pauli
)
count_y = single_pauli._count_y()
phase = -1.0 if single_pauli.phase == 2 else 1.0


out = clifford_ntk.kernels.fast_kernel(
    indices, vec, n_indices,
    z_mask, x_mask, count_y, phase
)

from timeit import default_timer as timer

start_time = timer()

out = clifford_ntk.kernels.fast_kernel(
    indices, vec, n_indices,
    z_mask, x_mask, count_y, phase
)

end_time = timer()
print(out)

print(end_time - start_time)