"""
encode_MNIST.py
Objective: create a dataset file where MNIST images are encoded in a
fast-to-read format for the C code in the heavy computing part
of the project.
"""

import torch, numpy as np
from timeit import default_timer as timer
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import struct
from tqdm import tqdm


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

def write_dataset(matrix: np.ndarray, path: str):
    with open(path, "wb") as f:
        for vector in tqdm(matrix):
            nonzero: np.ndarray = np.nonzero(vector)[0].astype(np.uint16)
            values: np.ndarray = vector[nonzero].astype(np.float32)
            n = len(nonzero)
            f.write(struct.pack("<I", n))
            f.write(nonzero.tobytes())
            #print(nonzero)
            f.write(values.tobytes())
            #print(values)

mnist_train = MNIST(".",
                    train = True,
                    transform = ToTensor(),
                    download = True)

mnist_test = MNIST(".",
                    train = False,
                    transform = ToTensor(),
                    download = True)

n_qubits: int = 10
path = "encoded_train"

def main(dataset: MNIST, path: str, n_qubits: int, dataset_name: str = ""):
    dataset_matrix = np.zeros((len(dataset), 2**n_qubits))
    print(f"Encoding {dataset_name}:")
    for i in tqdm(range(len(dataset_matrix))):
        vec = img_to_vec(
            get_image(i, dataset, n_qubits)
        )
        dataset_matrix[i,:] = vec / np.sqrt(np.sum(vec ** 2))
    print(f"Printing {dataset_name}:")
    write_dataset(dataset_matrix, path)



start_time = timer()

main(mnist_train, "./train.bin", n_qubits, "train set")
main(mnist_test, "./test.bin", n_qubits, "test set")


end_time = timer()

print("Success.")
print(f"Elapsed: {end_time - start_time}")