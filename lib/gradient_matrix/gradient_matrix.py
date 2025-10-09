import numpy as np
import subprocess
import io

def gradient_matrix(
        n_images: int,
        n_qubits: int,
        n_trainable_gates: int,
        input_data_path: str,
        input_pauli_path: str
) -> np.ndarray:
    result = subprocess.run(
        ["./lib/gradient_matrix/compiled_kernel",
         str(n_images),
         str(n_qubits),
         str(2 ** n_qubits),
         str(n_trainable_gates),
         input_data_path,
         input_pauli_path
         ],
         check=True,
         capture_output=True
    )
    G = np.loadtxt(io.BytesIO(result.stdout), delimiter=",")
    return G