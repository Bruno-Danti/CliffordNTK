/*
Usage:
compute_G_matrix.out n_images n_qubits
vector_length n_trainable_gates
data_path pauli_path

outputs the G matrix on stdout
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    uint16_t z_mask;
    uint16_t x_mask;
    int8_t y_count;
    int8_t phase;
} PauliTerm;

void read_data(
    const char *path,
    int vector_length,
    int n_images,
    float (*dataset)[vector_length],
    uint32_t *n_indices,
    uint16_t **indices
) {
    FILE *f = fopen(path, "rb");
    if (!f) {perror("open"); exit(1);}

    for (size_t i = 0; i < n_images; i++) {
        uint32_t n;
        if (fread(&n, sizeof(uint32_t), 1, f) != 1)
        {
            perror("fread failed");
        }
        
        ;
        n_indices[i] = n;

        indices[i] = malloc(n * sizeof(uint16_t));
        float *values = malloc(n * sizeof(float));

        if (fread(indices[i], sizeof(uint16_t), n, f) != n) {
            perror("fread failed");
        }

        if(fread(values, sizeof(float), n, f) != n) {
            perror("fread failed");
        }

        for (size_t j = 0; j < n; j++) {
            dataset[i][indices[i][j]] = values[j];
        }
        

        free(values);
    }
    

    fclose(f);
}

void read_paulis(
    const char *path,
    int n_qubits,
    int n_trainable_gates,
    PauliTerm (*paulis)[2][n_qubits]
) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open"); exit(1); }

    int n_qubits_read;
    if (fread(&n_qubits_read, sizeof(uint32_t), 1, f) != 1)
    {
        perror("fread failed");
    }
    

    for (size_t gate = 0; gate < n_trainable_gates; gate++) {
        for (size_t sign = 0; sign < 2; sign++) {
            for (size_t qubit = 0; qubit < n_qubits; qubit++) {
                if (
                    fread(&(paulis[gate][sign][qubit]),
                        sizeof(PauliTerm), 1, f)
                        != 1
                )
                {
                    perror("fread failed");
                }
            }
        }
    }

    fclose(f);
}

inline float fast_kernel(
    uint16_t * restrict indices, float * restrict vector, uint32_t n_indices,
    PauliTerm pauli
) {
    int phase_y;
    switch (pauli.y_count % 4)
    {
    case 1:
        return 0.0;
        break;
    case 3:
        return 0.0;
        break;
    case 0:
        phase_y = 1;
        break;
    case 2:
        phase_y = -1;
        break;
    
    default:
        break;
    }
    
    float out = 0.0;
    for (int k = 0; k < n_indices; k++)
    {
        uint16_t i = indices[k];
        float psi_i = vector[i];

        uint16_t j = i ^ pauli.x_mask;
        float psi_j = vector[j];

        int parity = __builtin_popcount(i & pauli.z_mask) % 2;
        int phase_i = (parity) ? -1 : 1;
        out += psi_i * psi_j * phase_i;
    }
    
    return out * phase_y * pauli.phase;
}


int main(int argc, char* argv[]) {
    const int n_images = atoi(argv[1]);
    const int n_qubits = atoi(argv[2]);
    const int vector_length = atoi(argv[3]);
    const int n_trainable_gates = atoi(argv[4]);

    float (*dataset)[vector_length] = calloc(n_images, sizeof *dataset);
    uint32_t *n_indices = malloc(n_images * sizeof(uint32_t));
    uint16_t **indices = malloc(n_images * sizeof(uint16_t*));
    PauliTerm (*paulis)[2][n_qubits] = malloc(n_trainable_gates * sizeof *paulis);

    read_data(
        argv[5], vector_length, n_images,
        dataset, n_indices, indices
    );
    read_paulis(argv[6], n_qubits, n_trainable_gates, paulis);
    

    float (*G)[n_trainable_gates] = calloc(n_images, sizeof *G);

    for (size_t i = 0; i < n_images; i++) {
        uint16_t *idx_i = indices[i];
        float *vector = dataset[i];
        int n_i = n_indices[i];
        for (size_t j = 0; j < n_trainable_gates; j++) {
            for (size_t q = 0; q < n_qubits; q++) {
                G[i][j] += 0.5 * (
                    fast_kernel(idx_i, vector, n_i, paulis[j][0][q])
                    -
                    fast_kernel(idx_i, vector, n_i, paulis[j][1][q])
                );
            }
        }
    }

    for (int i = 0; i < n_images; i++)
    {
        for (int j = 0; j < n_trainable_gates; j++)
        {
            if (j < n_trainable_gates - 1)
                printf("%f,", G[i][j]);
            else
                printf("%f", G[i][j]);
        }
        printf("\n");        
    }
    

    free(dataset);
    free(n_indices);
    for (size_t i = 0; i < n_images; i++) {
        free(indices[i]);
    }
    free(indices);
    free(paulis);
    free(G);
    
    return 0;
}