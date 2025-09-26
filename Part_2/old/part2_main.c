#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void read_sparse_dataset(
    const char *path,
    float ***dataset,
    uint16_t ***indices,
    uint32_t **n_indices,
    int n_images,
    int vector_length
){
    FILE *f = fopen(path, "rb");
    if (!f) {perror("open"); exit(1);}

    *dataset = malloc(n_images * sizeof(float*));
    *indices = malloc(n_images * sizeof(uint16_t*));
    *n_indices = malloc(n_images * sizeof(uint32_t));

    for (int i = 0; i < n_images; i++)
    {
        (*dataset)[i] = calloc(vector_length, sizeof(float));
        
        
        fread(&((*n_indices)[i]), sizeof(uint32_t), 1, f);
        uint32_t n = (*n_indices)[i];

        
        (*indices)[i] = malloc(n * sizeof(uint16_t));
        float *values = malloc(n * sizeof(float));
        
        fread((*indices)[i], sizeof(uint16_t), n, f);
        fread(values, sizeof(float), n, f);

        for (uint32_t j = 0; j < n; j++)
        {
            (*dataset)[i][(*indices)[i][j]] = values[j];
        }
        
        free(values);
    }
    fclose(f);
}

typedef struct {
    uint16_t z_mask;
    uint16_t x_mask;
    int8_t y_count;
    int8_t phase;
} PauliTerm;

void read_evolved_paulis(const char *path,
                         int n_qubits,
                         int n_gates,
                         PauliTerm ****out_ops) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open"); exit(1); }

    fread(&n_qubits, sizeof(uint32_t), 1, f);

    // Allocate [n_gates][2][n_qubits]
    PauliTerm ***ops = malloc(n_gates * sizeof(PauliTerm**));
    for (int g = 0; g < n_gates; g++) {
        ops[g] = malloc(2 * sizeof(PauliTerm*));
        for (int s = 0; s < 2; s++) {
            ops[g][s] = malloc((n_qubits) * sizeof(PauliTerm));
            fread(ops[g][s], sizeof(PauliTerm), n_qubits, f);
        }
    }

    fclose(f);
    *out_ops = ops;
}


float fast_kernel(
    uint16_t *indices, float *vector, uint32_t n_indices,
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



int main() {
    const int n_images = 60000;
    const int n_qubits = 10;
    const int vector_length = 1024;
    const int n_trainable_gates = 252;
    
    float **dataset;
    uint16_t **indices;
    uint32_t *n_indices;
    PauliTerm ***paulis;

    printf("Running part 2\n");


    read_sparse_dataset("encoded_dataset", &dataset, &indices, &n_indices,
    n_images, vector_length);

    printf("Dataset successfully read\n");

    read_evolved_paulis("evolved_paulis", n_qubits,
        n_trainable_gates, &paulis);
    
    printf("Evolved paulis successfully loaded\n");

    clock_t t0 = clock();
    printf("0\n");
    
    float (*G)[n_trainable_gates] = malloc(n_images * sizeof *G);
    if (G == NULL) {
        perror("malloc failed");
        exit(1);
    }
    printf("G matrix successfully declared\n");
    for (int i = 0; i < n_images; i++)
    {
        //printf("Outer loop\n");
        for (int j = 0; j < n_trainable_gates; j++)
        {
            //printf("Middle loop\n");
            G[i][j] = 0.0;
            for (int q = 0; q < n_qubits; q++)
            {
                //printf("Inner loop\n");
                G[i][j] += 0.5 * (
                    fast_kernel(indices[i], dataset[i], n_indices[i],
                        paulis[j][0][q])
                    -
                    fast_kernel(indices[i], dataset[i], n_indices[i],
                        paulis[j][1][q])
                );
                //printf("%f ", G[i][j]);
            }
            
            
        }
        // if (!(i % 100))
        // {
        //     printf("%i\n", i);
        // }
        
        
    }
    clock_t t1 = clock();
    printf("Elapsed: %f s\n", (float)(t1 - t0) / CLOCKS_PER_SEC);
    

    FILE *fp = fopen("G_matrix.csv", "w");
    if (fp == NULL) {
        perror("Failed to open file");
        return 1;
    }

    for (int i = 0; i < n_images; i++)
    {
        for (int j = 0; j < n_trainable_gates; j++)
        {
            if (j < n_trainable_gates - 1)
                fprintf(fp, "%f,", G[i][j]);
            else
                fprintf(fp, "%f", G[i][j]);
        }
        fprintf(fp, "\n");        
    }
    
    fclose(fp);
    return 0;
}




/*

int main() {
    const int n_images = 60000;
    const int n_qubits = 10;
    const int vector_length = 1024;
    const int n_trainable_gates = 252;
    
    float **dataset;
    uint16_t **indices;
    uint32_t *n_indices;
    PauliTerm ***paulis;

    printf("Running part 2\n");


    read_sparse_dataset("encoded_dataset", &dataset, &indices, &n_indices,
    n_images, vector_length);

    printf("Dataset successfully read\n");

    read_evolved_paulis("evolved_paulis", n_qubits,
        n_trainable_gates, &paulis);
    
    printf("Evolved paulis successfully loaded\n");

    clock_t t0 = clock();

    for (int i = 0; i < 1e9; i++)
    {
        fast_kernel(indices[0], dataset[0], n_indices[0],
        paulis[0][0][0]);
    }
    

    clock_t t1 = clock();
    printf("Elapsed: %f s\n", (float)(t1 - t0) / CLOCKS_PER_SEC);
}

*/