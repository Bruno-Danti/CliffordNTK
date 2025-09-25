#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

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

int main() {
    int n_qubits = 10;
    PauliTerm ***ops;
    read_evolved_paulis("evolved_paulis", n_qubits, 252, &ops);
    PauliTerm op = ops[0][0][0];
    printf("%i %i %i %i\n", op.z_mask, op.x_mask, op.y_count, op.phase);
    printf("Success!\n");
}



/*
TO DO: I HAVE TO MODIFY THE READ_DATASET.C IN SUCH A WAY THAT IT ALSO STORES
INDICES AND N_INDICES. THEN I HAVE TO MODIFY THIS READ_EVOLVED_PAULIS.C
IN SUCH A WAY THAT IT DOESN'T STORE N_QUBITS (WE ALREADY KNOW IT BEFOREHAND).
THEN I HAVE TO COPY FROM THE OLD PROJECT THE FAST KERNEL AND COMPUTE A G MATRIX,
TESTING THE ELAPSED TIME.
*/