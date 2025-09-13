#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex.h>
#include <inttypes.h>
#include <unistd.h>
//#define _POSIX_C_SOURCE 199309L
#include <time.h>

#define N_QUBITS 10


static inline int popcount64(uint64_t x) {
    return __builtin_popcountll(x);
}

double complex pauli_expectation(
    const uint64_t *indices,
    const double complex *data,
    int n_indices,
    uint64_t z_mask,
    uint64_t x_mask,
    double complex phase
) {
    double complex out = 0.0 + 0.0 * I;

    // number_of_Ys is taken mod 4 due to being an exponent to i.
    int number_of_Ys = popcount64(z_mask & x_mask) % 4;
    double complex y_factor;
    switch (number_of_Ys)
    {
    case 1:
        //y_factor = 0.0 + 1.0 * I;
        return 0.0 + 0.0 * I;
        //break;
    case 2:
        y_factor = -1.0 + 0.0 * I;
        break;
    case 3:
        return 0.0 + 0.0 * I;
        //y_factor = 0.0 - 1.0 * I;
        //break;
    default:
        y_factor = 1.0 + 0.0 * I;
        break;
    }

    //printf("%f + %f I\n", creal(y_factor), cimag(y_factor));

    //printf("Entering the for loop...\n");
    for (int k = 0; k < n_indices; k++)
    {
        // printf("k = %d\n", k);
        uint64_t i = indices[k];
        // printf("computed i\n");
        double complex amp_i = data[i];
        //printf("Successfully computed amp_i");
        uint64_t j = i ^ x_mask;
        double complex amp_j = data[j];
        //printf("Successfully computed amp_i and amp_j");
        if (amp_j != 0)
        {
            int parity = popcount64(i & z_mask) % 2;
            double complex phase_i = (parity ? -1.0 : 1.0) + 0.0 * I;

            // out += amp_i * conj(amp_j) * phase_i;
            out += amp_i * amp_j * phase_i;
        }
        

        // int parity = popcount64(i & z_mask) % 2;
        // double complex phase_i = (parity ? -1.0 : 1.0) + 0.0 * I;

        // out += amp_i * conj(amp_j) * phase_i;
    }
    

    return out * y_factor * phase;
}


double elapsed_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1.0e6;
}

int main(int argc, char const *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s inputfile\n", argv[0]);
        return 1;
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror("fopen"); return 1; }

    int n_nonzeros;
    fscanf(f, "%d", &n_nonzeros);

    uint64_t *indices = malloc(n_nonzeros * sizeof(uint64_t));
    for (int i = 0; i < n_nonzeros; i++) {
        fscanf(f, "%lu", &indices[i]);
    }
    //printf("Index = %ld\n", indices[n_nonzeros-1]);

    double _Complex *data = malloc((1ULL << N_QUBITS) * sizeof(double _Complex)); 
    //printf("%f\n", creal(data[1024]));



    for (int i = 0; i < (1ULL << N_QUBITS); i++) {
        double re, im;
        if (fscanf(f, "%lf %lf", &re, &im) != 2) break;
        data[i] = re + im * I;
    }

    uint64_t z_mask, x_mask;
    fscanf(f, "%lu", &z_mask);
    fscanf(f, "%lu", &x_mask);

    double re, im;
    fscanf(f, "%lf %lf", &re, &im);
    double _Complex phase = re + im * I;

    fclose(f);

    printf("Successfully read the file\n");



    clock_t t0 = clock();


    double complex result;
    for (size_t i = 0; i < 1e6; i++)
    {
        result = pauli_expectation(
        indices, data, n_nonzeros, z_mask, x_mask, phase);
    }
    
    // double _Complex result = pauli_expectation(
    //     indices, data, n_nonzeros, z_mask, x_mask, phase);

    clock_t t1 = clock();

    printf("Result: %.12f %.12f\n", creal(result), cimag(result));
    printf("Elapsed: %f s\n", (float)(t1 - t0) / CLOCKS_PER_SEC);

    free(indices);
    free(data);
    
    return 0;
}
