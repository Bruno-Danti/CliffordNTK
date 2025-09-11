#include <stdint.h>
#include <complex.h>

static inline int popcount64(uint64_t x) {
    return __builtin_popcount_ll(x);
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
        y_factor = 0.0 + 1.0 * I;
        break;
    case 2:
        y_factor = -1.0 + 0.0 * I;
        break;
    case 3:
        y_factor = 0.0 - 1.0 * I;
        break;
    default:
        y_factor = 1.0 + 0.0 * I;
        break;
    }

    for (int k = 0; k < n_indices; k++)
    {
        uint64_t i = indices[k];
        double complex amp_i = data[i];
        uint64_t j = i ^ x_mask;
        double complex amp_j = data[j];

        int parity = popcount64(i & z_mask) % 2;
        double complex phase_i = (parity ? -1.0 : 1.0) + 0.0 * I;

        out += amp_i * conj(amp_j) * phase_i;
    }
    

    return out * y_factor * phase;
}