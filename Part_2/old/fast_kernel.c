#include <stdint.h>

//     float ***dataset,
//     uint16_t ***indices,
//     uint32_t **n_indices,

typedef struct {
    uint16_t z_mask;
    uint16_t x_mask;
    int8_t y_count;
    int8_t phase;
} PauliTerm;

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