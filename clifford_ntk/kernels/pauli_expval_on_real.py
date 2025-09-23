from numba import njit

@njit(inline='always')
def popcount(x):
    # Kernighan’s algorithm
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c

@njit
def fast_kernel(
    indices, vector, n_indices,
    z_mask, x_mask, count_y, phase
):
    out = 0.0

    parity_y = count_y % 4
    if parity_y % 2 == 1:
        return 0.0
    # if parity_y == 1:
    #     phase_y = 1.0j
    # elif parity_y == 3:
    #     phase_y = -1.0j
    elif parity_y == 0:
        phase_y = 1.0
    elif parity_y == 2:
        phase_y = -1.0

    for k in range(n_indices):
        i = indices[k]
        amp_i = vector[i]

        j = i ^ x_mask
        amp_j = vector[j]

        parity = popcount(i & z_mask) % 2
        phase_i = -1.0 if parity else 1.0

        out += amp_i * amp_j * phase_i

    return out * phase_y * phase