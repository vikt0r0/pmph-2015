#include <stdio.h>

void matrix_transpose_seq(float *out[], float *in[], int m, int n) {
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            out[j][i] = in[i][j];
}

int main(int argc, char *argv[]) {
    // Generate a 1024 by 1024 matrix
    float in[1024][1024], transposed[1024][1024];

    return 0;
}

