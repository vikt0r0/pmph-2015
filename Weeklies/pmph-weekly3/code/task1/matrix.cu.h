#ifndef MATRIX_TRANSPOSE
#define MATRIX_TRANSPOSE

void matrix_generate_random(float *out[], int m, int n, float min, float max);
void matrix_transpose_seq(float out[][], float in[][], int m, int n);
void matrix_transpose_omp(float out[][], float in[][], int m, int n);
void matrix_transpose_cuda_naive(float out[][], float in[][], int m, int n);
void matrix_transpose_cuda_tiled(float out[][], float in[][], int m, int n);

#endif // MATRIX_TRANSPOSE
