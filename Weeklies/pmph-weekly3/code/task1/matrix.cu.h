#ifndef _MATRIX
#define _MATRIX

#define MATRIX_EQUALITY_EPSILON 0.00005

template <class T>
struct matrix_t {
  int height;
  int width;
  T *elements;
};

template <typename T>
void matrix_set_element(matrix_t<T> mat, int i, int j, T val);
template <typename T>
T matrix_get_element(matrix_t<T> mat, int i, int j);
template <typename T>
void matrix_generate_random(matrix_t<T> mat, float min, float max);
template <typename T>
void matrix_transpose_seq(matrix_t<T> out, const matrix_t<T> in);
template <typename T>
void matrix_transpose_omp(matrix_t<T> out, const matrix_t<T> in);
template <typename T>
void matrix_transpose_cuda_naive(const unsigned int block_size, matrix_t<T> out, const matrix_t<T> in);
template <typename T>
void matrix_transpose_cuda_tiled(const unsigned int block_size, matrix_t<T> out, const matrix_t<T> in);

#endif // _MATRIX
