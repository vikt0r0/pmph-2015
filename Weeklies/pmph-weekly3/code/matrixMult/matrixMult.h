#ifndef MAT_MULT
#define MAT_MULT

#include <stdio.h>

typedef struct {
  int height;
  int width;
  float *elements;
} Matrix;

int matrixMult(Matrix *a, Matrix *b, Matrix *res);

#define MAT_MULT_STATUS_SUCCESS 0
#define MAT_MULT_STATUS_ERROR 1

#endif
