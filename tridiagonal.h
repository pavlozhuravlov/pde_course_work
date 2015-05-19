#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define allocate(type, size) (type*)malloc(sizeof(type)*size)
#define print_vector(vector, size, pattern) for(int i = 0; i < size; ++i) printf(pattern, vector[i]);

typedef struct _matrix{
  int      size;
  double*  b;
  double** A;
} matrix;

/**
* read_matrix
* @param  char*   filename - name of file with matrix
* @return matrix* pointer at the type matrix
* @brief reads matrix, the vector of free members and its size
*/
matrix* read_matrix(char * filename)
{
  FILE * file = fopen(filename, "r");

  if(file == NULL)
  {
    return NULL;
  }

  matrix* m = allocate(matrix, 1);
  
  fscanf(file, "%d", &(m->size));

  if(m->size < 2)
  {
    return NULL;
  }

  m->A = allocate(double*, m->size);

  for(int i = 0; i < m->size; ++i)
  {
    m->A[i] = allocate(double, m->size);

    for(int j = 0; j < m->size; ++j)
    {
      fscanf(file, "%lf", &(m->A[i][j]));
    }
  }

  m->b = allocate(double, m->size);
  for(int i = 0; i < m->size; ++i)
  {
    fscanf(file, "%lf", &(m->b[i]));
  }

  return m;
}

/**
* write_result
* @param char* filename - name of destination file
* @param double* result - vector of found results
* @param int size       - size of vector
* @return void
* @brief writes vector of results to file
*/
void write_result(char * filename, double* result, int size)
{
  FILE * file = fopen(filename, "w");

  for(int i = 0; i < size; ++i)
  {
    fprintf(file, "%lf\n", result[i]);
  }
}
void calculate_alphas_and_betas(matrix* mtr, double* alphas, double* betas, int p)
{
  double** A    = mtr->A;
  double*  b    = mtr->b;
  int      size = mtr->size;

  // calculate initial alpha and beta
  alphas[0] = 0;
  betas[0]  = 0;
  alphas[1] = - A[0][1] / A[0][0];
  betas[1]  = b[0] / A[0][0];

  double denominator;
  for(int i = 1; i <= p; ++i)
  {
    denominator = (A[i][i] + alphas[i]*A[i][i-1]);
    alphas[i + 1] = (-A[i][i+1]) / denominator;
    betas[i + 1]  = (b[i] - A[i][i-1]*betas[i]) / denominator;
  }
}

void calculate_xies_and_etas(matrix* mtr, double* xies, double* etas, int p)
{
  double** A    = mtr->A;
  double*  b    = mtr->b;
  int      size = mtr->size;

  // calculate initial alpha and beta
  xies[0] = 0;
  etas[0]  = 0;
  xies[size-1] = - A[size-1][size-2] / A[size-1][size-1];
  etas[size-1] = b[size-1] / A[size-1][size-1];

  double denominator;
  for(int i = size - 2; i >= p; --i)
  {
    denominator = (A[i][i] + xies[i+1]*A[i][i+1]);
    xies[i] = (-A[i][i-1]) / denominator;
    etas[i] = (b[i] - A[i][i+1]*etas[i+1]) / denominator;
  }
}

double* tridiagonalmatrix_parallel_solve(matrix* mtr)
{
  int p = mtr->size / 2 + (mtr->size % 2);
  double* alphas = allocate(double, mtr->size);
  double* betas  = allocate(double, mtr->size);
  double* xies   = allocate(double, mtr->size);
  double* etas   = allocate(double, mtr->size);
  double* xs     = allocate(double, mtr->size);

  #pragma omp parallel sections
  {
    #pragma omp section
    {
      calculate_alphas_and_betas(mtr, alphas, betas, p);
    }
    #pragma omp section
    {
      calculate_xies_and_etas(mtr, xies, etas, p);
    }
  }

  xs[p] = (alphas[p+1]*etas[p+1] + betas[p+1]) / (1 - alphas[p+1]*xies[p+1]);

  #pragma omp parallel sections
  {
    #pragma omp section
    {
      for(int i = p - 1; i >= 0; --i)
      {
        xs[i] = alphas[i+1] * xs[i+1] + betas[i+1];
      }
    }
    #pragma omp section
    {
      for(int i = p; i < mtr->size - 1; ++i)
      {
        xs[i + 1] = xies[i+1] * xs[i] + etas[i+1];
      }
    }
  }

  return xs;
}

double* tridiagonal_solve(double** m, double* v, int s)
{
	matrix* mtr = allocate(matrix, 1);
	mtr->size = s;

	mtr->A = allocate(double*, mtr->size);
	for (int i = 0; i < s; ++i)
		mtr->A[i] = allocate(double, mtr->size);
	mtr->A = m;
	
	mtr->b = allocate(double, mtr->size);
	mtr->b = v;

	return tridiagonalmatrix_parallel_solve(mtr);	
}