#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "omp.h"
#include "tridiagonal.h"

#define X_LOWER_BOUND 0.0
#define X_UPPER_BOUND 1.0

#define T_LOWER_BOUND 0.0
#define T_UPPER_BOUND 1.0

#define X_GRID_SIZE 100
#define T_GRID_SIZE 100

#define H (X_UPPER_BOUND - X_LOWER_BOUND) / (X_GRID_SIZE - 1.0)    
#define TAU (T_UPPER_BOUND - T_LOWER_BOUND) / (T_GRID_SIZE - 1.0)  
#define SIGMA TAU / ( H * H )

#define A 1.0
#define B -9.0
#define C 1.0
#define LAMBDA 1.0

#define A_COEF(omega) A/(H*H) - B * omega * omega / (2.0 * H)
#define C_COEF -2.0*A/(H*H) - 1.0/TAU
#define B_COEF(omega) A/(H*H) + B * omega * omega / (2.0 * H)
#define F_COEF -1.0/TAU

// MATRIX OPERATIONS
void create_matrix(double ***array, int rows, int cols); 
void free_matrix(double ***array, int rows); 
void print_matrix(double** matrix , int rows, int cols); 

// WRITE FORMULAS HERE
double exact_solution_point(double x, double t); 

double** calculate_numerical_result(); // PARALLEL
double** calculate_exact_result(); // SEQUENTIAL
	
double** calculate_errors(double** numerical_result, double** exact_result); // SEQUENTIAL
double calculate_average_error(double** matrix); // SEQUENTIAL


int main(int argc, char **argv)
{	
    double** numerical_result = calculate_numerical_result();
	double** exact_result = calculate_exact_result();
	double** errors = calculate_errors(numerical_result, exact_result);
	double avg = calculate_average_error(errors);

	printf("\nNUMERICAL\n\n\n");
	print_matrix(numerical_result, T_GRID_SIZE, X_GRID_SIZE);

	printf("\nEXACT\n\n\n");
	print_matrix(exact_result, T_GRID_SIZE, X_GRID_SIZE);

	printf("\nERRORS\n\n\n");
	print_matrix(errors, T_GRID_SIZE, X_GRID_SIZE);

	printf("\nAVERAGE ERROR: %lf\n\n", avg);

	free_matrix(&numerical_result, T_GRID_SIZE);
	free_matrix(&exact_result, T_GRID_SIZE);
	free_matrix(&errors, T_GRID_SIZE);

	return 0;
}


double exact_solution_point(double x, double t) 
{
	return 1 / sqrt( C * exp( - ( 2 * LAMBDA * ( x + LAMBDA * t) ) / A) - B / ( 3 * LAMBDA) ) ; 
}


void create_matrix(double ***array, int rows, int cols) 
{
	*array = (double**)malloc(sizeof(double) * rows);
	for (int i = 0; i < rows; ++i)
		(*array)[i] = calloc(cols, sizeof(double));
}

void free_matrix(double ***array, int rows) 
{
	for (int i = 0; i < rows; ++i)
		free((*array)[i]);
	free(*array);
}

void print_matrix(double** matrix , int rows, int cols) 
{
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++)
            printf("%lf   ", matrix[i][j]);
        printf("\n\n");
        if (i == 1)
        	break;
    }
}

double** calculate_numerical_result()
{
	double** numerical_result; 
	double** tridiagonal_matrix;

	int tridiagonal_size = X_GRID_SIZE - 2;

	double* right_vector = malloc(sizeof(double) * tridiagonal_size);
	double* layer_result = malloc(sizeof(double) * tridiagonal_size);

	create_matrix(&numerical_result, T_GRID_SIZE, X_GRID_SIZE);
	create_matrix(&tridiagonal_matrix, tridiagonal_size, tridiagonal_size);

	for (int i = 0; i < X_GRID_SIZE; ++i)
		numerical_result[0][i] = exact_solution_point(X_LOWER_BOUND + H * i, T_LOWER_BOUND);

	for (int k = 1; k < T_GRID_SIZE; ++k)
	{
		numerical_result[k][0] = exact_solution_point(X_LOWER_BOUND, T_LOWER_BOUND + TAU * k);
		numerical_result[k][X_GRID_SIZE-1] = exact_solution_point(X_UPPER_BOUND, T_LOWER_BOUND + TAU * k);
	}

	for (int k = 0; k < T_GRID_SIZE - 1; ++k)
	{
		for (int i = 0; i < tridiagonal_size; ++i)
		{
			if (i != 0)
				tridiagonal_matrix[i][i-1] = A_COEF(numerical_result[k][i+1]);
			tridiagonal_matrix[i][i] = C_COEF;
			if (i != tridiagonal_size-1)
				tridiagonal_matrix[i][i+1] = B_COEF(numerical_result[k][i+1]);
			
			right_vector[i] = F_COEF * numerical_result[k][i];
			if (i == 0 || i == tridiagonal_size-1)
				right_vector[i] += (- A_COEF(numerical_result[k][i+1])) * numerical_result[k][i+1];
		}

		// if (k == 0) 
		// {
		// 		printf("\nDIAGONAL\n\n\n");
		// 		print_matrix(tridiagonal_matrix, tridiagonal_size, tridiagonal_size);
		// }

		layer_result = tridiagonal_solve(tridiagonal_matrix, right_vector, tridiagonal_size);

		for (int i = 1; i < X_GRID_SIZE-1; ++i)
			numerical_result[k+1][i] = layer_result[i-1];
	}
	free_matrix(&tridiagonal_matrix, tridiagonal_size);
	return numerical_result;
	
}

double** calculate_exact_result() 
{
	double** exact;
	create_matrix(&exact, T_GRID_SIZE, X_GRID_SIZE);
#pragma omp parallel for
	for (int k = 0; k < T_GRID_SIZE; ++k)
		for (int i = 0; i < X_GRID_SIZE; ++i)
			exact[k][i] = exact_solution_point(X_LOWER_BOUND + i * H, T_LOWER_BOUND + k * TAU);

	return exact;
}

double** calculate_errors(double** numerical_result, double** exact_result) 
{
	double** errors;
	create_matrix(&errors, T_GRID_SIZE, X_GRID_SIZE);

#pragma omp parallel for
	for (int k = 0; k < T_GRID_SIZE; ++k)
		for (int i = 0; i < X_GRID_SIZE; ++i)
			errors[k][i] = 100 * fabs(exact_result[k][i] - numerical_result[k][i]) / exact_result[k][i];
	return errors;
}

double calculate_average_error(double** matrix) 
{
	double sum, average;

#pragma omp parallel for reduction(+:sum)
	for (int k = 1; k < T_GRID_SIZE; ++k)
		for (int i = 1; i < X_GRID_SIZE-1; ++i)
			sum += matrix[k][i];
	average = sum / ((T_GRID_SIZE-1) * (X_GRID_SIZE-2));

	return average;
}