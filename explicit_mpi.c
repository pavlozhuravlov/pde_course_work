#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"

#define X_LOWER_BOUND 0.0
#define X_UPPER_BOUND 1.0

#define T_LOWER_BOUND 0.0
#define T_UPPER_BOUND 1.0

#define X_GRID_SIZE 10
#define T_GRID_SIZE 300

#define H (X_UPPER_BOUND - X_LOWER_BOUND) / (double) (X_GRID_SIZE)    
#define TAU (T_UPPER_BOUND - T_LOWER_BOUND) / (double) T_GRID_SIZE   

#define SIGMA TAU / ( H * H )

#define A 1.0
#define B -9.0
#define C 1.0
#define LAMBDA 1.0

// MATRIX OPERATIONS
void create_matrix(double ***array, int rows, int cols); 
void free_matrix(double ***array, int rows); 
void print_matrix(double** matrix , int rows, int cols); 

// WRITE FORMULAS HERE
double exact_solution_function(double x, double t); 
double calculate_next_layer_point(double previous, double current, double next, double x);

// APPROXIMATIONS FOR DERIVATIVES
double first_difference(double previous, double next); 
double second_difference(double previous, double current, double next);

double** calculate_numerical_result(int rank, int comm_size); //PARALLEL
double** calculate_exact_result(); // SEQUENTIAL

double** calculate_errors(double** numerical_result, double** exact_result); // SEQUENTIAL
double calculate_average_error(double** matrix); // SEQUENTIAL


int main(int argc, char **argv)
{

	int rank, comm_size;        

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double** numerical_result = calculate_numerical_result(rank, comm_size);

    if (rank == 0)
    {
		double** exact_result = calculate_exact_result();
		double** errors = calculate_errors(numerical_result, exact_result);
		double avg = calculate_average_error(errors);

		printf("\nNUMERICAL\n\n\n");
		print_matrix(numerical_result, T_GRID_SIZE, X_GRID_SIZE + 1);

		printf("\nEXACT\n\n\n");
		print_matrix(exact_result, T_GRID_SIZE, X_GRID_SIZE + 1);

		printf("\nERRORS\n\n\n");
		print_matrix(errors, T_GRID_SIZE, X_GRID_SIZE + 1);

		printf("\nAVERAGE ERROR: %lf\n\n", avg);

		free_matrix(&numerical_result, T_GRID_SIZE);
		free_matrix(&exact_result, T_GRID_SIZE);
		free_matrix(&errors, T_GRID_SIZE);
    }

	MPI_Finalize();
	return 0;
}


double exact_solution_function(double x, double t) 
{
	return 1 / sqrt( C * exp( - ( 2 * LAMBDA * ( x + LAMBDA * t) ) / A) - B / ( 3 * LAMBDA) ) ; 
}

double calculate_next_layer_point(double previous, double current, double next, double x)
{
	return current + TAU * ( A * second_difference(previous, current, next) 
			+ B * current * current * first_difference(previous, next) );
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
        if (i == 10)
        	break;
    }
}

double first_difference(double previous, double next) 
{
	return (next - previous) / (2.0 * H);
}

double second_difference(double previous, double current, double next) 
{	
	return (previous - 2.0 * current + next) / (H * H);
}	

double** calculate_numerical_result(int rank, int comm_size)
{
	double** local_grid;
	int local_grid_width = X_GRID_SIZE / comm_size;

	if (rank == comm_size - 1)
		local_grid_width += (X_GRID_SIZE % comm_size) + 1;

    create_matrix(&local_grid, T_GRID_SIZE, local_grid_width);

    double x;
	for (int i = 0; i < local_grid_width; ++i)
	{
		if (rank == comm_size - 1)
			x = X_UPPER_BOUND - H * (local_grid_width - 1) + H * i;
		else 
			x = X_LOWER_BOUND + rank * H * local_grid_width + H * i;
		local_grid[0][i] = exact_solution_function(x, T_LOWER_BOUND);
	}

	for (int k = 0; k < T_GRID_SIZE; ++k)
	{
		if (rank == 0)
			local_grid[k][0] = exact_solution_function(X_LOWER_BOUND, T_LOWER_BOUND + k * TAU);
		else if (rank == comm_size - 1)
			local_grid[k][local_grid_width-1] = exact_solution_function(X_UPPER_BOUND, T_LOWER_BOUND + k * TAU);
	}

	double previous, next;
	x = 0;

	for (int k = 0; k < T_GRID_SIZE - 1; ++k)
	{
		if (rank != comm_size - 1)
			MPI_Send(&(local_grid[k][local_grid_width - 1]), 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
		if (rank != 0)
			MPI_Send(&(local_grid[k][0]), 1, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD);

		if (rank != 0)
			MPI_Recv(&(previous), 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (rank != comm_size - 1)
			MPI_Recv(&(next), 1, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE	);

		if (rank != comm_size - 1)
		{
			// For Valik function
			// x = X_LOWER_BOUND + rank * H * local_grid_width + H * local_grid_width-1;
			local_grid[k+1][local_grid_width - 1] = calculate_next_layer_point(local_grid[k][local_grid_width-2], local_grid[k][local_grid_width-1], next, x);
		}

		for (int i = 1; i < local_grid_width - 1; ++i)
		{
			// For Valik function
			// if (rank == comm_size - 1)
			// 	x = X_UPPER_BOUND - H * (local_grid_width - 1) + H * i;
			// else 
			// 	x = X_LOWER_BOUND + rank * H * local_grid_width + H * i;
			local_grid[k+1][i] = calculate_next_layer_point(local_grid[k][i-1], local_grid[k][i], local_grid[k][i+1], x);
		}
		
		if (rank != 0)
		{
			// For Valik function
			// if (rank == comm_size - 1)
			// 	x = X_UPPER_BOUND - H * (local_grid_width - 1);
			// else 
			// 	x = X_LOWER_BOUND + rank * H * local_grid_width;
			local_grid[k+1][0] = calculate_next_layer_point(previous, local_grid[k][0], local_grid[k][1], x);
		}
	}


	double** global_grid = local_grid;
	if (rank == 0)
		create_matrix(&global_grid, T_GRID_SIZE, X_GRID_SIZE + 1);
	
	int* counts = malloc(sizeof(int) * comm_size);
	int* displacements = malloc(sizeof(int) * comm_size);

	counts[0] = X_GRID_SIZE / comm_size;
	displacements[0] = 0;
	for (int i = 1; i < comm_size - 1; ++i)
	{
		counts[i] = X_GRID_SIZE / comm_size;
		displacements[i] = displacements[i-1] + counts[i];
	}
	counts[comm_size - 1] = counts[comm_size - 2] + (X_GRID_SIZE % comm_size) + 1;
	displacements[comm_size -1] = displacements[comm_size - 2] + counts[comm_size - 2];

	for (int k = 0; k < T_GRID_SIZE; ++k)
		MPI_Gatherv(&(local_grid[k][0]), counts[rank], MPI_DOUBLE, 
				&(global_grid[k][0]), counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// for (int i = 0; i < comm_size; ++i)
	// {
	// 	if (rank == i)
	// 	{
	// 		printf("\nLOCAL GRID: RANK #%d\n\n", rank);
	// 		print_matrix(local_grid, T_GRID_SIZE, local_grid_width);
	// 	}
	// 	MPI_Barrier(MPI_COMM_WORLD);
	// }

	free(counts);
	free(displacements);
	free_matrix(&local_grid, T_GRID_SIZE);

	return global_grid;
}

double** calculate_exact_result() 
{
	double** exact;
	create_matrix(&exact, T_GRID_SIZE, X_GRID_SIZE + 1);

	for (int k = 0; k < T_GRID_SIZE; ++k)
		for (int i = 0; i < X_GRID_SIZE + 1; ++i)
			exact[k][i] = exact_solution_function(X_LOWER_BOUND + i * H, T_LOWER_BOUND + k * TAU);

	return exact;
}

double** calculate_errors(double** numerical_result, double** exact_result) 
{
	double** errors;
	create_matrix(&errors ,T_GRID_SIZE, X_GRID_SIZE);

	for (int k = 0; k < T_GRID_SIZE; ++k)
		for (int i = 0; i < X_GRID_SIZE + 1; ++i)
			errors[k][i] = fabs(exact_result[k][i] - numerical_result[k][i]) / exact_result[k][i] * 100;
	return errors;
}

double calculate_average_error(double** matrix) 
{
	double sum, average;

	for (int k = 0; k < T_GRID_SIZE; ++k)
		for (int i = 0; i < X_GRID_SIZE + 1; ++i)
			sum += matrix[k][i];
	average = sum / (T_GRID_SIZE * X_GRID_SIZE + 1);

	return average;
}