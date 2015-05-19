#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define X_GRID_SIZE 20
#define T_GRID_SIZE 800

#define H (X_UPPER_BOUND - X_LOWER_BOUND) / (X_GRID_SIZE - 1.0)    
#define TAU (T_UPPER_BOUND - T_LOWER_BOUND) / (T_GRID_SIZE - 1.0)

#define SIGMA TAU / ( H * H )

#define A 1.0
#define B -9.0
#define C 1.0
#define LAMBDA 1.0

#define X_LOWER_BOUND 0.0
#define X_UPPER_BOUND 1.0

#define T_LOWER_BOUND 0.0
#define T_UPPER_BOUND 1.0

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
    }
}

double exact_solution_function(double x, double t) 
{
	return 1 / sqrt( C * exp( - ( 2 * LAMBDA * ( x + LAMBDA * t) ) / A) - B / ( 3 * LAMBDA) ) ; 
}

double** calculate_exact_solution_matrix() 
{
	double** exact_solution;
	create_matrix(&exact_solution, T_GRID_SIZE, X_GRID_SIZE);

	for (int k = 0; k < T_GRID_SIZE; ++k)
		for (int i = 0; i < X_GRID_SIZE; ++i)
			exact_solution[k][i] = exact_solution_function(X_LOWER_BOUND + i * H, T_LOWER_BOUND + k * TAU);

	return exact_solution;
}

double first_difference(double previous, double next) 
{
	return (next - previous) / (2.0 * H);
}

double second_difference(double previous, double current, double next) 
{	
	return (previous - 2.0 * current + next) / (H * H);
}	

void set_initial_conditions(double** matrix)
{
	for (int i = 0; i < X_GRID_SIZE; ++i)
		matrix[0][i] = exact_solution_function(X_LOWER_BOUND + i * H, T_LOWER_BOUND);
}

void set_boundary_conditions(double** matrix) 
{
	for (int k = 0; k < T_GRID_SIZE; ++k)
	{
		matrix[k][0] = exact_solution_function(X_LOWER_BOUND, T_LOWER_BOUND + k * TAU);
		matrix[k][X_GRID_SIZE-1] = exact_solution_function(X_UPPER_BOUND, T_LOWER_BOUND + k * TAU);
	}
}

double calculate_next_layer_point(double previous, double current, double next, double x)
{
	return current + TAU * ( A * second_difference(previous, current, next) 
			+ B * current * current * first_difference(previous, next) );
}

double** calculate_numerical_result()
{
	double** grid;
	create_matrix(&grid ,T_GRID_SIZE, X_GRID_SIZE + 1);

	set_initial_conditions(grid);
	set_boundary_conditions(grid);

	double x = 0;

	for (int k = 0; k < T_GRID_SIZE - 1; ++k)
		for (int i = 1; i < X_GRID_SIZE - 1; ++i)
			grid[k+1][i] = calculate_next_layer_point(grid[k][i-1], grid[k][i], grid[k][i+1], x);

	return grid;
}

double** calculate_exact_result() 
{
	double** exact;
	create_matrix(&exact, T_GRID_SIZE, X_GRID_SIZE + 1);

	for (int k = 0; k < T_GRID_SIZE; ++k)
		for (int i = 0; i < X_GRID_SIZE; ++i)
			exact[k][i] = exact_solution_function(X_LOWER_BOUND + i * H, T_LOWER_BOUND + k * TAU);

	return exact;
}

double** calculate_errors(double** numerical_result, double** exact_result) 
{
	double** errors;
	create_matrix(&errors ,T_GRID_SIZE, X_GRID_SIZE);

	for (int k = 0; k < T_GRID_SIZE; ++k)
		for (int i = 0; i < X_GRID_SIZE; ++i)
			errors[k][i] = fabs(exact_result[k][i] - numerical_result[k][i]) / exact_result[k][i] * 100;
	return errors;
}

double calculate_average_error(double** matrix) 
{
	double sum, average;

	for (int k = 1; k < T_GRID_SIZE; ++k)
		for (int i = 1; i < X_GRID_SIZE-1; ++i)
			sum += matrix[k][i];
	average = sum / ((T_GRID_SIZE-1) * (X_GRID_SIZE-2));

	return average;
}

int main(int argc, char const *argv[])
{

	printf("Sigma: %f\n", SIGMA);
	if (SIGMA > 0.5	) 
	{
		printf("Sigma must be not greater than 0.5 \n");
		exit(1);
	}

	double** numerical_result = calculate_numerical_result();

	double** exact_result = calculate_exact_result();
	double** errors = calculate_errors(numerical_result, exact_result);
	double avg = calculate_average_error(errors);

	printf("\nNUMERICAL\n\n\n");
	// print_matrix(numerical_result, T_GRID_SIZE, X_GRID_SIZE);

	printf("\nEXACT\n\n\n");
	// print_matrix(exact_result, T_GRID_SIZE, X_GRID_SIZE);

	printf("\nERRORS\n\n\n");
	// print_matrix(errors, T_GRID_SIZE, X_GRID_SIZE);

	printf("\nAVERAGE ERROR: %lf\n\n", avg);

	free_matrix(&numerical_result, T_GRID_SIZE);
	free_matrix(&exact_result, T_GRID_SIZE);
	free_matrix(&errors, T_GRID_SIZE);

	return 0;

}