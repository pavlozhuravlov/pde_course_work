#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define X_GRID_SIZE 4
#define T_GRID_SIZE 40

#define H 1.0 / (double) (X_GRID_SIZE - 1)    
#define TAU 1.0 / (double) T_GRID_SIZE   

#define SIGMA TAU / ( H * H )
	
#define A -1.0
#define B 1.0
#define C 1.0
#define LAMBDA -1.0

#define BETA sqrt( -B / A )
#define MYU sqrt( A * LAMBDA / 2.0)

#define X_LOWER_BOUND 0.0
#define X_UPPER_BOUND 1.0

#define T_LOWER_BOUND 0.0
#define T_UPPER_BOUND 1.0

// Init matrix
int create_matrix(double ***array, int rows, int cols) 
{
	*array = (double**) malloc(sizeof(double) * rows);
	for (int i = 0; i < rows; ++i)
	{
		(*array)[i] = calloc(cols, sizeof(double));
	}
}

// Clear matrix
int free_matrix(double ***array, int rows) 
{
	free(&(*array)[0][0]);
	free(*array);
	return 0;
}

// Show matrix
void print_matrix(double** matrix , int rows, int cols) 
{
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%lf   ", matrix[i][j]);
        }
        printf("\n\n");
    }
}

// Solve exact soluction
double exact_solution_function(double x, double t) 
{
	return (-2 / LAMBDA) * log( BETA + C * exp(MYU*x - 0.5*A*LAMBDA*t ) ) ; 
}

// Calculate exact solution matrix
double** calculate_exact_solution_matrix() 
{
	double** exact_solution;
	create_matrix(&exact_solution, T_GRID_SIZE, X_GRID_SIZE);

	for (int k = 0; k < T_GRID_SIZE; ++k)
	{
		for (int i = 0; i < X_GRID_SIZE; ++i)
		{
			exact_solution[k][i] = exact_solution_function(X_LOWER_BOUND + i * H, T_LOWER_BOUND + k * TAU);
		}
	}

	return exact_solution;
}

// First derivative
double first_difference(double** matrix, int k, int i) 
{
	return (matrix[k][i + 1] - matrix[k][i - 1]) / (2.0 * H);
}

// Second derivative
double second_difference(double** matrix, int k, int i) 
{	
	return (matrix[k][i - 1] - 2.0 * matrix[k][i] + matrix[k][i + 1]) / (H * H);
}	

// Initial conditions
void set_initial_conditions(double** matrix)
{
	for (int i = 0; i < X_GRID_SIZE; ++i)
	{
		matrix[0][i] = exact_solution_function(X_LOWER_BOUND + i * H, T_LOWER_BOUND);
	}
}

// Boundary conditions
void set_boundary_conditions(double** matrix) 
{
	for (int k = 0; k < T_GRID_SIZE; ++k)
	{
		matrix[k][0] = exact_solution_function(X_LOWER_BOUND, T_LOWER_BOUND + k * TAU);
		matrix[k][X_GRID_SIZE - 1] = exact_solution_function(X_UPPER_BOUND, T_LOWER_BOUND + k * TAU);
	}
}	

// Next layer point
double calculate_next_layer_point(double** matrix, int k, int i)
{
	return matrix[k][i] + TAU * ( second_difference(matrix, k, i) + A + B * exp ( LAMBDA * matrix[k][i] ) );
}

// Numerical solution
double** calculate_numerical_solution_matrix()
{
	double** numerical_solution;
	create_matrix(&numerical_solution ,T_GRID_SIZE, X_GRID_SIZE);

	set_initial_conditions(numerical_solution);
	set_boundary_conditions(numerical_solution);

	for (int k = 0; k < T_GRID_SIZE - 1; ++k)
	{
		for (int i = 1; i < X_GRID_SIZE - 1; ++i)
		{
			numerical_solution[k+1][i] = calculate_next_layer_point(numerical_solution, k, i);
		}
	}

	return numerical_solution;
}

double** calculate_residual_matrix(double** numerical_solution, double** exact_solution) 
{
	double** residual_matrix;
	create_matrix(&residual_matrix ,T_GRID_SIZE, X_GRID_SIZE);

	for (int k = 0; k < T_GRID_SIZE; ++k)
	{
		for (int i = 0; i < X_GRID_SIZE; ++i)
		{
			residual_matrix[k][i] = fabs(exact_solution[k][i] - numerical_solution[k][i]);
		}
	}

	return residual_matrix;
}

double** calculate_residual_percentage_matrix(double** residual, double** exact_solution) 
{
	double** residual_percentage_matrix;
	create_matrix(&residual_percentage_matrix ,T_GRID_SIZE, X_GRID_SIZE);

	for (int k = 0; k < T_GRID_SIZE; ++k)
	{
		for (int i = 0; i < X_GRID_SIZE; ++i)
		{
			residual_percentage_matrix[k][i] = residual[k][i] / exact_solution[k][i] * 100;
		}
	}

	return residual_percentage_matrix;
}

double calculate_average_matrix_value(double** matrix, int rows, int cols) 
{
	double sum, average;

	for (int k = 0; k < rows; ++k)
		for (int i = 0; i < cols; ++i)
			sum += matrix[k][i];
	average = sum / (rows * cols);

	return average;
}

// Print main matrix
int main(int argc, char const *argv[])
{

	printf("Sigma: %f\n", SIGMA);
	
	if (SIGMA > 0.5	) 
	{
		printf("Sigma must be not greater than 0.5 \n");
		exit(1);
	}

	double** numerical_solution = calculate_numerical_solution_matrix();
	double** exact_solution = calculate_exact_solution_matrix();
	double** residual = calculate_residual_matrix(numerical_solution, exact_solution);
	
	// double** procent_matrix = calculate_residual_percentage_matrix(residual, exact_solution);
	// double average = calculate_average_matrix_value(procent_matrix, T_GRID_SIZE, X_GRID_SIZE);

	printf("NUMERICAL\n");
	printf("__________________________________________\n\n\n");
	print_matrix(numerical_solution, T_GRID_SIZE, X_GRID_SIZE);
	printf("__________________________________________\n\n");

	printf("EXACT\n");
	printf("__________________________________________\n\n\n");
	print_matrix(exact_solution, T_GRID_SIZE, X_GRID_SIZE);
	printf("__________________________________________\n\n");

	printf("RESIDUAL\n");
	printf("__________________________________________\n\n\n");
	print_matrix(residual, T_GRID_SIZE, X_GRID_SIZE);
	printf("__________________________________________\n\n");

	// printf("PROCENT MATRIX\n");
	// printf("__________________________________________\n\n\n");
	// printf("%lf\n", average);
	// printf("__________________________________________\n\n");

	// printf("PROCENT MATRIX\n");
	// printf("__________________________________________\n\n\n");
	// print_matrix(procent_matrix, T_GRID_SIZE, X_GRID_SIZE);
	// printf("__________________________________________\n\n");

	free_matrix(&numerical_solution, T_GRID_SIZE);
	free_matrix(&exact_solution, T_GRID_SIZE);
	free_matrix(&residual, T_GRID_SIZE);

	return 0;

}