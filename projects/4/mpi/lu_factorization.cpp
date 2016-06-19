#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <mpi.h>

void print_matrix(float** states, int n)
{
	std::cout << "current matrix:" << std::endl;
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			std::cout << states[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

int main(int argc, char** argv)
{
	int n = 0;
	int num_thread = 0;
	
	int my_id = 0;
	double start_time;
	double end_time;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_thread);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
		
	if (my_id == 0)
	{
		start_time = MPI_Wtime();
		if (argc != 2)
		{
			std::cout << "USAGE: lu_factorization <dim of matrix> " << std::endl;
			return -1;
		}

		int n = atoi(argv[1]);
		int num_thread = atoi(argv[2]);
	}	

	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&num_thread, 1, MPI_INT, 0, MPI_COMM_WORLD);
	int sqrt_num_thread = sqrt(num_thread);

	//initalize the matrix randomly.
	float** A = new float*[n];
	float** L = new float*[n];
	float** U = new float*[n];
	
	if (my_id == 0)
	{
		for (int i = 0; i < n; ++i)
		{
			A[i] = new float[n];
			L[i] = new float[n];
			U[i] = new float[n];
				
			srand (time(NULL));
			for (int j = 0; j < n; ++j) 
			{
				A[i][j] = 1 + rand() % 5;
				L[i][j] = 0; 
				U[i][j] = 0;
		
				if (i==j)
				{
					L[i][j] = 1;
				}
			}
		}
	}

	//print_matrix(A, n);
	
	MPI_Bcast(A, n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(L, n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(U, n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

		for (int k = 0; k < n; ++k)
		{
			for (int i = k + 1; i < n; ++i)
			{
				L[i][k] = A[i][k] / A[k][k];
			}
			for (int j = k; j < n; ++j)	
			{
				U[k][j] = A[k][j];
			}
			for (int i = k + 1; i < n; ++i)
			{
				for (int j = k + 1; j < n; ++j)
				{
					A[i][j] = A[i][j] - L[i][k] * U[k][j];	
				}
			}
		}

	//print_matrix(L, n);
	//print_matrix(U, n);
	
	return 0;	
}
