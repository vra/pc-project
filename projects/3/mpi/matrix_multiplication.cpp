#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <mpi.h>

void print_matrix(float* states, int n)
{
	std::cout << "matrix:" << std::endl;
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			std::cout << states[i*n+j] << " ";
		}
		std::cout << std::endl;
	}
}

int main(int argc, char** argv)
{
	int num_thread = 0;
	int n = 0;
	int num_gen = 0;

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
			std::cout << "USAGE: matrix_multiplication <dim of matrix>" << std::endl;
			return -1;
		}

		n = atoi(argv[1]);
	}

	//Broacast the variables
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	//NOTE: malloc and initalize must do for all threads!	
	float* A = new float[n * n];
	float* B = new float[n * n];
	float* C = new float[n * n];

	if (my_id == 0)
	{
		//initalize the matrix randomly.
		srand (time(NULL));
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				A[i * n + j] = 1 + (rand()) % 3;
				B[i * n + j] = 1 + (rand()) % 3;
				C[i * n + j] = 0;
			}
		}
	
		//print_matrix(A, n);
		//print_matrix(B, n);
	}

	
	MPI_Bcast(A, n * n, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(B, n * n, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(C, n * n, MPI_INT, 0, MPI_COMM_WORLD);

	int sqrt_p = sqrt(num_thread);
	int s_size = n / sqrt_p;
	
	//record the positon of each block of A and B, so that 
	//we don't have to move A or B, we just product partial A and B according to these variables
	int* x_pos_A = new int[num_thread];
	int* y_pos_A = new int[num_thread];
	int* x_pos_B = new int[num_thread];
	int* y_pos_B = new int[num_thread];
	
	//Initalize position values use each thread.
	x_pos_A[my_id] = my_id / sqrt_p;
	y_pos_A[my_id] = my_id % sqrt_p;
	x_pos_B[my_id] = my_id / sqrt_p;
	y_pos_B[my_id] = my_id % sqrt_p;
	MPI_Barrier(MPI_COMM_WORLD);	
//	printf("my_id: %d, x_pos_A: %d, y_pos_A: %d\n", my_id, x_pos_A[my_id], y_pos_A[my_id]);

	//looply move each block
	for (int k = 0; k < sqrt_p; ++k)
	{
		if (x_pos_A[my_id] > k)
		{
			y_pos_A[my_id] = (y_pos_A[my_id] + 1) % sqrt_p;		
		}
		if (y_pos_B[my_id] > k)
		{
			x_pos_B[my_id] = (x_pos_B[my_id] + 1) % sqrt_p;		
		}
	}	
	
	//calculate the product and move one step
	for (int k = 0; k < sqrt_p; ++k)
	{
		//Ci,j = Ci,j + Ai,j * Bi,j
		int i_A = x_pos_A[my_id];
		int j_A = y_pos_A[my_id];
		int i_B = x_pos_B[my_id];
		int j_B = y_pos_B[my_id];
		
		for (int iA = i_A * s_size; iA < (i_A+1) * s_size; ++iA)
		{
			for (int jB = j_B * s_size; jB < (i_B+1) * s_size; ++jB)
			{
				for (int m = 0; m < s_size; ++m)
				{
					C[jB * n + iA] += A[iA * n + (m + j_A * s_size)] * B[ (m + i_B*s_size)* n + jB];
				}
				
//				printf("======id: %d, C[%d, %d]= %d===========\n", my_id, jB, iA, C[jB * n + iA]);
			}
		}
		
		y_pos_A[my_id] = (y_pos_A[my_id] + 1) % sqrt_p;
		x_pos_B[my_id] = (x_pos_B[my_id] + 1) % sqrt_p;		
	}

	MPI_Barrier(MPI_COMM_WORLD);

	
	if (my_id ==0) 
		//print_matrix(C, n);
	if (my_id == 0)
	{
	
		//print_matrix(C, n);
		end_time = MPI_Wtime();
		std::cout << "*****Time cost: " << end_time - start_time << std::endl;
	}

	MPI_Finalize();
	return 0;	
}
