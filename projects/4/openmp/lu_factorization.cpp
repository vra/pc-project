#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <omp.h>
#include <sys/time.h>

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
	struct timeval start, end;
	gettimeofday(&start, NULL);

	if (argc != 3)
	{
		std::cout << "USAGE: lu_factorization <dim of matrix> <num of thread>" << std::endl;
		return -1;
	}

	int n = atoi(argv[1]);
	int num_thread = atoi(argv[2]);
	int sqrt_num_thread = sqrt(num_thread);
	
	srand (time(NULL));

	//initalize the matrix randomly.
	float** A = new float*[n];
	float** L = new float*[n];
	float** U = new float*[n];

	omp_set_num_threads(sqrt_num_thread);
	#pragma omp parallel
	{
		int id_x = omp_get_thread_num();
		
		for (int i = id_x; i < n; i = i + sqrt_num_thread)
		{
			A[i] = new float[n];
			L[i] = new float[n];
			U[i] = new float[n];
				
			omp_set_num_threads(sqrt_num_thread);
			#pragma omp parallel
			{
				int id_y = omp_get_thread_num();
				for (int j = id_y; j < n; j = j + sqrt_num_thread)
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
	}

	//print_matrix(A, n);
	
	omp_set_num_threads(num_thread);
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		for (int k = id; k < n; k = k + num_thread)
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
	}

	//print_matrix(L, n);
	//print_matrix(U, n);
	
	gettimeofday(&end, NULL);	
	double time_gap = (end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec;
	printf("Time cost: %.2lf s.\n", time_gap / 100000);  
	return 0;	
}
