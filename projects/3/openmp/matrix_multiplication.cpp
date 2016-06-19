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
		std::cout << "USAGE: matrix_multiplication <dim of matrix> <num of threads>" << std::endl;
		return -1;
	}

	int n = atoi(argv[1]);
	int num_thread = atoi(argv[2]);
	int sqrt_num_thread = int(sqrt(num_thread));
	
	srand (time(NULL));

	//initalize the matrix randomly.
	float** A = new float*[n];
	float** B = new float*[n];
	float** C = new float*[n];

	omp_set_num_threads(sqrt_num_thread);
	#pragma omp parallel 
	{
		int id_x = omp_get_thread_num();
		for (int i = id_x; i < n; i = i + sqrt_num_thread)
		{
			A[i] = new float[n];
			B[i] = new float[n];
			C[i] = new float[n];
			
			omp_set_num_threads(sqrt_num_thread);
			#pragma omp parallel 
			{
				int id_y = omp_get_thread_num();
				for (int j = id_y; j < n; j = j + sqrt_num_thread)
				{
					A[i][j] = float(rand() % 100) / 1.0 - 50;
					B[i][j] = float(rand() % 100) / 0.9 - 50;
					C[i][j] = 0;
				}
			}
		}
	}

//	print_matrix(A, n);
//	print_matrix(B, n);
	
	omp_set_num_threads(sqrt_num_thread);
	#pragma omp parallel
	{
		int id_x = omp_get_thread_num();
		for (int i = id_x; i < n; i = i + sqrt_num_thread)
		{
			omp_set_num_threads(sqrt_num_thread);
			#pragma omp parallel
			{
				int id_y = omp_get_thread_num();
				for (int j = id_y; j < n; j = j + sqrt_num_thread)
				{
					for (int k = 0; k < n; ++k)
					{
						C[i][j] += A[i][k] * B[k][j];
					}
				}
			}
		}
	}
		
//	print_matrix(C, n);

	gettimeofday(&end, NULL);	
	double time_gap = (end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec;
	printf("Time cost: %.2lf s.\n", time_gap / 100000);  	

	return 0;	
}
