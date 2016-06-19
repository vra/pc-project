#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>

void print_matrix(float* mat, int n)
{
	std::cout << "matrix:" << std::endl;
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			std::cout << mat[i * n + j] << " ";
		}
		std::cout << std::endl;
	}
}

//code do main the job on GPU
__global__ void matrixLUKernel(float* A_d, float* L_d, float* U_d, int n)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	printf("col:%d, row: %d\n", col, row);
	
	int k = col;
	
	for (int i = k+1; i < n; ++i)
	{
		L_d[i * n + k] = A_d[i *n + k] / A_d[k *n +k];
	}
	for (int j = k; j < n; ++j)
	{
		U_d[k *n + j] = A_d[k *n + j];
	}
	for (int i = k + 1; i < n; ++i)
	{
		for (int j = k + 1; j < n; ++j)
		{
			A_d[i * n + j] = A_d[i * n + j] - L_d[i * n + k] * U_d[k * n + j];
		}
	}
}


int main(int argc, char** argv)
{
	struct timeval start, end;
	gettimeofday(&start, NULL);

	if (argc != 3)
	{
		std::cout << "USAGE: lu_factorization <dim of matrix> <num of threads>" << std::endl;
		return -1;
	}
	
	int n = atoi(argv[1]); 
	int num_thread = atoi(argv[2]); 
	int num_block= n / num_thread; 

	cudaSetDevice(0);
	
	float* A= new float[n*n];
	float* A_d = NULL;
	float* L= new float[n*n];
	float* L_d = NULL;
	float* U= new float[n*n];
	float* U_d = NULL;

	srand (time(NULL));
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			A[i * n + j] = float(rand() % 100) / 3;
			L[i * n + j] = 0;
			U[i * n + j] = 0;

			if (i == j)
			{
				L[i * n + j] = 1;
			}
		}
				
	}	

	
	cudaMalloc((void**)&A_d, n * n * sizeof(float));
	cudaMalloc((void**)&L_d, n * n * sizeof(float));
	cudaMalloc((void**)&U_d, n * n * sizeof(float));

	cudaMemcpy(A_d, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(L_d, L, n * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(U_d, U, n * n * sizeof(float), cudaMemcpyHostToDevice);
	
	
//	printf("n: %d, num_block: %d, num_thread: %d\n", n, num_block, num_thread);
	//print_matrix(A, n);
	//print_matrix(B, n);

	matrixLUKernel <<< (num_block  num_thread)>>> (A_d, L_d, U_d, n);
	cudaThreadSynchronize();

	cudaMemcpy(L, L_d, n * n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(U, U_d, n * n * sizeof(float), cudaMemcpyDeviceToHost);

			
	
	cudaFree(A_d);
	cudaFree(L_d);
	cudaFree(U_d);

	delete[] A;
	delete[] L;
	delete[] U;

	//print_matrix(C, n);

	gettimeofday(&end, NULL);
	double time_gap = (end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec;	
	printf("Time cost: %.2lf s.\n", time_gap / 100000);       

	return 0;
}
