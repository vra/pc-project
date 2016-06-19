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
__global__ void matrixProductKernel(float* A_d, float* B_d, float* C_d, int n)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	float Pvalue = 0;
	
	for (int k = 0; k < n; ++k)
	{
		Pvalue += A_d[row * n + k] * B_d[k * n + col];
	}
	
	C_d[row * n + col] = Pvalue;
}


int main(int argc, char** argv)
{
	struct timeval start, end;
	gettimeofday(&start, NULL);

	if (argc != 3)
	{
		std::cout << "USAGE: matrix_multiplication <dim of matrix> <num of blocks>" << std::endl;
		return -1;
	}
	
	int n = atoi(argv[1]); 
	int num_block = atoi(argv[2]); 
	int num_thread = n / num_block; 

	cudaSetDevice(0);
	
	float* A= new float[n*n];
	float* A_d = NULL;
	float* B= new float[n*n];
	float* B_d = NULL;
	float* C= new float[n*n];
	float* C_d = NULL;

	srand (time(NULL));
	for (int i = 0; i < n * n; ++i)
	{
		A[i] = float(rand() % 100) / 3;
		B[i] = float(rand() % 100) / 7;
		C[i] = 0;
	}	

	
	cudaMalloc((void**)&A_d, n * n * sizeof(float));
	cudaMalloc((void**)&B_d, n * n * sizeof(float));
	cudaMalloc((void**)&C_d, n * n * sizeof(float));

	cudaMemcpy(A_d, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, n * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C_d, C, n * n * sizeof(float), cudaMemcpyHostToDevice);
	
	
	printf("n: %d, num_block: %d, num_thread: %d\n", n, num_block, num_thread);
	//print_matrix(A, n);
	//print_matrix(B, n);

	matrixProductKernel<<< dim3(num_block, num_block), dim3(num_thread, num_thread)>>> (A_d, B_d, C_d, n);
	cudaThreadSynchronize();

	cudaMemcpy(C, C_d, n * n * sizeof(float), cudaMemcpyDeviceToHost);

			
	
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

	delete[] A;
	delete[] B;
	delete[] C;

	//print_matrix(C, n);

	gettimeofday(&end, NULL);
	double time_gap = (end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec;	
	printf("Time cost: %.2lf s.\n", time_gap / 100000);       

	return 0;
}
