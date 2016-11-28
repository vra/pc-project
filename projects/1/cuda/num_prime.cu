#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>

//code that does the main job on GPU
__global__ void countNumOfPrimerKernel(int* n_array_d, bool* is_prime_d)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int i = row * blockDim.x + col + 2;
	
//	printf("gridDim: %d,block.x: %d, block.y: %d, thread.x: %d, thread:y: %d,\
// col: %d, row: %d, i: %d\n",gridDim.x, blockIdx.x, blockIdx.y, threadIdx.x,\
// threadIdx.y, col, row, i);

	bool has_factor = false;
	for (int j = 2; j < i; ++j)
	{
		if (i % j == 0)
		{
			has_factor = true;
			break;
		}	
	}
	if (!has_factor)
	{
		is_prime_d[i] = true;
	}
	
}

int main(int argc, char** argv)
{
	struct timeval start, end;
	gettimeofday(&start, NULL);
	if (argc != 3)
	{
		std::cout << "USAGE: num_prime <number of blocks> <integer>" 
				  << std::endl;
		return -1;
	}
	
	int num_blocks = atoi(argv[1]); 
	int n = atoi(argv[2]); 

	cudaSetDevice(0);
	
	int* n_array = new int[n];
	int* n_array_d = new int[n];
	bool* is_prime = new bool[n];
	bool* is_prime_d = new bool[n];

	for (int i = 0; i < n; ++i)
	{
		n_array[i] = i + 1;
		is_prime[i] = false;
	}	
	cudaMalloc((void**)&n_array_d, n * sizeof(int));
	cudaMemcpy(n_array_d, n_array, n * sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&is_prime_d, n * sizeof(bool));
	cudaMemcpy(is_prime_d, is_prime, n * sizeof(bool), cudaMemcpyHostToDevice);
	
	num_blocks = 1000 ;

	// use fixed number of thread in each block for convience
	int num_thread = n / num_blocks; 
	printf("num_blocks: %d, num_thread: %d\n", num_blocks, num_thread);
	dim3 dimBlock(num_thread, num_thread);
	dim3 dimGrid(num_blocks, 1);

	//countNumOfPrimerKernel<<<dimGrid, dimBlock>>> (n_array_d, is_prime_d);
	countNumOfPrimerKernel<<< dim3(num_blocks), dim3(num_thread)>>> \
		(n_array_d, is_prime_d);
	cudaThreadSynchronize();

	cudaMemcpy(is_prime, is_prime_d, n * sizeof(bool), cudaMemcpyDeviceToHost);
		
	int sum_num_prime = 0;
	for (int i = 0; i < n; ++i)
	{
		if (is_prime[i])
		{
			++sum_num_prime;
			std::cout << "prime: " << i << std::endl;
		}
	}
	
	std::cout << "Number of primes between 0 and " << n << " is: " 
			  << sum_num_prime << std::endl;
	
	
	cudaFree(n_array_d);
	cudaFree(is_prime_d);

	delete[] n_array;
	delete[] is_prime; 

	gettimeofday(&end, NULL);
	double time_gap = (end.tv_sec - start.tv_sec) * 1000000u 
					+ end.tv_usec - start.tv_usec;	
	printf("Time cost: %.2lf s.\n", time_gap / 100000);       

	return 0;
}
