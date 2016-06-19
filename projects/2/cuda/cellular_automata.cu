#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>

void print_matrix(int* states, int n)
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

//code do main the job on GPU
__global__ void countActiveNeigb(int* states_d, int* active_neigb_d, int n)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int i = row * blockDim.x + col;
	
	int peso_i = i / n;
	int peso_j = i % n;	
	
	for (int x = peso_i - 1; x < peso_i + 2; ++x)
	{
		for (int y = peso_j - 1; y < peso_j + 2; ++y)
		{
			if (x >= 0 && y >= 0 && x < n && y < n)
			{
				active_neigb_d[i] += states_d[x * n + y];
			}	
		}
	}
	active_neigb_d[i] -= states_d[peso_i * n + peso_j];
//	printf("i: %d, peso_i: %d, peso_j: %d, states_d[i][j]: %d, active_neigb_d[i][j]: %d\n", i, peso_i, peso_j, states_d[i], active_neigb_d[i]);
}

__global__ void updateStates(int* states_d, int* active_neigb_d)
{
	
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int i = row * blockDim.x + col;

	if ( (states_d[i] == 1 && (active_neigb_d[i] == 2 || active_neigb_d[i] == 3)) || (states_d[i] == 0 && active_neigb_d[i] == 3))
	{
		states_d[i] = 1;
	}	
	else 
	{
		states_d[i] = 0;
	}
}

int main(int argc, char** argv)
{
	struct timeval start, end;
	gettimeofday(&start, NULL);
	if (argc != 4)
	{
		std::cout << "USAGE: cellular_automata <dim of matrix> <numGen> <num of blocks>" << std::endl;
		return -1;
	}
	
	int n = atoi(argv[1]); 
	int num_gen = atoi(argv[2]); 
	int num_block = atoi(argv[3]); 

	int num_thread = n * n / num_block; 

	cudaSetDevice(0);
	
	int* states = new int[n*n];
	int* states_d = NULL;
	int* active_neigb = new int[n*n];
	int* active_neigb_d = NULL;

	srand (time(NULL));
	for (int i = 0; i < n*n; ++i)
	{
		states[i] = rand() % 2;
	}	
	
	cudaMalloc((void**)&states_d, n * n * sizeof(int));
	cudaMemcpy(states_d, states, n * n * sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&active_neigb_d, n * n * sizeof(int));
	cudaMemcpy(active_neigb_d, active_neigb, n * n *sizeof(int), cudaMemcpyHostToDevice);
	
	printf("n: %d, num_block: %d, num_thread: %d\n", n, num_block, num_thread);
//	print_matrix(states, n);

	for (int i = 0; i < num_gen; ++i)
	{
		countActiveNeigb<<< dim3(num_block), dim3(num_thread)>>> (states_d, active_neigb_d, n);
		cudaThreadSynchronize();

		cudaMemcpy(active_neigb, active_neigb_d, n * n * sizeof(int), cudaMemcpyDeviceToHost);
			
		//update states
		updateStates<<< dim3(num_block), dim3(num_thread)>>> (states_d, active_neigb_d);
		cudaThreadSynchronize();
		cudaMemcpy(states, states_d, n * n * sizeof(int), cudaMemcpyDeviceToHost);

//		print_matrix(states, n);
	}
	
	cudaFree(states_d);
	cudaFree(active_neigb_d);

	delete[] states;
	delete[] active_neigb; 

	gettimeofday(&end, NULL);
	double time_gap = (end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec;	
	printf("Time cost: %.2lf s.\n", time_gap / 100000);       

	return 0;
}
