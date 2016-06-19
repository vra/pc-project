#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>

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
	if (argc != 2)
	{
		std::cout << "USAGE: matrix_multiplication <dim of matrix>" << std::endl;
		return -1;
	}

	int n = atoi(argv[1]);
	
	srand (time(NULL));

	//initalize the matrix randomly.
	float** A = new float*[n];
	float** B = new float*[n];
	float** C = new float*[n];

	for (int i = 0; i < n; ++i)
	{
		A[i] = new float[n];
		B[i] = new float[n];
		C[i] = new float[n];
		for (int j = 0; j < n; ++j)
		{
			A[i][j] = float(rand() % 100) / 3 ;
			B[i][j] = float(rand() % 100) / 7;
			C[i][j] = 0;
		}
	}

	print_matrix(A, n);
	print_matrix(B, n);
	
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			for (int k = 0; k < n; ++k)
			{
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}

	print_matrix(C, n);
	return 0;	
}
