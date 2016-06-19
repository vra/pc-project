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
		std::cout << "USAGE: lu_factorization <dim of matrix>" << std::endl;
		return -1;
	}

	int n = atoi(argv[1]);
	
	srand (time(NULL));

	//initalize the matrix randomly.
	float** A = new float*[n];
	float** L = new float*[n];
	float** U = new float*[n];

	for (int i = 0; i < n; ++i)
	{
		A[i] = new float[n];
		L[i] = new float[n];
		U[i] = new float[n];
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

	print_matrix(A, n);
	
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

	print_matrix(L, n);
	print_matrix(U, n);
	return 0;	
}
