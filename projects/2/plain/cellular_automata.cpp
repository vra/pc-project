#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>

void print_matrix(bool** states, int n)
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
	if (argc != 3)
	{
		std::cout << "USAGE: cellular_automata <dim of matrix> <numGen>" << std::endl;
		return -1;
	}

	int n = atoi(argv[1]);
	int num_gen = atoi(argv[2]);
	
	srand (time(NULL));

	//initalize the matrix randomly.
	bool** states = new bool*[n];
	int** active_neigb = new int*[n];

	for (int i = 0; i < n; ++i)
	{
		states[i] = new bool[n];
		active_neigb[i] = new int[n];
		for (int j = 0; j < n; ++j)
		{
			states[i][j] = rand() % 2;
			active_neigb[i][j] = 0;
		}
	}

	print_matrix(states, n);

	for (int t = 0; t < num_gen; ++t)
	{
		//reset count
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j< n; ++j)
			{
				active_neigb[i][j] = 0;
			}
		}

		//count the states of its neigbour
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				for (int x = i - 1; x < i + 2; ++x)
				{
					for (int y = j - 1; y < j + 2; ++y)
					{
						
						if (x >= 0 && y >= 0 && x < n && y < n)
						{
							active_neigb[i][j] += states[x][y];
						}	
					}
				}	
				active_neigb[i][j] -= states[i][j];
			}
		}
		
		//update matrix
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				if ( (states[i][j] == 1 && (active_neigb[i][j] == 2 || active_neigb[i][j] == 3)) || (states[i][j] == 0 && active_neigb[i][j] == 3))
				{
					states[i][j] = 1;
				}	
				else 
				{
					states[i][j] = 0;
				}
			}
		}

		printf("*****The %dth time:\n", t);
		print_matrix(states, n);
	}
	
	return 0;	
}
