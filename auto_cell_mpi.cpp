#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <mpi.h>

void print_matrix(bool* states, int n)
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
	int* states = NULL;
	int* active_neigb = NULL;

	int my_id = 0;
	double start_time;
	double end_time;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_thread);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

	if (my_id == 0)
	{
		start_time = MPI_Wtime();
		if (argc != 3)
		{
			std::cout << "USAGE: auto_cell <dim of matrix> <num of Generatation>" << std::endl;
			return -1;
		}

		n = atoi(argv[1]);
		num_gen = atoi(argv[2]);

		//initalize the matrix randomly.
		states = new int[n * n];
		active_neigb = new int[n * n];

		srand (time(NULL));
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				states[i * n + j] = rand() % 2;
				active_neigb[i * n + j] = 0;
			}
		}
	}
	
	//Broacast the variables
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	int n_per_thread = n * n / num_thread;
	int* local_active_neigb = new int[n_per_thread];
	/*int* local_active_neigb = new int[n_per_thread];
	for (int i = 0; i < n_per_thread; ++i)
	{
		local_active_neigb[i] = 0;
	}	
*/
	for (int t = 0; t < num_gen; ++t)
	{
		//broadcast whole states matrix to all thread because we must all its info in each thread.
		MPI_Bcast(states, n * n, MPI_INT, 0, MPI_COMM_WORLD);
	
		//scatter only part of active_neigb using in calculating.
		MPI_Scatter(active_neigb, n_per_thread, MPI_INT, local_active_neigb, n_per_thread, MPI_INT, 0, MPI_COMM_WORLD);

		//reset count
		for (int i = 0; i < n_per_thread; ++i)
		{
			local_active_neigb[i] = 0;
		}

		//count the states of its neigbour
		for (int i = 0; i < n_per_thread; ++i)
		{
			int  peso_i = (my_id * n_per_thread + i) % n;
			int  peso_j = (my_id * n_per_thread + i) / n;
			//printf("numGen: %d,t: %d, n_per_thread: %d, my_id: %d, i: %d, peso_i: %d, peso_j: %d\n", num_gen, t, n_per_thread, my_id, i, peso_i, peso_j);

			for (int x = peso_i - 1; x < peso_i + 2; ++x)
			{
				for (int y = peso_j - 1; y < peso_j + 2; ++y)
				{
					if (x >= 0 && y >= 0 && x < n && y < n)
					{
						local_active_neigb[i] += states[x * n + y];
						//printf("x: %d, y: %d, local_active_neigb: %d\n", x, y, local_active_neigb[i]);
					}	
				}
			}	
			local_active_neigb[i] -= states[peso_i * n + peso_j];
		}
			
		//printf("t: %d, my_id: %d before gather,n_per_thread: %d\n", t, my_id, n_per_thread);
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Gather(local_active_neigb, 1, MPI_INT, active_neigb, 1, MPI_INT, 0, MPI_COMM_WORLD);

		//printf("t: %d, my_id: %d after gather\n", t, my_id);
		//update matrix in thread 0
		if (my_id == 0)
		{
			for (int i = 0; i < n; ++i)
			{
				for (int j = 0; j < n; ++j)
				{
					//printf("t: %d, my_id: %d in update matrix\n", t, my_id);
					if ( (states[i*n+j] == 1 && (active_neigb[i*n+j] == 2 || active_neigb[i*n+j] == 3)) || (states[i*n+j] == 0 && active_neigb[i*n+j] == 3))
					{
						states[i*n+j] = 1;
					}	
					else 
					{
						states[i*n+j] = 0;
					}
				}
			}
		}

	}//i < numGen
	
	if (my_id == 0)
	{
		end_time = MPI_Wtime();
		std::cout << "*****Time cost: " << end_time - start_time << std::endl;
	}

	MPI_Finalize();
	return 0;	
}
