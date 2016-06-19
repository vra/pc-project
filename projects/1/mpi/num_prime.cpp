#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv)
{
	int num_thread = 0;
	int my_id = 0;
    int n = 0;	
	int my_num_primer = 0;
	int sum_num_primer = 0;
	double start_time, end_time;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_thread);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

	if (my_id == 0)
	{
		start_time = MPI_Wtime();

    	if (argc != 2)
    	{
        	std::cout << "USAGE: num_prime <integer>" << std::endl;
        	return -1;
    	}

    	n = atoi(argv[1]);
		printf("n: %d\n", n);

	}

	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD); 

	//Calculate number of primers in each process
	for (int i = my_id + 2; i < n + 1; i = i + num_thread)
    {
		bool has_factor = false;
		for (int j = 2; j < int(sqrt(i)) + 1; ++j)
		{
			if (i % j == 0)
			{
				has_factor = true;
				break;
			}
		}
		if (!has_factor)
		{
			++ my_num_primer;
//			std::cout << "id: "<< my_id << ", primer:" << i << std::endl;
		}
    	
	}
	
	//reduce all the results
	MPI_Reduce(&my_num_primer, &sum_num_primer, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	
//	MPI_Barrier(MPI_COMM_WORLD);
	if (my_id == 0)
	{
	
		std::cout << "The number of primers between 0 and " << n << " is: " << sum_num_primer << std::endl;
	
		end_time = MPI_Wtime();
		std::cout << "*****Time cost: " << end_time - start_time << std::endl;
	}

	
	MPI_Finalize();
    return 0;
}
