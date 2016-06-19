#include <iostream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

int main(int argc, char** argv)
{
//	const clock_t begin_time = clock();
//	time_t start;
//	time_t end;

	struct timeval start, end;
	gettimeofday(&start, NULL);

//	time(&start);
    if (argc != 3)
    {
        std::cout << "USAGE: num_prime <num_of_thread> <integer>" << std::endl;
        return -1;
    }

    int num_thread = atoi(argv[1]);
    int n = atoi(argv[2]);

	std::cout << "num of thread: " << num_thread << std::endl;
	std::cout << " n: " << n << std::endl;
    int* num_primer = new int[num_thread];
	for (int i = 0; i < num_thread; ++i)
	{
		num_primer[i] = 0;
	}

	omp_set_num_threads(num_thread);
	#pragma omp parallel shared(n, num_primer)
	{
		int id = omp_get_thread_num();
		
    	for (int i = id + 2; i < n + 1; i = i + num_thread)
    	{
			bool has_factor = false;
			#pragma omp parallel shared(n, i, num_primer, has_factor)
			{
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
					++num_primer[id];
					std::cout << "id: "<< id << ", primer:" << i << std::endl;
				}
			}//pragma
    	}
	}//pragma
	
	//add all primers
	int sum_num_primer = 0;
	for (int i = 0; i < num_thread; ++i)
	{
		sum_num_primer += num_primer[i];	
	}

	std::cout << "The number of primers between 0 and " << n << " is: " << sum_num_primer << std::endl;

//	time(&end);
	gettimeofday(&end, NULL);
	double time_gap = (end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec;
//	std::cout << "*****Time cost: " << float(clock() - begin_time) / CLOCKS_PER_SEC << "s" << std::endl;
//	double time_gap = difftime(end, start);
	printf("Time cost: %.2lf s.\n", time_gap / 100000);

    return 0;
}

