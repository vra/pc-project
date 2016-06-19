#include <iostream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv)
{
	const clock_t begin_time = clock();

    if (argc != 2)
    {
        std::cout << "USAGE: num_prime <integer>" << std::endl;
        return -1;
    }

    int n = atoi(argv[1]);
	std::cout << "n:" << n << std::endl;
    int num_primer = 0;

    for (int i = 2; i < n + 1; ++i)
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
			++num_primer;
			std::cout << "primer:" << i << std::endl;
		}
    }

	std::cout << "The number of primers between 0 and " << n << " is: " << num_primer << std::endl;

	std::cout << "*****Time cost: " << float(clock() - begin_time) / CLOCKS_PER_SEC << "s" << std::endl;
    return 0;
}

