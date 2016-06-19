from pyspark import SparkContext
from random import random
from math import sqrt

sc = SparkContext(appName='CountNumOfPrime')

def is_prime(x):
	for i in range(2, int(sqrt(x)) + 1):
		if x % i == 0:
			return 0
	
	return 1


n = 1000000
count = sc.parallelize(range(2, n + 1)).map(is_prime).reduce(lambda x, y : x + y)

print('*****result: :%d*****' %(count))
