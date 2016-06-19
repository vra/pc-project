from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import Matrices 
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix,BlockMatrix
from random import random
import numpy as np
import sys

sc = SparkContext(appName="MatrixMultiplication")
sqlContext = SQLContext(sc)

#indexedRows = sc.parallelize([IndexedRow(0, [1, 2, 3]), 
#                              IndexedRow(1, [4, 5, 6]), 
#                              IndexedRow(2, [7, 8, 9]), 
#                              IndexedRow(3, [10, 11, 12])])
#

def main():
	if len(sys.argv) < 2:
		print('USAGE: matrix_mult.py <dim of matrix>')
		return 
	
	
	n = int(sys.argv[1])
	dm2 = Matrices.dense(n, n, np.random.randint(1, n * n, n * n).tolist())
	blocks1 = sc.parallelize([((0,0), dm2)])
	m2 = BlockMatrix(blocks1, n,n)
	m3 = BlockMatrix(blocks1, n,n)
	ret = m3.multiply(m2).toIndexedRowMatrix().toRowMatrix().rows.collect()
	print('****************n:', n)


if __name__ == '__main__':
	main()
