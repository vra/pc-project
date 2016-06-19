from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import Matrices 
from pyspark.mllib.linalg.distributed import RowMatrix, IndexedRow, IndexedRowMatrix,BlockMatrix
from random import random
import numpy as np
import sys
import time

sc = SparkContext(appName="LUFactorization")
sqlContext = SQLContext(sc)
sc.setLogLevel("ERROR")


def get_elem(rowMatrix, i, j):
	return rowMatrix.rows.collect()[i][j]

def set_elem(rowMatrix, i, j, value):
	n = rowMatrix.numRows()
	listOfElems = [rowMatrix.rows.collect()[my_iter].toArray() for my_iter in range(n)]
	a = np.array(listOfElems)
	np.put(a, i * n + j, value)
	
	return RowMatrix(sc.parallelize(a), n, n)

def print_matrx(rowMatrix, name):
	n = rowMatrix.numRows()
	listOfElems = [rowMatrix.rows.collect()[my_iter].toArray() for my_iter in range(n)]
	a = np.array(listOfElems)
	print('*********matrix:%s******************\n' %(name))	
	print(a)	

def lu_factorization(A):
	n = A.numRows()
	
	L = RowMatrix(sc.parallelize(np.eye(n)), n, n)
	U = RowMatrix(sc.parallelize(np.zeros((n,n))), n, n)
	
	for k  in range(0, n):
		for i in range(k+1, n):
			L = set_elem(L, i, k, get_elem(A, i, k) / get_elem(A, k, k))

		for j in range(k, n):
			U = set_elem(U, k, j, get_elem(A, k, j))
		
		for i in range(k+1, n):
			for j in range(k+1, n):
				A = set_elem(A, i, j, get_elem(A, i, j) - get_elem(L, i, k) * get_elem(U, k, j))

	return L, U
				

def main():
	if len(sys.argv) < 2:
		print('USAGE: lu_factorization.py <dim of matrix>')
		return 
	
	
	n = int(sys.argv[1])
	rows = sc.parallelize(np.random.randint(n*n, size=(n,n)))
	mat = RowMatrix(rows, n,n)

		
	L, U = lu_factorization(mat)
	print('**************finish LU Factorization!')
#	print_matrx(mat, 'A')
#	print_matrx(L, 'L')
#	print_matrx(U, 'U')
	


if __name__ == '__main__':
	start_time = time.time()
	main()
	print('-------run time: %s s' %(time.time() -start_time))
