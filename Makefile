naive: naive.cpp
	g++ -O3 -march=native -o naive naive.cpp

openmp: openmp.cpp
	g++ -O3 -march=native -fopenmp -o openmp openmp.cpp

blas: blas.cpp
	g++ -O3 -march=native -o blas blas.cpp -lblas

naive-cuda: naive-cuda.cu
	nvcc -O3  -o naive-cuda naive-cuda.cu

.PHONY: clean all format

all: naive openmp blas naive-cuda

format:
	clang-format -i -style=Google *.cpp *.cu
clean:
	rm -f naive openmp blas naive-cuda