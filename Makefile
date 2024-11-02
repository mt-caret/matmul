cpu: cpu.cpp
	g++ -O3 -march=native -fopenmp -o cpu cpu.cpp -lblas

gpu: gpu.cu
	nvcc -O3 -o gpu gpu.cu -lblas

query-gpu: query-gpu.cu
	nvcc -o query-gpu query-gpu.cu

.PHONY: clean all format

all: cpu gpu query-gpu

format:
	clang-format -i -style=Google *.cpp *.cu
clean:
	rm -f cpu gpu query-gpu
