#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

vector<float> create_random_matrix(int n) {
  vector<float> matrix(n * n);
  for (int i = 0; i < n * n; ++i) {
    matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
  return matrix;
}

vector<float> naive_matrix_multiply(const vector<float> &m1,
                                    const vector<float> &m2, int n) {
  vector<float> result(n * n, 0);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        result[i * n + j] += m1[i * n + k] * m2[k * n + j];
      }
    }
  }

  return result;
}

#define TILE_SIZE 32
__global__ void tiled_matrix_multiply_kernel(float *m1, float *m2,
                                             float *result, int n) {
  __shared__ float shared_m1[TILE_SIZE][TILE_SIZE];
  __shared__ float shared_m2[TILE_SIZE][TILE_SIZE];

  int tx = threadIdx.x, ty = threadIdx.y;

  int row = blockIdx.y * TILE_SIZE + ty;
  int col = blockIdx.x * TILE_SIZE + tx;

  float sum = 0;
  for (int tile = 0; tile < n / TILE_SIZE; ++tile) {
    shared_m1[ty][tx] = m1[row * n + (tile * TILE_SIZE + tx)];
    shared_m2[ty][tx] = m2[(tile * TILE_SIZE + ty) * n + col];
    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += shared_m1[ty][k] * shared_m2[k][tx];
    }
    __syncthreads();
  }

  if (row < n && col < n) {
    result[row * n + col] = sum;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " <size> <runs>\n";
    return EXIT_FAILURE;
  }

  try {
    int size = stoi(argv[1]);
    int runs = stoi(argv[2]);

    if (size <= 0) {
      throw invalid_argument("Size must be a positive integer.");
    }
    if (runs <= 0) {
      throw invalid_argument("Number of runs must be a positive integer.");
    }

    vector<float> m1 = create_random_matrix(size),
                  m2 = create_random_matrix(size);

    // Transfer matrices to device
    float *d_m1, *d_m2, *d_result;
    cudaMalloc(&d_m1, size * size * sizeof(float));
    cudaMalloc(&d_m2, size * size * sizeof(float));
    cudaMalloc(&d_result, size * size * sizeof(float));
    cudaMemcpy(d_m1, m1.data(), size * size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, m2.data(), size * size * sizeof(float),
               cudaMemcpyHostToDevice);

    if (size % TILE_SIZE != 0) {
      throw invalid_argument("Size must be divisible by TILE_SIZE.");
    }

    // Define grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x,
                 (size + blockDim.y - 1) / blockDim.y);

    if (size < 1000) {
      // Check if lines up with CUDA implementation up to floating point error
      vector<float> naive_result = naive_matrix_multiply(m1, m2, size);
      vector<float> cuda_result(size * size);

      // Run CUDA kernel
      tiled_matrix_multiply_kernel<<<gridDim, blockDim>>>(d_m1, d_m2, d_result,
                                                          size);
      cudaDeviceSynchronize();

      // Copy result back to host
      cudaMemcpy(cuda_result.data(), d_result, size * size * sizeof(float),
                 cudaMemcpyDeviceToHost);

      const float epsilon = 1e-3f;
      for (int i = 0; i < size * size; ++i) {
        if (std::abs(naive_result[i] - cuda_result[i]) > epsilon) {
          cerr << "Significant mismatch at index " << i
               << ": naive = " << naive_result[i]
               << ", cuda = " << cuda_result[i] << '\n';
          return EXIT_FAILURE;
        }
      }
    }

    // Warmup
    for (int i = 0; i < 10; ++i) {
      tiled_matrix_multiply_kernel<<<gridDim, blockDim>>>(d_m1, d_m2, d_result,
                                                          size);
      cudaDeviceSynchronize();
    }

    cout << "size,run,runtime_us,method" << endl;
    for (int i = 0; i < runs; ++i) {
      auto start_time = chrono::high_resolution_clock::now();
      tiled_matrix_multiply_kernel<<<gridDim, blockDim>>>(d_m1, d_m2, d_result,
                                                          size);
      cudaDeviceSynchronize();
      auto end_time = chrono::high_resolution_clock::now();
      auto duration =
          chrono::duration_cast<chrono::microseconds>(end_time - start_time)
              .count();
      cout << size << "," << i + 1 << "," << duration << ",naive_cuda" << endl;
    }

    // Clean up
    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_result);
  } catch (const exception &e) {
    cerr << "Error: " << e.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}