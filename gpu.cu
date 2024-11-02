#include <cblas.h>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "helper_cuda.hpp"

using namespace std;

vector<float> create_random_matrix(int n) {
  vector<float> matrix(n * n);
  for (int i = 0; i < n * n; ++i) {
    matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
  return matrix;
}

vector<float> blas_matrix_multiply(const vector<float> &m1,
                                   const vector<float> &m2, int n) {
  vector<float> result(n * n);

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f,
              m1.data(), n, m2.data(), n, 0.0f, result.data(), n);

  return result;
}

__global__ void naive_matrix_multiply_kernel(float *m1, float *m2,
                                             float *result, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    float sum = 0;
    for (int k = 0; k < n; ++k) {
      sum += m1[row * n + k] * m2[k * n + col];
    }
    result[row * n + col] = sum;
  }
}

void measure_naive(const vector<float> &m1, const vector<float> &m2, int size,
                   int runs) {
  // Transfer matrices to device
  float *d_m1, *d_m2, *d_result;
  checkCudaErrors(cudaMalloc(&d_m1, size * size * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_m2, size * size * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_result, size * size * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_m1, m1.data(), size * size * sizeof(float),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_m2, m2.data(), size * size * sizeof(float),
                             cudaMemcpyHostToDevice));

  // Define grid and block dimensions
  dim3 blockDim(32, 32);
  dim3 gridDim((size + blockDim.x - 1) / blockDim.x,
               (size + blockDim.y - 1) / blockDim.y);

  if (size < 5000) {
    // Check if lines up with CUDA implementation up to floating point error
    vector<float> blas_result = blas_matrix_multiply(m1, m2, size);
    vector<float> cuda_result(size * size);

    // Run CUDA kernel
    naive_matrix_multiply_kernel<<<gridDim, blockDim>>>(d_m1, d_m2, d_result,
                                                        size);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(cuda_result.data(), d_result,
                               size * size * sizeof(float),
                               cudaMemcpyDeviceToHost));

    const float epsilon = 1e-2f;
    for (int i = 0; i < size * size; ++i) {
      if (std::abs(blas_result[i] - cuda_result[i]) > epsilon) {
        cerr << "Significant mismatch at index " << i
             << ": blas = " << blas_result[i] << ", cuda = " << cuda_result[i]
             << '\n';
        throw runtime_error(
            "Significant mismatch between blas and naive implementations");
      }
    }
  }

  // Warmup
  for (int i = 0; i < 10; ++i) {
    naive_matrix_multiply_kernel<<<gridDim, blockDim>>>(d_m1, d_m2, d_result,
                                                        size);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }

  for (int i = 0; i < runs; ++i) {
    auto start_time = chrono::high_resolution_clock::now();
    naive_matrix_multiply_kernel<<<gridDim, blockDim>>>(d_m1, d_m2, d_result,
                                                        size);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    auto end_time = chrono::high_resolution_clock::now();
    auto duration =
        chrono::duration_cast<chrono::microseconds>(end_time - start_time)
            .count();
    cout << size << "," << i + 1 << "," << duration << ",naive_cuda" << endl;
  }

  // Clean up
  checkCudaErrors(cudaFree(d_m1));
  checkCudaErrors(cudaFree(d_m2));
  checkCudaErrors(cudaFree(d_result));
}

extern __shared__ float shared[];
__global__ void tiled_matrix_multiply_kernel(float *m1, float *m2,
                                             float *result, int n,
                                             size_t tile_size) {
  float *shared_m1 = shared;
  float *shared_m2 = shared + tile_size * tile_size;

  int tx = threadIdx.x, ty = threadIdx.y;

  int row = blockIdx.y * tile_size + ty;
  int col = blockIdx.x * tile_size + tx;

  float sum = 0;
  for (int tile = 0; tile < n / tile_size; ++tile) {
    shared_m1[ty * tile_size + tx] = m1[row * n + (tile * tile_size + tx)];
    shared_m2[ty * tile_size + tx] = m2[(tile * tile_size + ty) * n + col];
    __syncthreads();

    for (int k = 0; k < tile_size; ++k) {
      sum += shared_m1[ty * tile_size + k] * shared_m2[k * tile_size + tx];
    }
    __syncthreads();
  }

  if (row < n && col < n) {
    result[row * n + col] = sum;
  }
}

void measure_tiled(const vector<float> &m1, const vector<float> &m2, int size,
                   int runs, size_t tile_size) {
  // Transfer matrices to device
  float *d_m1, *d_m2, *d_result;
  checkCudaErrors(cudaMalloc(&d_m1, size * size * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_m2, size * size * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_result, size * size * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_m1, m1.data(), size * size * sizeof(float),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_m2, m2.data(), size * size * sizeof(float),
                             cudaMemcpyHostToDevice));

  // Define grid and block dimensions
  dim3 blockDim(tile_size, tile_size);
  dim3 gridDim((size + blockDim.x - 1) / blockDim.x,
               (size + blockDim.y - 1) / blockDim.y);
  size_t shared_mem_size = tile_size * tile_size * sizeof(float) * 2;

  if (size < 5000) {
    // Check if lines up with CUDA implementation up to floating point error
    vector<float> blas_result = blas_matrix_multiply(m1, m2, size);
    vector<float> cuda_result(size * size);

    // Run CUDA kernel
    tiled_matrix_multiply_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        d_m1, d_m2, d_result, size, tile_size);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(cuda_result.data(), d_result,
                               size * size * sizeof(float),
                               cudaMemcpyDeviceToHost));

    const float epsilon = 1e-2f;
    for (int i = 0; i < size * size; ++i) {
      if (std::abs(blas_result[i] - cuda_result[i]) > epsilon) {
        cerr << "Significant mismatch at index " << i
             << ": blas = " << blas_result[i] << ", cuda = " << cuda_result[i]
             << '\n';
        throw runtime_error(
            "Significant mismatch between blas and tiled implementations");
      }
    }
  }

  // Warmup
  for (int i = 0; i < 10; ++i) {
    tiled_matrix_multiply_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        d_m1, d_m2, d_result, size, tile_size);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }

  for (int i = 0; i < runs; ++i) {
    auto start_time = chrono::high_resolution_clock::now();
    tiled_matrix_multiply_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        d_m1, d_m2, d_result, size, tile_size);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    auto end_time = chrono::high_resolution_clock::now();
    auto duration =
        chrono::duration_cast<chrono::microseconds>(end_time - start_time)
            .count();
    cout << size << "," << i + 1 << "," << duration
         << ",tiled_cuda(tile_size=" << tile_size << ")" << endl;
  }

  // Clean up
  checkCudaErrors(cudaFree(d_m1));
  checkCudaErrors(cudaFree(d_m2));
  checkCudaErrors(cudaFree(d_result));
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

    cout << "size,run,runtime_us,method" << endl;
    measure_naive(m1, m2, size, runs);
    measure_tiled(m1, m2, size, runs, 8);
    measure_tiled(m1, m2, size, runs, 16);
    measure_tiled(m1, m2, size, runs, 32);
  } catch (const exception &e) {
    cerr << "Error: " << e.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}