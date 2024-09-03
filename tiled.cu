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

int main(int argc, char *argv[]) {
  if (argc != 4) {
    cerr << "Usage: " << argv[0] << " <size> <runs> <tile_size>\n";
    return EXIT_FAILURE;
  }

  try {
    int size = stoi(argv[1]);
    int runs = stoi(argv[2]);
    int tile_size = stoi(argv[3]);

    if (size <= 0) {
      throw invalid_argument("Size must be a positive integer.");
    }
    if (runs <= 0) {
      throw invalid_argument("Number of runs must be a positive integer.");
    }
    if (tile_size <= 0) {
      throw invalid_argument("Tile size must be a positive integer.");
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

    if (size % tile_size != 0) {
      throw invalid_argument("Size must be divisible by tile_size.");
    }

    // Define grid and block dimensions
    dim3 blockDim(tile_size, tile_size);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x,
                 (size + blockDim.y - 1) / blockDim.y);
    size_t shared_mem_size = tile_size * tile_size * sizeof(float) * 2;

    // Check if device has enough shared memory
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    if (shared_mem_size > device_prop.sharedMemPerBlock) {
      throw runtime_error("Device does not have enough shared memory.");
    }

    if (size <= 1024) {
      // Check if lines up with CUDA implementation up to floating point error
      vector<float> naive_result = naive_matrix_multiply(m1, m2, size);
      vector<float> cuda_result(size * size);

      // Run CUDA kernel
      tiled_matrix_multiply_kernel<<<gridDim, blockDim, shared_mem_size>>>(
          d_m1, d_m2, d_result, size, tile_size);
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
      tiled_matrix_multiply_kernel<<<gridDim, blockDim, shared_mem_size>>>(
          d_m1, d_m2, d_result, size, tile_size);
      cudaDeviceSynchronize();
    }

    cout << "size,run,runtime_us,method" << endl;
    for (int i = 0; i < runs; ++i) {
      auto start_time = chrono::high_resolution_clock::now();
      tiled_matrix_multiply_kernel<<<gridDim, blockDim, shared_mem_size>>>(
          d_m1, d_m2, d_result, size, tile_size);
      cudaDeviceSynchronize();
      auto end_time = chrono::high_resolution_clock::now();
      auto duration =
          chrono::duration_cast<chrono::microseconds>(end_time - start_time)
              .count();
      cout << size << "," << i + 1 << "," << duration
           << ",tiled_cuda(tile_size=" << tile_size << ")" << endl;
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