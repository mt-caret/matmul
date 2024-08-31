#include <omp.h>

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

vector<float> openmp_matrix_multiply(const vector<float> &m1,
                                     const vector<float> &m2, int n) {
  vector<float> result(n * n, 0);

#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        result[i * n + j] += m1[i * n + k] * m2[k * n + j];
      }
    }
  }

  return result;
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

    if (size < 1000) {
      // Check if lines up with naive implementation up to floating point error
      vector<float> naive_result = naive_matrix_multiply(m1, m2, size);
      vector<float> openmp_result = openmp_matrix_multiply(m1, m2, size);
      const float epsilon = 1e-3f;
      for (int i = 0; i < size * size; ++i) {
        if (std::abs(naive_result[i] - openmp_result[i]) > epsilon) {
          cerr << "Significant mismatch at index " << i
               << ": naive = " << naive_result[i]
               << ", openmp = " << openmp_result[i] << '\n';
          return EXIT_FAILURE;
        }
      }
    }

    // Warmup
    for (int i = 0; i < 10; ++i) {
      vector<float> _m3 = openmp_matrix_multiply(m1, m2, size);
    }

    cout << "size,run,runtime_us,method" << endl;
    for (int i = 0; i < runs; ++i) {
      auto start_time = chrono::high_resolution_clock::now();
      vector<float> _m4 = openmp_matrix_multiply(m1, m2, size);
      auto end_time = chrono::high_resolution_clock::now();
      auto duration =
          chrono::duration_cast<chrono::microseconds>(end_time - start_time)
              .count();
      cout << size << "," << i + 1 << "," << duration << ",parallel" << endl;
    }
  } catch (const exception &e) {
    cerr << "Error: " << e.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}