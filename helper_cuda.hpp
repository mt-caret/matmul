template <typename T>
void _checkCudaErrors(T result, char const *const func, const char *const file,
                      int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" (%s)\n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorString(result),
            cudaGetErrorName(result), func);
    exit(EXIT_FAILURE);
  }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) _checkCudaErrors((val), #val, __FILE__, __LINE__)

void _checkCublasErrors(cublasStatus_t result, char const *const func,
                        const char *const file, int const line) {
  if (result != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS error at %s:%d code=%d(%s) \"%s\" (%s)\n", file,
            line, static_cast<unsigned int>(result),
            cublasGetStatusString(result), cublasGetStatusName(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCublasErrors(call) \
  _checkCublasErrors((call), #call, __FILE__, __LINE__)
