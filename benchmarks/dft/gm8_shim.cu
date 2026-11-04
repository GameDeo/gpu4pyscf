// extern "C" shim around GEMMul8 (Ozaki scheme II, INT8 tensor cores) for ctypes use.
// Build (after `make GPU_ARCH=<cc> OZ2_BACKENDS=INT8` in the GEMMul8 checkout):
//   nvcc -O3 -std=c++20 -shared -Xcompiler -fPIC -arch=sm_<cc> gm8_shim.cu -I<GEMMul8>/include \
//        -L<GEMMul8>/lib -lgemmul8 -lcublas -lcublasLt -o libgm8shim.so
#include "gemmul8.hpp"
#include <cublas_v2.h>

extern "C" {
// Column-major DGEMM, C = alpha*op(A)*op(B) + beta*C, emulated with `num_moduli` INT8 moduli.
// If work == NULL nothing is computed and the required workspace size in bytes is returned.
size_t gm8_dgemm(void* handle, int op_a, int op_b, size_t m, size_t n, size_t k,
                 double alpha, const double* A, size_t lda, const double* B, size_t ldb,
                 double beta, double* C, size_t ldc, int num_moduli, int fastmode, void* work)
{
    auto t = gemmul8::gemm<double, gemmul8::Backend::INT8>(
        (cublasHandle_t)handle, (cublasOperation_t)op_a, (cublasOperation_t)op_b, m, n, k,
        &alpha, A, lda, B, ldb, &beta, C, ldc, num_moduli, fastmode != 0, work);
    return work == nullptr ? (size_t)t[0] : 0;
}
}
