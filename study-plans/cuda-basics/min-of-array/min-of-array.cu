#include <cuda_runtime.h>
#include <float.h>

__global__ void init_result(float* result) {
    result[0] = FLT_MAX;
}

// Float atomic min via CAS on the int bit pattern.
__device__ __forceinline__ void atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    while (value < __int_as_float(old)) {
        int assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
}

__global__ void min_kernel(const float* input, float* result, int N) {
    __shared__ float sdata[256];  // must equal blockDim.x (256 in solve)

    unsigned int tid = threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Grid-stride loop; out-of-range threads keep the identity +FLT_MAX.
    float local = FLT_MAX;
    for (unsigned int i = blockIdx.x * blockDim.x + tid; i < (unsigned int)N; i += stride) {
        local = fminf(local, input[i]);
    }
    sdata[tid] = local;
    __syncthreads();

    // Tree reduction in shared memory.
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // One thread per block merges into result[0].
    if (tid == 0) {
        atomicMinFloat(result, sdata[0]);
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    init_result<<<1, 1>>>(result);
    min_kernel<<<blocks, threads>>>(input, result, N);
    cudaDeviceSynchronize();
}