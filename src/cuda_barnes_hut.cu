#include <Lagrange_CUDA/octree.h>

__global__ void initOctreeKernel(const OctreeGPU &tree) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tree.numNodes) return;

    tree.start[i] = 0;
    tree.count[i] = 0;
    tree.morton[i] = 0;
    tree.mass[i] = 0.f;
    tree.com[i] = make_float3(0, 0, 0);

#pragma unroll
    for (int c = 0; c < 8; c++) {
        tree.child[i * 8 + c] = 0xFFFFFFFF;
    }
}

///////////////////////////////////////////
// Morton encoding helpers (64-bit Morton)
//////////////////////////////////////////

__host__ __device__ uint64_t expandBits21(uint64_t x) {
    x &= 0x1FFFFFULL;
    x = (x | (x << 32)) & 0x1F00000000FFFFULL;
    x = (x | (x << 16)) & 0x1F0000FF0000FFULL;
    x = (x | (x << 8))  & 0x100F00F00F00F00FULL;
    x = (x | (x << 4))  & 0x10C30C30C30C30C3ULL;
    x = (x | (x << 2))  & 0x1249249249249249ULL;
    return x;
}

__host__ __device__ uint64_t morton3D_64(uint64_t x) {
    return (expandBits21(x) << 2) | (expandBits21(x) << 1) | expandBits21(z);
}