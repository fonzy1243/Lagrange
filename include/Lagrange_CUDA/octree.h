#ifndef LAGRANGE_OCTREE_H
#define LAGRANGE_OCTREE_H

// Traversal access pattern
// uint32_t childIndex = tree.child[nodeIndex * 8 * i];
// if (childIndex != 0xFFFFFFFF) {
//     // process child
// }
struct OctreeGPU {
    uint32_t* child;
    uint32_t* start;
    uint32_t* count;
    uint32_t* morton;
    float3* com;
    float* mass;

    uint32_t numNodes;
};

inline void allocateOctreeGPU(OctreeGPU &tree, uint32_t numNodes) {
    tree.numNodes = numNodes;

    cudaMalloc(&tree.child, 8 * numNodes * sizeof(uint32_t));
    cudaMalloc(&tree.start, numNodes * sizeof(uint32_t));
    cudaMalloc(&tree.count, numNodes * sizeof(uint32_t));
    cudaMalloc(&tree.morton, numNodes * sizeof(uint32_t));
    cudaMalloc(&tree.com, numNodes * sizeof(float3));
    cudaMalloc(&tree.mass, numNodes * sizeof(float));
}

inline void deallocateOctreeGPU(OctreeGPU &tree) {
    cudaFree(tree.child);
    cudaFree(tree.start);
    cudaFree(tree.count);
    cudaFree(tree.morton);
    cudaFree(tree.com);
    cudaFree(tree.mass);
}

#endif //LAGRANGE_OCTREE_H