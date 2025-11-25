# Lagrange: Accelerating the N-Body Problem Using CUDA and SIMT

## Abstract
The N-body problem, a fundamental challenge in the field of physics, astrophysics, and molecular mechanics, involves the calculation of pairwise interactions among a system of particles. Traditional CPU-based implementations of this problem struggle with large-scale simulations due to the square growth of the number of interactions, which imposes a significant computational overhead. To address this, we propose leveraging NVIDIA’s Compute Unified Device Architecture (CUDA) and its Single Instruction Multiple Threads (SIMT) model to speed up these computations. Utilizing CUDA C++ as the primary programming language, we aim to compare the baseline performance of the algorithm with O(N2) force calculations to a parallelized implementation, including various memory and algorithmic optimizations, in terms of the metrics of execution time, memory bandwidth, and GFLOPS, among others. The project will be implemented primarily on a GPU platform, whereas the software stack includes CUDA Toolkit, NVCC, OpenGL, and NVIDIA Nsight Systems.

## Short Simulation/Execution of Programs for All Implementations
1. Naive CPU Implementation\
https://drive.google.com/file/d/1PTybv8xNMW4pyHnmIUZu2qGyMwShtHq3

2. Naive CUDA Implementation\
https://drive.google.com/file/d/1WOGhtpgsKHZf_yEUXzm89CWfaRtjVuq0

3. Barnes-Hut + CUDA Implementation\
https://drive.google.com/file/d/1-eP6WKkLlPk96OPdO3MZnQIjvVBFStPB

## Discussion of Parallel Algorithms Implemented in the Program

### Sequential Baseline
The computational bottleneck of the naive CPU implementation resides entirely within the `computeForces` method.
- **Algorithmic Structure** 
    - The implementation utilizes a nested loop to calculate the gravitational interactions between all pairs of bodies.
- **Computational Cost** 
    - This constitutes an $O(N^2)$ complexity. For every simulation step, the system performs $\frac{N(N-1)}{2}$ pairwise force calculations.
- **Integration** 
    - Following the force calculation, the `step` method performs an $O(N)$ linear pass to update positions and velocities using the computed accelerations.

In the later implementations (Naive CUDA and Barnes-Hut + CUDA), they specifically target to improve these two loops: the $O(N^2)$ force calculation, which serves as the primary target, and the $O(N)$ integration step.

### Naive CUDA Implementation
The first parallel implementation retains the $O(N^2)$ algorithmic complexity but leverages the massive parallelism of the GPU to overcome the quadratic cost through raw throughput.
- **Decomposition of the Outer Loop**
    - Unlike the naive CPU implementation, which iterates sequentially through every body to accumulate force, the outer loop of the naive CUDA implementation is unrolled onto the GPU grid.
- **Mapping**
    - The current body `i` is mapped into a unique CUDA thread index `idx` calculated via `blockIdx.x * blockDim.x + threadIdx.x`.
- **Execution**
    - Rather than one processor calculating forces for bodies $0$ to $N$ sequentially, $N$ threads calculate forces for $N$ bodies simultaneously.

### Barnes-Hut Algorithm + CUDA Implementation
In this implementation, we reduce the complexity from $O(N^2)$ to $O(N \log N)$ by parallelizing the construction of the data structure as well, not just the calculations.
- **Parallel Tree Construction**
    - Since recursion is inefficient on GPUs due to stack depth limitations, the CUDA implementation replaces sequential recursion with an iterative, atomic-locking approach.
    - Before building the tree, the code computes the global bounds of the system. This is done via `compute_bounding_box`, which utilizes shared memory reduction within blocks and atomic operations to aggregate global results.
    - The `build_tree` kernel allows bodies to insert themselves into the octree in parallel. 
    - In terms of race conditions where two bodies enter the same cell simultaneously, `atomicCAS` (compare and swap) "locks" a child pointer. If the cell is locked, the thread waits, but if it is occupied, the tree deepens.
- **Force Calculation**
    - This mirrors the `computeForces` method in the sequential version, but the logic is different.
    - Instead of an inner loop iterating over $N$ bodies, each thread maintains a local `stack` array to simulate recursion while traversing the Octree.
    - With a Multipole Acceptance Criterion, the kernel checks the ratio of the cell radius to the distance (`width / dist < THETA`). If the node is sufficiently far, the thread computes the force using the cell's center of mass (`CoG`) and ceases traversal of that branch. This pruning of the traversal path is the source of the logarithmic speedup.
- **Bottom-Up Parallel Reduction**
    - The `compute_cog` kernel implements a level-by-level parallel reduction.
    - It organizes cells by depth using `organize_cells_by_depth` and processes them from the bottom of tree up to the root $(d = MAXDEPTH \rightarrow 0)$. This ensures that when a parent node is processed, its children have strictly finished their calculations, avoiding the need for global barriers.

## Performance Comparisons Between All Three Implementations
The performance of each implementation was measured by averaging the execution time over 30 runs.

**Table 1: Average Execution Time Per Step for Each Implementation**

| Implementation | N_BODIES: 100 | N_BODIES: 1,000 | N_BODIES: 10,000 | N_BODIES: 500,000 |
| :--- | :--- | :--- | :--- | :--- |
| **Naive CPU** | 0 s | 0 s | 0 s | 0 s |
| **Naive CUDA**| 0 s | 0 s | 0 s | 0 s |
| **Barnes-Hut + CUDA** | **0 s** | **0 s** | **0 s** | **0 s** |

**Table 2: Speedup Factor Relative to the Naive CPU Implementation**


| Implementation | N_BODIES: 100 | N_BODIES: 1,000 | N_BODIES: 10,000 | N_BODIES: 500,000 |
| :--- | :--- | :--- | :--- | :--- |
| **Naive CUDA**| 0x | 0x | 0x | 0x |
| **Barnes-Hut + CUDA** | **0x** | **0x** | **0x** | **0x** |

**Table 3: GFLOPS Comparison of Each Implementation**


| Implementation | N_BODIES: 100 | N_BODIES: 1,000 | N_BODIES: 10,000 | N_BODIES: 500,000 |
| :--- | :--- | :--- | :--- | :--- |
| **Naive CPU** | 0 GFLOPS | 0 GFLOPS | 0 GFLOPS | 0 GFLOPS |
| **Naive CUDA**| 0 GFLOPS | 0 GFLOPS | 0 GFLOPS | 0 GFLOPS |
| **Barnes-Hut + CUDA** | **0 GFLOPS** | **0 GFLOPS** | **0 GFLOPS** | **0 GFLOPS** |

## Short Video Presenting the Final Project\
https://drive.google.com/file/d/1c2ytpLZ44CvkR99P6sxtljHpZgqqty6X
