//
// Created by Roi Victor Roberto on 11/12/2025.
//
#include <iostream>
#include <vector>
#include <random>
#include <cstdio>
#include <cuda_runtime.h>

// --- CUDA error-checking ---
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// CUDA Kernel: Naive O(N^2) force calculation
__global__ void calculate_forces_kernel(float3* positions, float* masses, float3* forces, int N, float softening_squared) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float3 p_i = positions[idx];
        float m_i = masses[idx];
        float3 force_acc = make_float3(0.0f, 0.0f, 0.0f);

        // Iterate over all other bodies to accumulate forces
        for (int j = 0; j < N; ++j) {
            if (idx == j) {
                continue;
            }

            float3 p_j = positions[j];
            float m_j = masses[j];

            float3 r;
            r.x = p_j.x - p_i.x;
            r.y = p_j.y - p_i.y;
            r.z = p_j.z - p_i.z;

            float dist_sq = r.x * r.x + r.y * r.y + r.z * r.z + softening_squared;

            // Calculate inverse distance cubed
            // rsqrtf is the fast reciprocal square root
            float inv_dist = rsqrtf(dist_sq);
            float inv_dist_cubed = inv_dist * inv_dist * inv_dist;

            // We leave G out for now and apply it in the update step for efficiency
            float s = m_j * inv_dist_cubed;

            force_acc.x += s * r.x;
            force_acc.y += s * r.y;
            force_acc.z += s * r.z;
        }

        forces[idx] = force_acc;
    }
}


// CUDA Kernel: Update position and velocity of each body
// Using a simple Euler integration method for this naive implementation.
__global__ void update_bodies_kernel(float3* positions, float3* velocities, float3* forces, float* masses, int N, float dt, float G) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float3 p = positions[idx];
        float3 v = velocities[idx];
        float3 f = forces[idx];
        float m = masses[idx];

        // Gravitational constant is applied here
        f.x *= G * m;
        f.y *= G * m;
        f.z *= G * m;

        float3 a = make_float3(f.x / m, f.y / m, f.z / m);

        // Update velocity
        v.x += a.x * dt;
        v.y += a.y * dt;
        v.z += a.z * dt;

        // Update position
        p.x += v.x * dt;
        p.y += v.y * dt;
        p.z += v.z * dt;

        positions[idx] = p;
        velocities[idx] = v;
    }
}

// Host function to initialize bodies in a random uniform distribution
void initialize_bodies(std::vector<float3>& positions, std::vector<float3>& velocities, std::vector<float>& masses, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> mass_dist(0.5f, 1.5f);

    for (int i = 0; i < N; ++i) {
        positions[i] = make_float3(pos_dist(gen), pos_dist(gen), pos_dist(gen));
        velocities[i] = make_float3(0.0f, 0.0f, 0.0f); // Start from rest
        masses[i] = mass_dist(gen);
    }
}

int main() {
    const int N = 1024; // Number of bodies (configurable)
    const float dt = 0.001f; // Integration time step
    const int n_steps = 10000; // Total simulation steps
    const float G = 1; // A realistic G is very small, use 1 for simulation stability
    const float G_sim = 1.0f;
    const float softening = 1e-5f; // Softening parameter
    const float softening_squared = softening * softening;

    std::cout << "--- Phase 2: Naive CUDA N-Body Simulation ---" << std::endl;
    std::cout << "Number of bodies: " << N << std::endl;
    std::cout << "Time steps: " << n_steps << std::endl;

    // --- 1. Host Memory Allocation and Initialization ---
    std::vector<float3> h_positions(N);
    std::vector<float3> h_velocities(N);
    std::vector<float> h_masses(N);

    initialize_bodies(h_positions, h_velocities, h_masses, N);

    // --- 2. Device Memory Allocation ---
    float3 *d_positions, *d_velocities, *d_forces;
    float *d_masses;

    checkCudaErrors(cudaMalloc(&d_positions, N * sizeof(float3)));
    checkCudaErrors(cudaMalloc(&d_velocities, N * sizeof(float3)));
    checkCudaErrors(cudaMalloc(&d_forces, N * sizeof(float3)));
    checkCudaErrors(cudaMalloc(&d_masses, N * sizeof(float)));

    // --- 3. Copy Data from Host to Device ---
    checkCudaErrors(cudaMemcpy(d_positions, h_positions.data(), N * sizeof(float3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_velocities, h_velocities.data(), N * sizeof(float3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_masses, h_masses.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // --- 4. Simulation Setup ---
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    std::cout << "Starting simulation..." << std::endl;
    checkCudaErrors(cudaEventRecord(start));

    // --- 5. Main Simulation Loop ---
    for (int step = 0; step < n_steps; ++step) {
        // a. Calculate forces
        calculate_forces_kernel<<<gridSize, blockSize>>>(d_positions, d_masses, d_forces, N, softening_squared);
        checkCudaErrors(cudaGetLastError()); // To check for kernel launch errors

        // b. Update positions and velocities
        update_bodies_kernel<<<gridSize, blockSize>>>(d_positions, d_velocities, d_forces, d_masses, N, dt, G_sim);
        checkCudaErrors(cudaGetLastError());
    }

    // --- 6. Timing and Performance Calculation ---
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    double seconds = milliseconds / 1000.0;

    // GFLOPS calculation:
    double interactions_per_step = (double)N * (double)N;
    double flops_per_step = interactions_per_step * 20.0 + N * 15.0;
    double gflops = (flops_per_step * n_steps) / (seconds * 1e9);

    std::cout << "Simulation finished." << std::endl;
    std::cout << "Total execution time: " << seconds << " seconds" << std::endl;
    std::cout << "Average time per step: " << milliseconds / n_steps << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    // --- 7. Clean up ---
    checkCudaErrors(cudaFree(d_positions));
    checkCudaErrors(cudaFree(d_velocities));
    checkCudaErrors(cudaFree(d_forces));
    checkCudaErrors(cudaFree(d_masses));

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    return 0;
}