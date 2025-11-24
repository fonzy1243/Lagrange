#ifndef LAGRANGE_NAIVE_CUDA_CUH
#define LAGRANGE_NAIVE_CUDA_CUH

#include <cuda_gl_interop.h>
#include <glad/glad.h>

#define G 1.0f
#define DT_NAIVE 0.001f
#define SOFTENING_NAIVE 1e-5f

// CUDA Kernel: Naive O(N^2) force calculation
__global__ void calculate_forces_kernel(float3* positions, float* masses, float3* forces, int N, float softening_squared) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float3 p_i = positions[idx];
        float3 force_acc = make_float3(0.0f, 0.0f, 0.0f);

        for (int j = 0; j < N; ++j) {
            if (idx == j) continue;

            float3 p_j = positions[j];
            float m_j = masses[j];

            float3 r;
            r.x = p_j.x - p_i.x;
            r.y = p_j.y - p_i.y;
            r.z = p_j.z - p_i.z;

            float dist_sq = r.x * r.x + r.y * r.y + r.z * r.z + softening_squared;
            float inv_dist = rsqrtf(dist_sq);
            float inv_dist_cubed = inv_dist * inv_dist * inv_dist;

            float s = m_j * inv_dist_cubed;

            force_acc.x += s * r.x;
            force_acc.y += s * r.y;
            force_acc.z += s * r.z;
        }

        forces[idx] = force_acc;
    }
}

__global__ void update_bodies_kernel(float3* positions, float3* velocities, float3* forces, float* masses, int N, float dt, float _G) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float3 p = positions[idx];
        float3 v = velocities[idx];
        float3 f = forces[idx];
        float m = masses[idx];

        f.x *= _G;
        f.y *= _G;
        f.z *= _G;

        v.x += f.x * dt;
        v.y += f.y * dt;
        v.z += f.z * dt;

        p.x += v.x * dt;
        p.y += v.y * dt;
        p.z += v.z * dt;

        positions[idx] = p;
        velocities[idx] = v;
    }
}

__global__ void copy_to_gl_buffer_naive(const float3* positions, float* gl_buffer, const int n_bodies) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_bodies) return;

    float3 p = positions[gid];
    gl_buffer[gid * 3 + 0] = p.x;
    gl_buffer[gid * 3 + 1] = p.y;
    gl_buffer[gid * 3 + 2] = p.z;
}

class NaiveCUDA {
    int n_bodies;
    float3 *d_positions, *d_velocities, *d_forces;
    float *d_masses;

    GLuint vbo;
    cudaGraphicsResource *cuda_vbo_resource;
    bool gl_interop_initialized;

    int blockSize;
    int gridSize;

public:
    NaiveCUDA(int num_bodies) : n_bodies(num_bodies), gl_interop_initialized(false), blockSize(256) {
        gridSize = (n_bodies + blockSize - 1) / blockSize;

        cudaMalloc(&d_positions, n_bodies * sizeof(float3));
        cudaMalloc(&d_velocities, n_bodies * sizeof(float3));
        cudaMalloc(&d_forces, n_bodies * sizeof(float3));
        cudaMalloc(&d_masses, n_bodies * sizeof(float));
    }

    ~NaiveCUDA() {
        if (gl_interop_initialized) {
            cudaGraphicsUnregisterResource(cuda_vbo_resource);
        }

        cudaFree(d_positions);
        cudaFree(d_velocities);
        cudaFree(d_forces);
        cudaFree(d_masses);
    }

    void initialize(const float* initial_pos, const float* initial_vel) {
        float3* h_pos = new float3[n_bodies];
        float3* h_vel = new float3[n_bodies];
        float* h_masses = new float[n_bodies];

        for (int i = 0; i < n_bodies; i++) {
            h_pos[i].x = initial_pos[i * 4 + 0];
            h_pos[i].y = initial_pos[i * 4 + 1];
            h_pos[i].z = initial_pos[i * 4 + 2];
            h_masses[i] = initial_pos[i * 4 + 3];

            h_vel[i].x = initial_vel[i * 4 + 0];
            h_vel[i].y = initial_vel[i * 4 + 1];
            h_vel[i].z = initial_vel[i * 4 + 2];
        }

        cudaMemcpy(d_positions, h_pos, n_bodies * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_velocities, h_vel, n_bodies * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_masses, h_masses, n_bodies * sizeof(float), cudaMemcpyHostToDevice);

        delete[] h_pos;
        delete[] h_vel;
        delete[] h_masses;
    }

    void setup_gl_interop(GLuint vertex_buffer) {
        vbo = vertex_buffer;
        cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);
        gl_interop_initialized = true;
    }

    void step() {
        const float softening_sq = SOFTENING_NAIVE * SOFTENING_NAIVE;

        calculate_forces_kernel<<<gridSize, blockSize>>>(d_positions, d_masses, d_forces, n_bodies, softening_sq);
        cudaDeviceSynchronize();

        update_bodies_kernel<<<gridSize, blockSize>>>(d_positions, d_velocities, d_forces, d_masses, n_bodies, DT_NAIVE, G);
        cudaDeviceSynchronize();
    }

    void update_gl_buffer() {
        if (!gl_interop_initialized) return;

        float* d_gl_ptr;
        size_t num_bytes;

        cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_gl_ptr, &num_bytes, cuda_vbo_resource);

        copy_to_gl_buffer_naive<<<gridSize, blockSize>>>(d_positions, d_gl_ptr, n_bodies);

        cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
    }
};

#endif // LAGRANGE_NAIVE_CUDA_CUH