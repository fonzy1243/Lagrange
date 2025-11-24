#ifndef LAGRANGE_BARNES_CUDA_CUH
#define LAGRANGE_BARNES_CUDA_CUH

#include <cuda_gl_interop.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_DEPTH 32
#define NULL_PTR (-1)
#define LOCKED (-2)

#define G 0.05
#define THETA 0.5f
#define EPS 0.1f
#define DT 0.001f

#define COLLISION_RADIUS 0.001f

/// Device functions

// Check if an index is a body or an empty cell
__device__ inline bool is_body(const int idx, const int n_bodies) {
    return (idx >= 0 && idx < n_bodies);
}

// Determine which octant a body belongs to
__device__ inline int get_octant(float4 p_body, float4 p_cell, float radius) {
    int octant = 0;
    if (p_body.x > p_cell.x) octant |= 1;
    if (p_body.y > p_cell.y) octant |= 2;
    if (p_body.z > p_cell.z) octant |= 4;
    return octant;
}

// Atomic min/max float
// Source - https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
// Posted by timothygiraffe
// Retrieved 2025-11-24, License - CC BY-SA 4.0
__device__ __forceinline__ float atomicMinFloat(float* addr, const float value) {
    float old;
    old = !signbit(value) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = !signbit(value) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

/// Kernels

// Kernel 1: Bounding box
__global__ inline void compute_bounding_box(const float4* pos_mass, const int n_bodies, float4* global_min, float4* global_max) {
    // Shared memory for reduction
    __shared__ float3 s_min[BLOCK_SIZE];
    __shared__ float3 s_max[BLOCK_SIZE];

    const int t_id = threadIdx.x;
    int g_id = blockIdx.x * blockDim.x + threadIdx.x;

    float3 my_min = {1e30f, 1e30f, 1e30f};
    float3 my_max = {-1e30f, -1e30f, -1e30f};

    while (g_id < n_bodies) {
        float4 p = pos_mass[g_id];
        my_min.x = fminf(my_min.x, p.x);
        my_min.y = fminf(my_min.y, p.y);
        my_min.z = fminf(my_min.z, p.z);
        my_max.x = fmaxf(my_max.x, p.x);
        my_max.y = fmaxf(my_max.y, p.y);
        my_max.z = fmaxf(my_max.z, p.z);
        g_id += gridDim.x * blockDim.x;
    }

    s_min[t_id] = my_min;
    s_max[t_id] = my_max;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t_id < s) {
            s_min[t_id].x = fminf(s_min[t_id].x, s_min[t_id + s].x);
            s_min[t_id].y = fminf(s_min[t_id].y, s_min[t_id + s].y);
            s_min[t_id].z = fminf(s_min[t_id].z, s_min[t_id + s].z);
            s_max[t_id].x = fmaxf(s_max[t_id].x, s_max[t_id + s].x);
            s_max[t_id].y = fmaxf(s_max[t_id].y, s_max[t_id + s].y);
            s_max[t_id].z = fmaxf(s_max[t_id].z, s_max[t_id + s].z);
        }
        __syncthreads();
    }

    if (t_id == 0) {
        atomicMinFloat(&global_min->x, s_min[0].x);
        atomicMinFloat(&global_min->y, s_min[0].y);
        atomicMinFloat(&global_min->z, s_min[0].z);

        atomicMaxFloat(&global_max->x, s_max[0].x);
        atomicMaxFloat(&global_max->y, s_max[0].y);
        atomicMaxFloat(&global_max->z, s_max[0].z);
    }
}

// Kernel 2: Iterative tree-build
// Figure 6.9
__global__ inline void build_tree(const float4* pos_mass, int* child_ptrs, const int n_bodies,
                           int* n_cells_counter, float4* cell_pos, float* radius_arr,
                           int* cell_depth) {  // Add depth parameter
    int g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int root = n_bodies;

    if (g_id >= n_bodies) return;

    const float4 my_pos = pos_mass[g_id];
    int curr = root;
    float r = radius_arr[0];

    int depth = 0;
    while (depth < MAX_DEPTH) {
        int octant = get_octant(my_pos, cell_pos[curr], r);
        int child_idx = curr * 8 + octant;

        int dest = child_ptrs[child_idx];

        if (dest == LOCKED) {
            continue;
        }

        if (dest != NULL_PTR && !is_body(dest, n_bodies)) {
            curr = dest;
            r *= 0.5f;
            depth++;
            continue;
        }

        if (const int old_val = atomicCAS(&child_ptrs[child_idx], dest, LOCKED); old_val != dest) {
            continue;
        }

        if (dest == NULL_PTR) {
            child_ptrs[child_idx] = g_id;
            break;
        }

        const int new_cell = atomicAdd(n_cells_counter, 1);

        const float4 parent_pos = cell_pos[curr];
        const float new_r = r * 0.5f;
        float4 new_pos = parent_pos;

        new_pos.x += (octant & 1) ? new_r : -new_r;
        new_pos.y += (octant & 2) ? new_r : -new_r;
        new_pos.z += (octant & 4) ? new_r : -new_r;

        cell_pos[new_cell] = new_pos;
        radius_arr[new_cell] = new_r;
        cell_depth[new_cell] = depth + 1;  // Store depth of new cell

        for (int i = 0; i < 8; i++) {
            child_ptrs[new_cell * 8 + i] = NULL_PTR;
        }

        __threadfence();

        child_ptrs[child_idx] = new_cell;

        const int old_body = dest;
        const float4 old_pos = pos_mass[old_body];
        const int old_octant = get_octant(old_pos, new_pos, new_r);
        child_ptrs[new_cell * 8 + old_octant] = old_body;

        curr = new_cell;
        r = new_r;
        depth++;
    }
}

// Kernel 3: CoG
// Figure 6.11
__global__ inline void compute_cog(float4* pos_mass, const int* child_ptr,
                            const int* cells_at_depth_flat,
                            const int depth_level,
                            const int n_cells_at_depth,
                            int n_bodies,
                            const int n_cells) {
    const int g_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_id >= n_cells_at_depth) return;

    // Calculate flat index
    const int idx = cells_at_depth_flat[depth_level * n_cells + g_id];

    float4 cm = {};

    for (int i = 0; i < 8; i++) {
        const int child = child_ptr[idx * 8 + i];

        if (child == NULL_PTR) continue;

        float4 c_data = pos_mass[child];
        cm.x += c_data.x * c_data.w;
        cm.y += c_data.y * c_data.w;
        cm.z += c_data.z * c_data.w;
        cm.w += c_data.w;
    }

    if (cm.w > 0.f) {
        cm.x /= cm.w;
        cm.y /= cm.w;
        cm.z /= cm.w;
    }

    pos_mass[idx].x = cm.x;
    pos_mass[idx].y = cm.y;
    pos_mass[idx].z = cm.z;

    __threadfence();

    pos_mass[idx].w = cm.w;
}

// Kernel 5: Compute forces
// Figure 6.13
__global__ inline void compute_forces(const float4* pos_mass, float4* acc, const int* child_ptr, const int n_bodies, const int root,
    const float* radius_arr) {
    const int g_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_id >= n_bodies) return;

    const float4 my_pos = pos_mass[g_id];
    float4 my_acc = {};

    int stack[MAX_DEPTH];
    int top = 0;
    stack[top++] = root;

    // __shared__ float4 s_node_data[BLOCK_SIZE / WARP_SIZE];
    // __shared__ int s_child_base[BLOCK_SIZE / WARP_SIZE];
    //
    // int warp_id = threadIdx.x / WARP_SIZE;
    // int lane_id = threadIdx.x % WARP_SIZE;

    while (top > 0) {
        const int node = stack[--top];

        float4 node_pos = pos_mass[node];
        const float dx = node_pos.x - my_pos.x;
        const float dy = node_pos.y - my_pos.y;
        const float dz = node_pos.z - my_pos.z;
        const float dist_sq = dx * dx + dy * dy + dz * dz + EPS * EPS;
        const float dist = sqrtf(dist_sq);

        const bool is_leaf = is_body(node, n_bodies);

        const float width = radius_arr[node] * 2.f;

        if (const bool far_enough = (width / dist) < THETA; is_leaf || far_enough) {
            // Compute force
            if (node != g_id) {
                const float m = node_pos.w;
                const float f = G * m / (dist_sq * dist);
                my_acc.x += -f * dx;
                my_acc.y += -f * dy;
                my_acc.z += -f * dz;
            }
        } else {
            // Push children to stack
            const int base = node * 8;
            for (int i = 0; i < 8; i++) {
                if (const int child = child_ptr[base + i]; child != NULL_PTR) {
                    if (top < MAX_DEPTH) stack[top++] = child;
                }
            }
        }
    }

    acc[g_id] = my_acc;
}

__global__ inline void detect_collisions_tree(
    const float4* pos_mass, float4* vel,
    const int* child_ptr, const int n_bodies, const int root,
    const float* radius_arr) {

    const int g_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_id >= n_bodies) return;

    const float4 my_pos = pos_mass[g_id];
    float4 my_vel = vel[g_id];

    int stack[MAX_DEPTH];
    int top = 0;
    stack[top++] = root;

    while (top > 0) {
        const int node = stack[--top];

        float4 node_pos = pos_mass[node];
        const float dx = node_pos.x - my_pos.x;
        const float dy = node_pos.y - my_pos.y;
        const float dz = node_pos.z - my_pos.z;
        const float dist_sq = dx * dx + dy * dy + dz * dz;
        const float dist = sqrtf(dist_sq);

        const bool is_leaf = is_body(node, n_bodies);
        const float width = radius_arr[node] * 2.f;

        // Check if cell's closest point is within collision range
        const float cell_min_dist = fmaxf(0.0f, dist - width * 0.866f);

        if (is_leaf) {
            // Check collision with this particle
            if (node != g_id && dist < COLLISION_RADIUS && dist > 1e-6f) {
                // Normal vector
                const float nx = dx / dist;
                const float ny = dy / dist;
                const float nz = dz / dist;

                const float4 other_vel = vel[node];

                // Relative velocity
                const float dvx = other_vel.x - my_vel.x;
                const float dvy = other_vel.y - my_vel.y;
                const float dvz = other_vel.z - my_vel.z;

                // Relative velocity along collision normal
                const float dvn = dvx*nx + dvy*ny + dvz*nz;

                // Only resolve if particles are approaching
                if (dvn > 0) {
                    const float mi = my_pos.w;
                    const float mj = node_pos.w;

                    constexpr float restitution = 0.8f;
                    const float impulse = (1.0f + restitution) * dvn / (mi + mj);

                    // Apply impulse
                    my_vel.x += impulse * mj * nx;
                    my_vel.y += impulse * mj * ny;
                    my_vel.z += impulse * mj * nz;
                }
            }
        } else if (cell_min_dist < COLLISION_RADIUS) {
            // Cell is close enough - need to check children
            const int base = node * 8;
            for (int i = 0; i < 8; i++) {
                if (const int child = child_ptr[base + i]; child != NULL_PTR) {
                    if (top < MAX_DEPTH) stack[top++] = child;
                }
            }
        }
        // If cell is too far, we skip it entirely (don't push children)
    }

    vel[g_id] = my_vel;
}

__global__ inline void separate_overlaps_tree(
    float4* pos_mass, const int* child_ptr,
    const int n_bodies, const int root,
    const float* radius_arr) {

    const int g_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_id >= n_bodies) return;

    float4 my_pos = pos_mass[g_id];
    float3 separation = {0.0f, 0.0f, 0.0f};
    int collision_count = 0;

    int stack[MAX_DEPTH];
    int top = 0;
    stack[top++] = root;

    while (top > 0) {
        const int node = stack[--top];

        float4 node_pos = pos_mass[node];
        const float dx = node_pos.x - my_pos.x;
        const float dy = node_pos.y - my_pos.y;
        const float dz = node_pos.z - my_pos.z;
        const float dist_sq = dx * dx + dy * dy + dz * dz;
        const float dist = sqrtf(dist_sq);

        const bool is_leaf = is_body(node, n_bodies);
        const float width = radius_arr[node] * 2.f;
        const float cell_min_dist = fmaxf(0.0f, dist - width * 0.866f);

        if (is_leaf) {
            if (node != g_id && dist < COLLISION_RADIUS && dist > 1e-6f) {
                const float overlap = COLLISION_RADIUS - dist;
                separation.x -= (dx / dist) * overlap * 0.5f;
                separation.y -= (dy / dist) * overlap * 0.5f;
                separation.z -= (dz / dist) * overlap * 0.5f;
                collision_count++;
            }
        } else if (cell_min_dist < COLLISION_RADIUS) {
            const int base = node * 8;
            for (int i = 0; i < 8; i++) {
                if (const int child = child_ptr[base + i]; child != NULL_PTR) {
                    if (top < MAX_DEPTH) stack[top++] = child;
                }
            }
        }
    }

    if (collision_count > 0) {
        my_pos.x += separation.x / collision_count;
        my_pos.y += separation.y / collision_count;
        my_pos.z += separation.z / collision_count;
        pos_mass[g_id] = my_pos;
    }
}


// Kernel 6: Integration
__global__ inline void update_bodies(float4* pos_mass, float4* vel, const float4* acc, const int n_bodies) {
    const int g_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_id >= n_bodies) return;

    float4 p = pos_mass[g_id];
    float4 v = vel[g_id];
    float4 a = acc[g_id];

    v.x += a.x * DT;
    v.y += a.y * DT;
    v.z += a.z * DT;

    p.x += v.x * DT;
    p.y += v.y * DT;
    p.z += v.z * DT;

    pos_mass[g_id] = p;
    vel[g_id] = v;
}

/// Global helpers

__global__ inline void reset_arrays(float4* pos_mass, int* child_ptrs, int *cell_depth, const int n_bodies, const int max_nodes) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < max_nodes) {
        for (int i = 0; i < 8; i++) {
            child_ptrs[gid * 8 + i] = NULL_PTR;
        }
    }

    if (gid >= n_bodies && gid < max_nodes) {
        pos_mass[gid].w = -1.0f;
        pos_mass[gid].x = 0.0f;
        pos_mass[gid].y = 0.0f;
        pos_mass[gid].z = 0.0f;
        cell_depth[gid] = -1;
    }
}

__global__ inline void organize_cells_by_depth(const int* cell_depth, const int n_bodies, const int total_cells,
                                        int* cells_by_depth, int* cells_count_by_depth, const int n_cells) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int cell_idx = n_bodies + gid;

    if (cell_idx >= total_cells) return;

    if (int depth = cell_depth[cell_idx]; depth >= 0 && depth < MAX_DEPTH) {
        const int pos = atomicAdd(&cells_count_by_depth[depth], 1);
        cells_by_depth[depth * n_cells + pos] = cell_idx;
    }
}

__global__ inline void copy_to_gl_buffer(const float4* pos_mass, float* gl_buffer, const int n_bodies) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_bodies) return;

    float4 p = pos_mass[gid];
    gl_buffer[gid * 3 + 0] = p.x;
    gl_buffer[gid * 3 + 1] = p.y;
    gl_buffer[gid * 3 + 2] = p.z;
}

class BarnesHut {
    int n_bodies;
    int n_cells;
    int total_nodes;

    float4 *d_pos_mass, *d_vel, *d_acc;
    int *d_child_ptrs, *d_n_cells_count, *d_cell_depth;
    int *d_cells_by_depth_flat, *d_cells_count_by_depth;
    float4 *d_g_min, *d_g_max;
    float *d_radius;

    GLuint vbo;
    cudaGraphicsResource *cuda_vbo_resource;
    bool gl_interop_initialized;

public:
    BarnesHut(const int num_bodies) : n_bodies(num_bodies), gl_interop_initialized(false) {
        n_cells = n_bodies;
        total_nodes = n_bodies + n_cells;

        const size_t size_pos = total_nodes * sizeof(float4);
        const size_t size_ptr = total_nodes * 8 * sizeof(int);

        cudaMalloc(&d_pos_mass, size_pos);
        cudaMalloc(&d_vel, n_bodies * sizeof(float4));
        cudaMalloc(&d_acc, n_bodies * sizeof(float4));
        cudaMalloc(&d_child_ptrs, size_ptr);
        cudaMalloc(&d_n_cells_count, sizeof(int));
        cudaMalloc(&d_g_min, sizeof(float4));
        cudaMalloc(&d_g_max, sizeof(float4));
        cudaMalloc(&d_radius, total_nodes * sizeof(float));
        cudaMalloc(&d_cell_depth, total_nodes * sizeof(int));
        cudaMalloc(&d_cells_count_by_depth, MAX_DEPTH * sizeof(int));
        cudaMalloc(&d_cells_by_depth_flat, MAX_DEPTH * n_cells * sizeof(int));
    }

    ~BarnesHut() {
        if (gl_interop_initialized) {
            cudaGraphicsUnregisterResource(cuda_vbo_resource);
        }

        cudaFree(d_pos_mass);
        cudaFree(d_vel);
        cudaFree(d_acc);
        cudaFree(d_child_ptrs);
        cudaFree(d_n_cells_count);
        cudaFree(d_g_min);
        cudaFree(d_g_max);
        cudaFree(d_radius);
        cudaFree(d_cell_depth);
        cudaFree(d_cells_count_by_depth);

    }

    void initialize(const float4* initial_positions, const float4* initial_velocities) const {
        cudaMemcpy(d_pos_mass, initial_positions, n_bodies * sizeof(float4),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel, initial_velocities, n_bodies * sizeof(float4),
                   cudaMemcpyHostToDevice);

        cudaMemset(d_acc, 0, n_bodies * sizeof(float4));
    }

    void setup_gl_interop(GLuint vertex_buffer) {
        vbo = vertex_buffer;
        cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
                                     cudaGraphicsMapFlagsWriteDiscard);
        gl_interop_initialized = true;
    }

    void step() const {
        int threadsPerBlock = BLOCK_SIZE;
        int blocks = (total_nodes + threadsPerBlock - 1) / threadsPerBlock;
        int bodyBlocks = (n_bodies + threadsPerBlock - 1) / threadsPerBlock;

        // Reset
        float4 min_box = make_float4(3e37, 3e37f, 3e37f, 0);
        float4 max_box = make_float4(-3e37f, -3e37f, -3e37f, 0);
        int cells_count = n_bodies;

        cudaMemcpy(d_n_cells_count, &cells_count, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_g_min, &min_box, sizeof(float4), cudaMemcpyHostToDevice);
        cudaMemcpy(d_g_max, &max_box, sizeof(float4), cudaMemcpyHostToDevice);

        reset_arrays<<<blocks, threadsPerBlock>>>(d_pos_mass, d_child_ptrs, d_cell_depth,
                                                   n_bodies, total_nodes);
        cudaDeviceSynchronize();

        // Bounding box
        compute_bounding_box<<<bodyBlocks, threadsPerBlock>>>(d_pos_mass, n_bodies,
                                                               d_g_min, d_g_max);
        cudaDeviceSynchronize();

        // Setup root
        cudaMemcpy(&min_box, d_g_min, sizeof(float4), cudaMemcpyDeviceToHost);
        cudaMemcpy(&max_box, d_g_max, sizeof(float4), cudaMemcpyDeviceToHost);

        float4 root_center;
        root_center.x = (min_box.x + max_box.x) * 0.5f;
        root_center.y = (min_box.y + max_box.y) * 0.5f;
        root_center.z = (min_box.z + max_box.z) * 0.5f;
        root_center.w = -1.0f;

        float dx = max_box.x - min_box.x;
        float dy = max_box.y - min_box.y;
        float dz = max_box.z - min_box.z;
        float max_dim = fmaxf(dx, fmaxf(dy, dz));
        float root_rad = max_dim * 0.5f;

        cudaMemcpy(&d_pos_mass[n_bodies], &root_center, sizeof(float4),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(&d_radius[n_bodies], &root_rad, sizeof(float),
                   cudaMemcpyHostToDevice);

        // Build tree
        build_tree<<<bodyBlocks, threadsPerBlock>>>(d_pos_mass, d_child_ptrs, n_bodies,
                                                     d_n_cells_count, d_pos_mass,
                                                     d_radius, d_cell_depth);
        cudaDeviceSynchronize();

        // Organize cells by depth
        cudaMemset(d_cells_count_by_depth, 0, MAX_DEPTH * sizeof(int));

        int depth_zero = 0;
        cudaMemcpy(&d_cell_depth[n_bodies], &depth_zero, sizeof(int),
                   cudaMemcpyHostToDevice);

        int cell_blocks = (n_cells + threadsPerBlock - 1) / threadsPerBlock;
        organize_cells_by_depth<<<cell_blocks, threadsPerBlock>>>(
            d_cell_depth, n_bodies, total_nodes, d_cells_by_depth_flat,
            d_cells_count_by_depth, n_cells);
        cudaDeviceSynchronize();

        // Center of mass (bottom-up)
        int h_cells_count[MAX_DEPTH];
        cudaMemcpy(h_cells_count, d_cells_count_by_depth, MAX_DEPTH * sizeof(int),
                   cudaMemcpyDeviceToHost);

        for (int d = MAX_DEPTH - 1; d >= 0; d--) {
            int count = h_cells_count[d];
            if (count > 0) {
                int level_blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
                compute_cog<<<level_blocks, threadsPerBlock>>>(
                    d_pos_mass, d_child_ptrs, d_cells_by_depth_flat, d, count,
                    n_bodies, n_cells);
                cudaDeviceSynchronize();
            }
        }

        // Forces
        compute_forces<<<bodyBlocks, threadsPerBlock>>>(d_pos_mass, d_acc, d_child_ptrs,
                                                         n_bodies, n_bodies, d_radius);
        cudaDeviceSynchronize();

        // Collision detection
        detect_collisions_tree<<<bodyBlocks, threadsPerBlock>>>(d_pos_mass, d_vel, d_child_ptrs, n_bodies, n_bodies, d_radius);
        cudaDeviceSynchronize();

        separate_overlaps_tree<<<bodyBlocks, threadsPerBlock>>>(d_pos_mass, d_child_ptrs, n_bodies, n_bodies, d_radius);
        cudaDeviceSynchronize();

        // Integration
        update_bodies<<<bodyBlocks, threadsPerBlock>>>(d_pos_mass, d_vel, d_acc, n_bodies);
        cudaDeviceSynchronize();
    }

    void update_gl_buffer() {
        if (!gl_interop_initialized) return;

        float* d_gl_ptr;
        size_t num_bytes;

        cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_gl_ptr, &num_bytes,
                                             cuda_vbo_resource);

        int threadsPerBlock = BLOCK_SIZE;
        int blocks = (n_bodies + threadsPerBlock - 1) / threadsPerBlock;
        copy_to_gl_buffer<<<blocks, threadsPerBlock>>>(d_pos_mass, d_gl_ptr, n_bodies);

        cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
    }

    void get_positions(float4* host_positions) const {
        cudaMemcpy(host_positions, d_pos_mass, n_bodies * sizeof(float4),
                   cudaMemcpyDeviceToHost);
    }
};

#endif