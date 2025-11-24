#ifndef LAGRANGE_SIMULATOR_INTERFACE_H
#define LAGRANGE_SIMULATOR_INTERFACE_H

#include <glad/glad.h>

// Choose your implementation here:
// #define USE_BARNES_HUT
#define USE_NAIVE_CUDA
// #define USE_SEQUENTIAL

#ifdef USE_BARNES_HUT
    #include <nbody_barnes_cuda/barnes_cuda.cuh>
    typedef BarnesHut SimulatorType;
    #define INIT_FORMAT_CUDA
#elif defined(USE_NAIVE_CUDA)
    #include <nbody_naive_cuda/naive_cuda.cuh>
    typedef  NaiveCUDA SimulatorType;
#else
    #include <nbody_naive/Sequential.hpp>
    typedef Sequential SimulatorType;
    #define INIT_FORMAT_CPU
#endif

#endif //LAGRANGE_SIMULATOR_INTERFACE_H