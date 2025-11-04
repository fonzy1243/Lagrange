cmake_minimum_required(VERSION 4.0)
project(Lagrange CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(Lagrange main.cu)

set_target_properties(Lagrange PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
