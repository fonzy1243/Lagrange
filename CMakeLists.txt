cmake_minimum_required(VERSION 4.0)
project(Lagrange CUDA CXX C)

set(CMAKE_CUDA_STANDARD 23)

find_package(OpenGL REQUIRED)

include_directories(${OpenGL_INCLUDE_DIRS})
include_directories("${CMAKE_SOURCE_DIR}/include")

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "GLFW lib only")
set(GLAD_GL "${CMAKE_SOURCE_DIR}/src")

add_subdirectory(.\\glfw)

file(GLOB_RECURSE sources
        "${CMAKE_SOURCE_DIR}/src/*.c"
        "${CMAKE_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/*.cu"
        "${GLAD_GL}/*.c"
)

add_executable(Lagrange ${sources})

set_target_properties(Lagrange PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(Lagrange glfw)
target_link_libraries(Lagrange OpenGL::GL)