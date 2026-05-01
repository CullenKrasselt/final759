cmake_minimum_required(VERSION 3.18)
project(minesweeper_cuda CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(minesweeper_cuda
    main_cuda.cpp
    solver_cuda.cu
    solver.cpp
)
target_include_directories(minesweeper_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(minesweeper_cuda PROPERTIES
    CUDA_ARCHITECTURES "75;80;86;89"
)
target_compile_options(minesweeper_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -O3
        --generate-line-info
    >
)
target_link_libraries(minesweeper_cuda PRIVATE
    CUDA::cudart
    CUDA::curand
    OpenMP::OpenMP_CXX
)
