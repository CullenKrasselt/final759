cmake_minimum_required(VERSION 3.18)
project(minesweeper_mc CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Release build flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O2)

# ---------------------------------------------------------------------------
# Sequential target (original)
# ---------------------------------------------------------------------------
add_executable(minesweeper_mc
    main.cpp
    solver.cpp
)
target_include_directories(minesweeper_mc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# ---------------------------------------------------------------------------
# OpenMP — uncomment the three lines below to build the parallel version:
# ---------------------------------------------------------------------------
# find_package(OpenMP REQUIRED)
# target_compile_options(minesweeper_mc PRIVATE ${OpenMP_CXX_FLAGS})
# target_link_libraries(minesweeper_mc OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# CUDA target (minesweeper_cuda)
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

add_executable(minesweeper_cuda
    main_cuda.cpp
    solver_cuda.cu
    solver.cpp          # host-side helpers (constraint extraction etc.) reused
)

target_include_directories(minesweeper_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Generate code for common architectures; add yours if needed.
# sm_75 = Turing (RTX 2000), sm_80 = Ampere (RTX 3000/A100),
# sm_86 = Ampere (RTX 3060-3090), sm_89 = Ada (RTX 4000)
set_target_properties(minesweeper_cuda PROPERTIES
    CUDA_ARCHITECTURES "75;80;86;89"
)

target_compile_options(minesweeper_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -O3
        --generate-line-info   # useful for Nsight profiling
    >
)

target_link_libraries(minesweeper_cuda PRIVATE
    CUDA::cudart
    CUDA::curand
)
