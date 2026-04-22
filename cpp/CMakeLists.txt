cmake_minimum_required(VERSION 3.16)
project(minesweeper_mc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O2)

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
# CUDA — will be added as a separate target (minesweeper_cuda) later.
# ---------------------------------------------------------------------------
