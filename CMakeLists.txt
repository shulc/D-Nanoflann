cmake_minimum_required(VERSION 3.16)
project(d_nanoflann CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Position-independent code is required so the static lib can be linked
# into a D executable without relocation errors on Linux/ELF.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# We only need the nanoflann header; it ships no CMake install target.
# Pull the include path directly from the submodule.
set(NANOFLANN_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/extern/nanoflann/include")

# ---- C shim static library -------------------------------------------------
# nanoflann is header-only; the shim is the only compilation unit.
add_library(nanoflann_c STATIC csrc/nanoflann_c.cpp)

target_include_directories(nanoflann_c
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include   # nanoflann_c.h
    PRIVATE ${NANOFLANN_INCLUDE}                  # nanoflann.hpp
)

# Keep the static build small and warning-clean.
target_compile_options(nanoflann_c PRIVATE -Wall -Wextra -Wno-unused-parameter)

# Explicitly disable shared so callers can rely on a static .a only.
set_target_properties(nanoflann_c PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
