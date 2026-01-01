# Copyright (C) 2019-2023 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

set(base_depend_libs
        knowhere
        )

# Find OpenBLAS if available (needed by milvus-common)
find_package(OpenBLAS QUIET)
if(OpenBLAS_FOUND)
    list(APPEND base_depend_libs OpenBLAS::OpenBLAS)
endif()

if(WITH_PROFILER)
    set(base_depend_libs
        ${base_depend_libs}
        tcmalloc_and_profiler
    )
endif()

#==============================================================================
# Sparse benchmark using Catch2 (doesn't require HDF5 or GTest)
find_package(Catch2)
if(NOT Catch2_FOUND)
    message("Catch2 not found, fetching Catch2 from github")
    Include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.3.1
    )
    FetchContent_MakeAvailable(Catch2)
endif()

add_executable(benchmark_sparse hdf5/benchmark_sparse.cpp)
target_link_libraries(benchmark_sparse PRIVATE
        ${base_depend_libs}
        Catch2::Catch2WithMain
        atomic)
install(TARGETS benchmark_sparse DESTINATION unittest)

#==============================================================================
# HDF5 and GTest-dependent benchmarks
find_package(HDF5)
find_package(GTest)

if(HDF5_FOUND AND GTest_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
    set(unittest_libs
            GTest::gtest
            GTest::gtest_main)

    set(depend_libs
            ${base_depend_libs}
            ${HDF5_LIBRARIES}
            )

    macro(benchmark_test target file)
        set(FILE_SRCS ${file})
        add_executable(${target} ${FILE_SRCS})
        target_link_libraries(${target} ${depend_libs} ${unittest_libs})

        # this is needed for clang in Debug compilation mode
        target_link_libraries(${target} atomic)

        install(TARGETS ${target} DESTINATION unittest)
    endmacro()

    benchmark_test(benchmark_binary                hdf5/benchmark_binary.cpp)
    benchmark_test(benchmark_binary_range          hdf5/benchmark_binary_range.cpp)
    benchmark_test(benchmark_float                 hdf5/benchmark_float.cpp)
    benchmark_test(benchmark_float_bitset          hdf5/benchmark_float_bitset.cpp)
    benchmark_test(benchmark_float_qps             hdf5/benchmark_float_qps.cpp)
    benchmark_test(benchmark_float_range           hdf5/benchmark_float_range.cpp)
    benchmark_test(benchmark_float_range_bitset    hdf5/benchmark_float_range_bitset.cpp)
    benchmark_test(benchmark_simd_qps              hdf5/benchmark_simd_qps.cpp)
benchmark_test(benchmark_zipper_merge          hdf5/benchmark_zipper_merge.cpp)

    benchmark_test(gen_hdf5_file hdf5/gen_hdf5_file.cpp)
    benchmark_test(gen_fbin_file hdf5/gen_fbin_file.cpp)
else()
    if(NOT HDF5_FOUND)
        message(WARNING "HDF5 not found. HDF5-dependent benchmarks will not be built.")
    endif()
    if(NOT GTest_FOUND)
        message(WARNING "GTest not found. GTest-dependent benchmarks will not be built.")
    endif()
endif()
