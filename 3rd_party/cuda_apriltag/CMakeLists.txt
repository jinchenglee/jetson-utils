cmake_minimum_required(VERSION 3.25 FATAL_ERROR)  # Ensure user has a compatible version of CMake

project(cuda_apriltag LANGUAGES CUDA CXX)

# Set CUDA architecture for Jetson Orin (Ampere)
set(CMAKE_CUDA_ARCHITECTURES "87")
# Set CUDA standard to C++20
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON) # Ensure C++20 is strictly required

cmake_policy(SET CMP0104 NEW)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Ensure C++20 is strictly required
set(CMAKE_CXX_EXTENSIONS OFF)

# Base CUDA flags that are always needed
set(BASE_CUDA_FLAGS "--expt-relaxed-constexpr --extended-lambda --std c++20")

# Look for required packages.
find_package(CUDA 12.6 REQUIRED)
find_package(glog REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenCV 4.11 REQUIRED)

if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")

    if(CMAKE_CUDA_COMPILER MATCHES "nvcc")
        message(STATUS "Using NVCC as CUDA compiler")
        if(CMAKE_BUILD_TYPE MATCHES "Debug")
            set(CMAKE_CUDA_FLAGS "${BASE_CUDA_FLAGS} -g")
        else()
            set(CMAKE_CUDA_FLAGS "${BASE_CUDA_FLAGS} -Xptxas -O3")
        endif()
    else()
        message(STATUS "Using a different CUDA compiler")
        if(CMAKE_BUILD_TYPE MATCHES "Debug")
            set(CMAKE_CUDA_FLAGS "${BASE_CUDA_FLAGS} -g -gdwarf-4 -O0")
        else()
            set(CMAKE_CUDA_FLAGS "${BASE_CUDA_FLAGS} -O3")
        endif()
    endif()
else()
    message(FATAL_ERROR "CUDA compiler not found. Please set CMAKE_CUDA_COMPILER.")
endif()

# Add OpenCV include directories to CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${OpenCV_INCLUDE_DIRS}")

if(ENABLE_ASAN)
    message(STATUS "AddressSanitizer is enabled")
    set(ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS} ${ASAN_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${COMMON_FLAGS} ${ASAN_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ASAN_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${ASAN_FLAGS}")
endif()

if(ENABLE_MSAN)
    message(STATUS "MemorySanitizer is enabled")
    set(MSAN_FLAGS "-fsanitize=memory -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer -fsanitize-memory-use-after-dtor")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS} ${MSAN_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${COMMON_FLAGS} ${MSAN_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MSAN_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${MSAN_FLAGS}")
endif()

# Set source files
set(CUDA_APRILTAG_SOURCES
    apriltag_detect.cu
    apriltag_gpu.cu
    apriltag_utils.cu
    cuda_frc971.cu
    gpu_detector_test.cu
    labeling_allegretti_2019_BKE.cu
    line_fit_filter.cu
    points.cu
    threshold.cu
)

# Create shared library
add_library(cuda_apriltag SHARED ${CUDA_APRILTAG_SOURCES})

# Set include directories
target_include_directories(cuda_apriltag
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../apriltag
        ${CMAKE_CURRENT_SOURCE_DIR}/../apriltag/common
        ${OpenCV_INCLUDE_DIRS}
    PRIVATE
        ${CUDA_INCLUDE_DIRS}
)

# Link against OpenCV and AprilTag
target_link_libraries(cuda_apriltag
    PUBLIC
        apriltag
    PRIVATE
        ${OpenCV_LIBS}
)

# Set compile definitions
target_compile_definitions(cuda_apriltag
    PRIVATE
        CUDA_APRILTAG_EXPORTS
        APRILTAG_EXPORTS
    PUBLIC
        CUDA_APRILTAG_EXPORTS
        APRILTAG_EXPORTS
)

# Set CUDA specific properties
set_target_properties(cuda_apriltag PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)
 
# Add the test executable
add_executable(gpu_detector_test gpu_detector_test.cu)
target_link_libraries(gpu_detector_test
    PRIVATE
        apriltag
        cuda_apriltag
        ${OpenCV_LIBS}
        glog::glog
        GTest::GTest
        ${CUDA_LIBRARIES}
)
