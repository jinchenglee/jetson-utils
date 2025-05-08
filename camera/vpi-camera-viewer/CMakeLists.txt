cmake_minimum_required(VERSION 2.8)

# find required packages
find_package(VPI 3.2 REQUIRED)

# add the executable
add_executable(vpi-camera-viewer ../vpi-camera-viewer.cpp)

# link against required libraries
target_link_libraries(vpi-camera-viewer
    jetson-utils
    vpi
)

# include directories
target_include_directories(vpi-camera-viewer PRIVATE
    ${VPI_INCLUDE_DIRS}
)

# Add VPI library path
link_directories(${VPI_LIBRARY_DIRS}) 