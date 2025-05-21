file(GLOB videoViewerSources *.cpp)
file(GLOB videoViewerIncludes *.h )

# Add AprilTag include directories
include_directories(${CMAKE_SOURCE_DIR}/3rd_party/apriltag)
include_directories(${CMAKE_SOURCE_DIR}/3rd_party/apriltag/common)

# Add v4l2-console include directory for NvAnalysis
include_directories(${CMAKE_SOURCE_DIR}/camera/v4l2-console)

# Find VPI package
find_package(OpenCV REQUIRED)
find_package(vpi 3.2 REQUIRED)
include_directories(${VPI_INCLUDE_DIRS})

# Build video-viewer
add_executable(video-viewer video-viewer.cpp)
target_link_libraries(video-viewer jetson-utils apriltag local-cuda)

# Build video-viewer-raw
add_executable(video-viewer-raw video-viewer-raw.cpp)
target_link_libraries(video-viewer-raw jetson-utils apriltag local-cuda)

# Build video-viewer-pva
add_executable(video-viewer-pva video-viewer-pva.cpp)
target_link_libraries(video-viewer-pva vpi opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio jetson-utils local-cuda ${VPI_LIBRARIES})

# Build one-image-viewer
add_executable(one-image-viewer one-image-viewer.cpp)
target_link_libraries(one-image-viewer jetson-utils apriltag local-cuda)

# Build one-image-viewer-pva
add_executable(one-image-viewer-pva one-image-viewer-pva.cpp)
target_link_libraries(one-image-viewer-pva vpi opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio jetson-utils local-cuda ${VPI_LIBRARIES})

install(TARGETS video-viewer video-viewer-pva one-image-viewer one-image-viewer-pva DESTINATION bin)
