file(GLOB videoViewerSources *.cpp)
file(GLOB videoViewerIncludes *.h )

# Add AprilTag include directories
include_directories(${CMAKE_SOURCE_DIR}/3rd_party/apriltag)
include_directories(${CMAKE_SOURCE_DIR}/3rd_party/apriltag/common)

# Build video-viewer
add_executable(video-viewer video-viewer.cpp)
target_link_libraries(video-viewer jetson-utils apriltag)

# Build one-image-viewer
add_executable(one-image-viewer one-image-viewer.cpp)
target_link_libraries(one-image-viewer jetson-utils apriltag)

install(TARGETS video-viewer one-image-viewer DESTINATION bin)
