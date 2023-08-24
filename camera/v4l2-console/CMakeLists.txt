
file(GLOB v4l2ConsoleSources *.cpp)
file(GLOB v4l2ConsoleIncludes *.h )

cuda_add_library(local-cuda ./NvAnalysis.cu)

add_executable(v4l2-console ${v4l2ConsoleSources} 
    #${CMAKE_SOURCE_DIR}/3rd_party/apriltag/tag16h5.c
    )
target_include_directories(v4l2-console PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>/3rd_party/apriltag)
target_link_libraries(v4l2-console jetson-utils local-cuda apriltag)
