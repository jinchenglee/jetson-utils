
file(GLOB v4l2ConsoleSources *.cpp)
file(GLOB v4l2ConsoleIncludes *.h )

cuda_add_library(local-cuda ./NvAnalysis.cu)

add_executable(v4l2-console ${v4l2ConsoleSources})
target_link_libraries(v4l2-console jetson-utils local-cuda)
