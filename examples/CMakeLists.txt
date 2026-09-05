add_executable(basic basic.cpp)
target_link_libraries(basic PRIVATE GPUMemoryService gms_warnings)
add_executable(recovery recovery.cpp)
target_link_libraries(recovery PRIVATE GPUMemoryService gms_warnings)
