add_executable(gms gms.cpp)
target_link_libraries(gms PRIVATE GPUMemoryService gms_warnings)
