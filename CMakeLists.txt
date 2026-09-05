cmake_minimum_required(VERSION 3.24)
project(GPUMemoryService VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(GNUInstallDirs)

option(GMS_BUILD_TESTS "Build tests" ON)
option(GMS_BUILD_EXAMPLES "Build examples" ON)
option(GMS_BUILD_TOOLS "Build tools" ON)
option(GMS_BUILD_BENCHMARKS "Build benchmarks" ON)
option(GMS_ENABLE_CUDA "Build CUDA-backed adapter & real GPU proof" ON)

add_library(gms_warnings INTERFACE)
if(MSVC)
  target_compile_options(gms_warnings INTERFACE /W4 /WX)
else()
  target_compile_options(gms_warnings INTERFACE -Wall -Wextra -Wpedantic -Werror)
endif()

add_library(GPUMemoryService STATIC
  src/region_allocator.cpp
  src/pressure.cpp
  src/allocation.cpp
  src/persistence.cpp
  src/protocol.cpp
  src/governed_memory.cpp
  src/host_backend.cpp
)
add_library(GPUMemoryService::GPUMemoryService ALIAS GPUMemoryService)
target_include_directories(GPUMemoryService PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_compile_definitions(GPUMemoryService PUBLIC GMS_VERSION_STR="1.0.0")
if(MSVC)
  target_compile_options(GPUMemoryService PRIVATE /W4 /WX)
else()
  target_compile_options(GPUMemoryService PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# ---- CUDA-backed adapter + real GPU proof (optional) ----
if(GMS_ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 120)
    add_library(GPUMemoryServiceCuda STATIC src/cuda/cuda_backend.cu)
    add_library(GPUMemoryService::GPUMemoryServiceCuda ALIAS GPUMemoryServiceCuda)
    target_link_libraries(GPUMemoryServiceCuda PUBLIC GPUMemoryService CUDA::cudart)
    target_include_directories(GPUMemoryServiceCuda PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
      "$<INSTALL_INTERFACE:include>")
    set_target_properties(GPUMemoryServiceCuda PROPERTIES CUDA_ARCHITECTURES 120
      CUDA_COMPILER_OPTIONS "-Xcompiler=/W4;-Xcompiler=/WX")
    set(GMS_HAS_CUDA TRUE)
  else()
    set(GMS_HAS_CUDA FALSE)
  endif()
endif()

# ---- Install / export / package config ----
include(CMakePackageConfigHelpers)

set(GMS_CONFIG_CUDA FALSE)
if(GMS_HAS_CUDA)
  set(GMS_CONFIG_CUDA TRUE)
endif()

install(TARGETS GPUMemoryService EXPORT GPUMemoryServiceTargets
        ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/gpu_memory_service DESTINATION include)

if(GMS_HAS_CUDA)
  install(TARGETS GPUMemoryServiceCuda EXPORT GPUMemoryServiceTargets
          ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
endif()

install(EXPORT GPUMemoryServiceTargets NAMESPACE GPUMemoryService::
        DESTINATION lib/cmake/GPUMemoryService)
configure_package_config_file(cmake/GPUMemoryServiceConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/GPUMemoryServiceConfig.cmake"
  INSTALL_DESTINATION lib/cmake/GPUMemoryService)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/GPUMemoryServiceConfigVersion.cmake"
  VERSION 1.0.0 COMPATIBILITY SameMajorVersion)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/GPUMemoryServiceConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/GPUMemoryServiceConfigVersion.cmake"
  DESTINATION lib/cmake/GPUMemoryService)

if(GMS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
# ---- Tooling, examples, benchmarks ----
# ---- Reference multi-process deployment ----
add_executable(coordinator coordinator/coordinator.cpp)
target_link_libraries(coordinator PRIVATE GPUMemoryService gms_warnings ws2_32)
target_include_directories(coordinator PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
if(GMS_HAS_CUDA)
  add_executable(worker worker/worker.cpp)
  target_link_libraries(worker PRIVATE GPUMemoryService gms_warnings CUDA::cudart ws2_32)
  target_include_directories(worker PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

if(GMS_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
if(GMS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
if(GMS_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()