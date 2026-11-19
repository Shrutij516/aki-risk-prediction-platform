cmake_minimum_required(VERSION 3.18)
project(aki_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# LibTorch from the installed PyTorch wheel (pass -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')).
find_package(Torch REQUIRED)

add_executable(aki_infer aki_infer.cpp)
target_link_libraries(aki_infer PRIVATE ${TORCH_LIBRARIES})
target_compile_options(aki_infer PRIVATE ${TORCH_CXX_FLAGS})
# Find libtorch*.dylib/.so at run time without DYLD_LIBRARY_PATH.
get_filename_component(TORCH_LIB_DIR "${TORCH_INSTALL_PREFIX}/lib" ABSOLUTE)
set_target_properties(aki_infer PROPERTIES BUILD_RPATH "${TORCH_LIB_DIR}")
