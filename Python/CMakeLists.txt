cmake_minimum_required(VERSION 3.15)

message("DML_PATH=${DML_PATH}")
message("DMLX_PATH=${DMLX_PATH}")

project(pydirectml)

if(MSVC)
# Enables some MSVC C++17 and newer features
set(PYBIND11_CPP_STANDARD /std:c++latest)
endif()

add_subdirectory(pybind11)

pybind11_add_module(
    pydirectml
    src/util.cpp
    src/module.cpp
    src/device.cpp
    )

target_compile_options(pydirectml PRIVATE /W4 /WX)
target_include_directories(pydirectml PRIVATE pybind11/include ${DML_PATH}/include ${DMLX_PATH})
target_link_directories(pydirectml PRIVATE ${DML_PATH}/bin/x64-win)
target_link_libraries(pydirectml PRIVATE dxgi.lib d3d12.lib directml.lib)
