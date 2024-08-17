cmake_minimum_required(VERSION 3.18)
project(directml_cv VERSION 0.0.0 LANGUAGES CXX)

# To avoid runtime dependency on MSVC runtime DLLs
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>)

include(FetchContent)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
  endif()

# -----------------------------------------------------------------------------
# wil - Windows-related helper types/macros
# -----------------------------------------------------------------------------
FetchContent_Declare(
    wil
    URL https://github.com/microsoft/wil/archive/refs/tags/v1.0.240803.1.zip
    URL_HASH SHA256=353D2D7F2ACEA5642689A1BA85213C1AC6816457038B54AC02158B893E0F389F
)

FetchContent_GetProperties(wil)
if(NOT wil_POPULATED)
    FetchContent_Populate(wil)
endif()

add_library(wil INTERFACE)
target_include_directories(wil INTERFACE "${wil_SOURCE_DIR}/include")

# -----------------------------------------------------------------------------
# main sample source
# -----------------------------------------------------------------------------

add_executable(directml_cv main.cpp)
target_link_libraries(directml_cv PRIVATE wil d3d12 dxgi)
target_compile_features(directml_cv PRIVATE cxx_std_20)
target_include_directories(directml_cv PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")