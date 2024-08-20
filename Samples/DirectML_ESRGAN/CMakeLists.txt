cmake_minimum_required(VERSION 3.18)
project(directml_esrgan VERSION 0.0.0 LANGUAGES CXX)

# To avoid runtime dependency on MSVC runtime DLLs
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>)

include(FetchContent)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

set(TARGET_ARCH ${CMAKE_CXX_COMPILER_ARCHITECTURE_ID})
if(TARGET_ARCH STREQUAL AMD64)
    set(TARGET_ARCH x64)
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
# onnxruntime
# -----------------------------------------------------------------------------
FetchContent_Declare(
    ort
    URL https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/1.19.0
    URL_HASH SHA256=FD4E65EBD23385FD8ED792F0C859D2AA255DC1761D0E5761087568FF255187AC
)

FetchContent_MakeAvailable(ort)

set(ort_bin_dir "${ort_SOURCE_DIR}/runtimes/win-${TARGET_ARCH}/native")

add_library(ort INTERFACE)
target_include_directories(ort INTERFACE "${ort_SOURCE_DIR}/build/native/include")
target_link_libraries(ort INTERFACE "${ort_bin_dir}/onnxruntime.lib")

# -----------------------------------------------------------------------------
# directx-headers - to get the latest DXCore header with GENERIC_ML GUID
# -----------------------------------------------------------------------------
FetchContent_Declare(
    dxheaders
    GIT_REPOSITORY https://github.com/microsoft/DirectX-Headers
    GIT_TAG de28d93dfa9ebf3e473127c1c657e1920a5345ee # v1.613.1
)

FetchContent_MakeAvailable(dxheaders)

set(ort_bin_dir "${ort_SOURCE_DIR}/runtimes/win-${TARGET_ARCH}/native")

add_library(dxheaders INTERFACE)
target_include_directories(dxheaders INTERFACE ${dxheaders_SOURCE_DIR}/include/directx)
target_link_libraries(dxheaders INTERFACE Microsoft::DirectX-Guids)

# -----------------------------------------------------------------------------
# cxxopts - for parsing command line arguments
# -----------------------------------------------------------------------------

FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts
    GIT_TAG v3.2.1
)

set(CXXOPTS_BUILD_EXAMPLES OFF CACHE INTERNAL "Set to ON to build examples")
set(CXXOPTS_BUILD_TESTS OFF CACHE INTERNAL "Set to ON to build tests")
set(CXXOPTS_ENABLE_INSTALL OFF CACHE INTERNAL "Generate the install target")
FetchContent_MakeAvailable(cxxopts)

# -----------------------------------------------------------------------------
# directml
# -----------------------------------------------------------------------------
FetchContent_Declare(
    dml
    URL https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.15.1
    URL_HASH SHA256=F43EB3D8B39F588B6CF049EAF1E0EA6006774EB8C93C661A1DE7721E674F7F56
)

FetchContent_MakeAvailable(dml)

set(dml_bin_dir "${dml_SOURCE_DIR}/bin/${TARGET_ARCH}-win")

add_library(dml INTERFACE)
target_include_directories(dml INTERFACE "${dml_SOURCE_DIR}/include")
target_link_libraries(dml INTERFACE "${dml_bin_dir}/directml.lib")

# -----------------------------------------------------------------------------
# main sample source
# -----------------------------------------------------------------------------

add_executable(directml_esrgan main.cpp)
target_link_libraries(directml_esrgan PRIVATE wil ort dml d3d12 dxcore dxheaders cxxopts)
target_compile_features(directml_esrgan PRIVATE cxx_std_20)
target_include_directories(directml_esrgan PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_custom_command(
    TARGET directml_esrgan 
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ort_bin_dir}/onnxruntime.dll" $<TARGET_FILE_DIR:directml_esrgan>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dml_bin_dir}/directml.dll" $<TARGET_FILE_DIR:directml_esrgan>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/esrgan.onnx" $<TARGET_FILE_DIR:directml_esrgan>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/zebra.jpg" $<TARGET_FILE_DIR:directml_esrgan>
)