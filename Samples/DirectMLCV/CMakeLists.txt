cmake_minimum_required(VERSION 3.18)
project(directml_cv VERSION 0.0.0 LANGUAGES CXX)

# To avoid runtime dependency on MSVC runtime DLLs
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>)

include(FetchContent)

# ------------------------------------
# imgui
# ------------------------------------
FetchContent_Declare(
    imgui
    URL https://github.com/ocornut/imgui/archive/refs/tags/v1.91.0.zip
    URL_HASH SHA256=1974ED49FA36F37215B3641D09187B67029AE938B9B370A024C24DF69334F192
)

FetchContent_MakeAvailable(imgui)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_dx12.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
)

target_include_directories(imgui PUBLIC "${imgui_SOURCE_DIR}" "${imgui_SOURCE_DIR}/backends")

# ------------------------------------
# wil
# ------------------------------------
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

# ------------------------------------
# sample executable
# ------------------------------------

add_executable(
    directml_cv 
    WIN32
    src/main.cpp 
    src/app.cpp
    src/app.h
    src/swapchain.cpp
    src/swapchain.h
)
target_link_libraries(directml_cv PRIVATE imgui wil d3d12 dxgi)
target_compile_features(directml_cv PRIVATE cxx_std_20)
target_include_directories(directml_cv PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")