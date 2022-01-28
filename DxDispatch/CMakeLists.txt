cmake_minimum_required(VERSION 3.18)

project(dxdispatch VERSION 0.4.0 LANGUAGES CXX)

# Statically link runtime library to avoid runtime dependency on Visual C++ redistributable
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>)

# ==============================================================================
# External Libraries
# ==============================================================================
include(FetchContent)
include(cmake/gsl.cmake)
include(cmake/rapidjson.cmake)
include(cmake/fmt.cmake)
include(cmake/wil.cmake)
include(cmake/directml.cmake)
include(cmake/directmlx.cmake)
include(cmake/direct3d.cmake)
include(cmake/dxheaders.cmake)
include(cmake/pix.cmake)
include(cmake/gtest.cmake)
include(cmake/dxc.cmake)
include(cmake/half.cmake)
include(cmake/cxxopts.cmake)

# ==============================================================================
# Model Library
# ==============================================================================
add_library(
    model STATIC 
    src/model/JsonParsers.cpp 
    src/model/Model.cpp
)

target_link_libraries(
    model 
    PRIVATE 
    fmt::fmt-header-only
    Microsoft::DirectMLX
    PUBLIC
    Microsoft.GSL::GSL
    rapidjson::rapidjson
    Microsoft::DirectML
    Microsoft::WIL
    Half::Half
)

target_compile_features(model PRIVATE cxx_std_17)
target_precompile_headers(model PRIVATE src/model/pch.h)
target_include_directories(model INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/model)

# ==============================================================================
# Main Executable
# ==============================================================================
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/dxdispatch/config.h.in config.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/dxdispatch/dxdispatch.rc.in dxdispatch.rc)

add_executable(
    dxdispatch
    src/dxdispatch/Adapter.cpp
    src/dxdispatch/Device.cpp
    src/dxdispatch/main.cpp
    src/dxdispatch/DmlDispatchable.cpp
    src/dxdispatch/HlslDispatchable.cpp
    src/dxdispatch/Executor.cpp
    src/dxdispatch/CommandLineArgs.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/config.h
    ${CMAKE_CURRENT_BINARY_DIR}/dxdispatch.rc
)

target_include_directories(dxdispatch PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(
    dxdispatch 
    PRIVATE 
    Microsoft.GSL::GSL
    fmt::fmt-header-only
    Microsoft::WIL
    Microsoft::DirectML
    Microsoft::Direct3D12
    Microsoft::PIX
    Microsoft::DirectMLX
    Microsoft::DirectX-Headers
    Microsoft::DirectX-Guids
    Microsoft::DXC
    dxcore
    model
    cxxopts
)

target_compile_features(dxdispatch PRIVATE cxx_std_17)
target_precompile_headers(dxdispatch PRIVATE src/dxdispatch/pch.h)

# Copy DLLs into executable's build output directory for debugging purposes.
add_custom_command(
    TARGET dxdispatch 
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:Microsoft::DirectML> $<TARGET_FILE_DIR:dxdispatch>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_PROPERTY:Microsoft::DirectML,DEBUG_DLL_PATH> $<TARGET_FILE_DIR:dxdispatch>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:Microsoft::PIX> $<TARGET_FILE_DIR:dxdispatch>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:Microsoft::DXC> $<TARGET_FILE_DIR:dxdispatch>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_PROPERTY:Microsoft::DXC,DXIL_PATH> $<TARGET_FILE_DIR:dxdispatch>
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:dxdispatch>/$<TARGET_PROPERTY:Microsoft::Direct3D12,SUBDIR_PATH>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_PROPERTY:Microsoft::Direct3D12,CORE_DLL_PATH> $<TARGET_FILE_DIR:dxdispatch>/$<TARGET_PROPERTY:Microsoft::Direct3D12,SUBDIR_PATH>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_PROPERTY:Microsoft::Direct3D12,DEBUG_DLL_PATH> $<TARGET_FILE_DIR:dxdispatch>/$<TARGET_PROPERTY:Microsoft::Direct3D12,SUBDIR_PATH>
)

install(
    FILES 
    $<TARGET_FILE:dxdispatch> 
    ${CMAKE_CURRENT_SOURCE_DIR}/ThirdPartyNotices.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/doc/Guide.md
    DESTINATION bin
)

install(
    DIRECTORY 
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    DESTINATION bin
)

# ==============================================================================
# Tests
# ==============================================================================
enable_testing()
include(GoogleTest)

add_executable(
    jsontests 
    src/test/JsonParserTests.cpp
)

target_compile_features(jsontests PRIVATE cxx_std_17)
target_link_libraries(
    jsontests 
    PRIVATE 
    gtest_main 
    fmt::fmt-header-only
    Microsoft::DirectMLX
    model
)
target_include_directories(jsontests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/dxdispatch)
gtest_discover_tests(jsontests)

add_test(test_dml_convolution dxdispatch ${CMAKE_CURRENT_SOURCE_DIR}/models/dml_convolution.json)
set_tests_properties(test_dml_convolution PROPERTIES PASS_REGULAR_EXPRESSION "Resource 'output': 6, 8, 12, 14")

add_test(test_dml_cumulative_product dxdispatch ${CMAKE_CURRENT_SOURCE_DIR}/models/dml_cumulative_product.json)
set_tests_properties(test_dml_cumulative_product PROPERTIES PASS_REGULAR_EXPRESSION "Resource 'Out': 2, 8, 64, 192")

add_test(test_dml_element_wise_add dxdispatch ${CMAKE_CURRENT_SOURCE_DIR}/models/dml_element_wise_add.json)
set_tests_properties(test_dml_element_wise_add PROPERTIES PASS_REGULAR_EXPRESSION "Resource 'Out': 6, 10, -2")

add_test(test_dml_element_wise_add1 dxdispatch ${CMAKE_CURRENT_SOURCE_DIR}/models/dml_element_wise_add1.json)
set_tests_properties(test_dml_element_wise_add1 PROPERTIES PASS_REGULAR_EXPRESSION "Resource 'Out': 6, 10, -0.432332")

add_test(test_dml_element_wise_clip dxdispatch ${CMAKE_CURRENT_SOURCE_DIR}/models/dml_element_wise_clip.json)
set_tests_properties(test_dml_element_wise_clip PROPERTIES PASS_REGULAR_EXPRESSION "Resource 'A': -2.5, -2.5, -2, -1, 0, 1, 2, 2.5, 2.5")

add_test(test_dml_element_wise_identity dxdispatch ${CMAKE_CURRENT_SOURCE_DIR}/models/dml_element_wise_identity.json)
set_tests_properties(test_dml_element_wise_identity PROPERTIES PASS_REGULAR_EXPRESSION "Resource 'Out': 12, 14, 16")

add_test(test_dml_fill_value_sequence dxdispatch ${CMAKE_CURRENT_SOURCE_DIR}/models/dml_fill_value_sequence.json)
set_tests_properties(test_dml_fill_value_sequence PROPERTIES PASS_REGULAR_EXPRESSION "Resource 'Out': 3.2, 4.7, 6.2, 7.7, 9.2")

add_test(test_dml_join dxdispatch ${CMAKE_CURRENT_SOURCE_DIR}/models/dml_join.json)
set_tests_properties(test_dml_join PROPERTIES PASS_REGULAR_EXPRESSION "Resource 'Out': 1, 2, 3, 100, 115, 5, 6, 7, 8, 9")

add_test(test_dml_reduce dxdispatch ${CMAKE_CURRENT_SOURCE_DIR}/models/dml_reduce.json)
set_tests_properties(test_dml_reduce PROPERTIES PASS_REGULAR_EXPRESSION "Resource 'output': 6, 15, 24")

add_test(test_dml_slice dxdispatch ${CMAKE_CURRENT_SOURCE_DIR}/models/dml_slice.json)
set_tests_properties(test_dml_slice PROPERTIES PASS_REGULAR_EXPRESSION "Resource 'output': 7, 9, 12, 14, 17, 19")

add_test(test_dml_split dxdispatch ${CMAKE_CURRENT_SOURCE_DIR}/models/dml_split.json)
set_tests_properties(test_dml_split PROPERTIES PASS_REGULAR_EXPRESSION "Resource 'Out1': 1, 2\nResource 'Out2': 3, 4\nResource 'Out3': 5, 6")

add_test(test_dml_upsample_2d dxdispatch ${CMAKE_CURRENT_SOURCE_DIR}/models/dml_upsample_2d.json)
set_tests_properties(test_dml_upsample_2d PROPERTIES PASS_REGULAR_EXPRESSION "Resource 'output': 1, 1.25, 1.75, 2, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3, 3.25, 3.75, 4")

add_test(test_owned_by_dml dxdispatch ${CMAKE_CURRENT_SOURCE_DIR}/models/owned_by_dml.json)
set_tests_properties(test_owned_by_dml PROPERTIES PASS_REGULAR_EXPRESSION "Resource 'Out': 6, 10, -2")