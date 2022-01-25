FetchContent_Declare(
    direct3d
    URL https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/1.4.10
    URL_HASH SHA256=8e7fc23973aedcc1d98584c7b260d0073b6fa6da3552da386566410ccea0ad26
)

FetchContent_MakeAvailable(direct3d)

add_library(direct3d INTERFACE IMPORTED)
add_library(Microsoft::Direct3D12 ALIAS direct3d)

target_include_directories(direct3d INTERFACE ${direct3d_SOURCE_DIR}/build/native/include)

if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(direct3d_PLATFORM ${CMAKE_GENERATOR_PLATFORM})
    if(NOT direct3d_PLATFORM)
        set(direct3d_PLATFORM ${CMAKE_VS_PLATFORM_NAME})
    endif()
    if(NOT direct3d_PLATFORM)
        set(direct3d_PLATFORM x64)
    endif()

    # The agility SDK does not redistribute import libraries since the inbox version of
    # d3d12.dll provides all necessary exports.
    target_link_libraries(direct3d INTERFACE d3d12)

    # This must match the SDK version of the agility SDK.
    target_compile_definitions(direct3d INTERFACE DIRECT3D_AGILITY_SDK_VERSION=4)

    # It is preferred to place the redist DLLs in a subfolder of the application to avoid mismatching
    # the debug layer and product interfaces.
    target_compile_definitions(direct3d INTERFACE DIRECT3D_AGILITY_SDK_PATH=u8"./D3D12/")


    if (${direct3d_PLATFORM} MATCHES Win32)
        set(direct3d_bin_dir ${direct3d_SOURCE_DIR}/build/native/bin/win32)
    elseif (${direct3d_PLATFORM} MATCHES x64)
        set(direct3d_bin_dir ${direct3d_SOURCE_DIR}/build/native/bin/x64)
    elseif (${direct3d_PLATFORM} MATCHES ARM)
        set(direct3d_bin_dir ${direct3d_SOURCE_DIR}/build/native/bin/arm)
    elseif (${direct3d_PLATFORM} MATCHES ARM64)
        set(direct3d_bin_dir ${direct3d_SOURCE_DIR}/build/native/bin/arm64)
    endif()

    install(
        FILES 
            ${direct3d_bin_dir}/D3D12Core.dll
            ${direct3d_bin_dir}/d3d12SDKLayers.dll
            ${direct3d_bin_dir}/d3dconfig.exe
        DESTINATION bin/D3D12
    )

    set_property(TARGET direct3d PROPERTY CORE_DLL_PATH ${direct3d_bin_dir}/D3D12Core.dll)
    set_property(TARGET direct3d PROPERTY DEBUG_DLL_PATH ${direct3d_bin_dir}/d3d12SDKLayers.dll)
    set_property(TARGET direct3d PROPERTY D3DCONFIG_EXE_PATH ${direct3d_bin_dir}/d3dconfig.exe)
    set_property(TARGET direct3d PROPERTY SUBDIR_PATH "D3D12")

endif()