# =============================================================================
# Helper function to introduce a target representing a D3D12 implementation.
# This target will implicitly depend on a system version of DXCore (Windows/WSL)
# or DXGI (GDK).
#
# The main function is `add_d3d12_target`, which has the following parameters:
#
# - CACHE_PREFIX     : string used to prefix cache variables associated with the function.
# 
# The following cache variables are defined after calling the main function 
# (all variable names are prefixed with the value of CACHE_PREFIX):
#
# - D3D12_TYPE
#       nuget  : Use NuGet distribution of D3D12 (Agility SDK). Redistributable.
#       gdk    : Use system-installed GDK. System dependency (no redistributable files).
#       winsdk : Use Windows SDK. System dependency (no redistributable files).
#       wsl    : Link against WSL system libraries. System dependency (no redistributable files).
#
# - D3D12_NUGET_ID : ID of the D3D12 NuGet package (TYPE == nuget).
# - D3D12_NUGET_VERSION : Version of the D3D12 NuGet package (TYPE == nuget).
# - D3D12_NUGET_HASH : SHA256 hash of the D3D12 NuGet package (TYPE == nuget).
# - D3D12_HEADERS_TAG : Git commit/tag for the open-source DX headers.
# =============================================================================

include_guard()
include(FetchContent)
include(${CMAKE_CURRENT_LIST_DIR}/helper_platform.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/helper_redist.cmake)

# -----------------------------------------------------------------------------
# Add cache variables.
# -----------------------------------------------------------------------------
function(init_d3d12_cache_variables prefix)
    if(NOT prefix)
        message(FATAL_ERROR "The prefix must not be an empty string.")
    endif()

    # <PREFIX>_D3D12_TYPE
    if(TARGET_XBOX)
        set(default_type gdk)
    elseif(TARGET_WINDOWS)
        set(default_type nuget)
    else()
        set(default_type wsl)
    endif()
    set(${prefix}_D3D12_TYPE 
        "${default_type}" 
        CACHE STRING "D3D12 dependency type"
    )
    set_property(CACHE ${prefix}_D3D12_TYPE PROPERTY STRINGS nuget gdk winsdk wsl)

    # <PREFIX>_D3D12_NUGET_ID
    set(${prefix}_D3D12_NUGET_ID
        Microsoft.Direct3D.D3D12
        CACHE STRING "ID of the D3D12 NuGet package (TYPE == nuget)."
    )

    # <PREFIX>_D3D12_NUGET_VERSION
    set(${prefix}_D3D12_NUGET_VERSION
        1.611.2
        CACHE STRING "Version of the D3D12 NuGet package (TYPE == nuget)."
    )

    # <PREFIX>_D3D12_NUGET_HASH
    set(${prefix}_D3D12_NUGET_HASH 
        414858c5cf25e43022938ab992d78a9639b1a9a48efe5b4a05fca2df9294388c
        CACHE STRING "SHA256 hash of the D3D12 NuGet package (TYPE == nuget)."
    )

    # <PREFIX>_D3D12_HEADERS_TAG
    set(${prefix}_D3D12_HEADERS_TAG
        de28d93dfa9ebf3e473127c1c657e1920a5345ee # v1.613.1
        CACHE STRING "Git commit/tag for headers in the DirectX-Headers repo."
    )
endfunction()

# -----------------------------------------------------------------------------
# Init using a NuGet distribution.
# -----------------------------------------------------------------------------
function(init_d3d12_target_nuget target_name pkg_id pkg_version pkg_hash)
    if(NOT TARGET_WINDOWS)
        message(FATAL_ERROR "The NuGet version of D3D12 only works on Windows")
    endif()

    if(NOT pkg_version MATCHES "[0-9]+\\.([0-9]+)\\.[0-9]+.*")
        message(FATAL_ERROR "Could not parse D3D12 package version")
    endif()
    set(agility_sdk_version ${CMAKE_MATCH_1})

    set(content ${target_name}_content)
    FetchContent_Declare(
        ${content}
        URL "https://www.nuget.org/api/v2/package/${pkg_id}/${pkg_version}"
        URL_HASH SHA256=${pkg_hash}
    )
    FetchContent_MakeAvailable(${content})

    if (TARGET_ARCH STREQUAL X86)
        set(bin_path ${${content}_SOURCE_DIR}/build/native/bin/win32)
    elseif (TARGET_ARCH STREQUAL X64)
        set(bin_path ${${content}_SOURCE_DIR}/build/native/bin/x64)
    elseif (TARGET_ARCH STREQUAL ARM)
        set(bin_path ${${content}_SOURCE_DIR}/build/native/bin/arm)
    elseif (TARGET_ARCH STREQUAL ARM64)
        set(bin_path ${${content}_SOURCE_DIR}/build/native/bin/arm64)
    endif()

    # The agility SDK does not redistribute import libraries since the inbox version of
    # d3d12.dll provides all necessary exports.
    target_link_libraries(${target_name} INTERFACE 
        Microsoft::DirectX-Headers 
        Microsoft::DirectX-Guids 
    )
    
    target_compile_definitions(${target_name} INTERFACE DIRECT3D_AGILITY_SDK_VERSION=${agility_sdk_version})

    target_append_redist_file(${target_name} "${bin_path}/D3D12Core.dll" "D3D12/D3D12Core.dll")
    target_append_redist_file(${target_name} "${bin_path}/d3d12SDKLayers.dll" "D3D12/d3d12SDKLayers.dll")
    target_append_redist_file(${target_name} "${bin_path}/d3dconfig.exe" "D3D12/d3dconfig.exe")

    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "NuGet (${pkg_id}.${pkg_version})")
endfunction()

# -----------------------------------------------------------------------------
# Init using a GDK.
# -----------------------------------------------------------------------------
function(init_d3d12_target_gdk target_name)
    if(NOT TARGET_XBOX)
        message(FATAL_ERROR "The GDK version of D3D12 only works on Xbox")
    endif()

    target_link_libraries(
        ${target_name}
        INTERFACE
        d3d12_${TARGET_XBOX_FILE_SUFFIX}.lib
        xg_${TARGET_XBOX_FILE_SUFFIX}.lib
        xgameruntime.lib
        xgameplatform.lib
        xmem.lib
    )

    target_compile_definitions(${target_name} INTERFACE DIRECT3D_AGILITY_SDK_VERSION=0)
    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "GDK")
endfunction()

# -----------------------------------------------------------------------------
# Init using Windows SDK.
# -----------------------------------------------------------------------------
function(init_d3d12_target_winsdk target_name)
    if(NOT TARGET_WINDOWS)
        message(FATAL_ERROR "The SDK version of D3D12 only works on Windows")
    endif()

    target_link_libraries(${target_name} INTERFACE Microsoft::DirectX-Headers Microsoft::DirectX-Guids)
    target_compile_definitions(${target_name} INTERFACE DIRECT3D_AGILITY_SDK_VERSION=0)
    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "Windows SDK")
endfunction()

# -----------------------------------------------------------------------------
# Init using WSL system libraries and open-source headers.
# -----------------------------------------------------------------------------
function(init_d3d12_target_wsl target_name)
    target_link_libraries(${target_name} INTERFACE Microsoft::DirectX-Headers Microsoft::DirectX-Guids)
    target_link_directories(${target_name} INTERFACE /usr/lib/wsl/lib)
    target_include_directories(${target_name} INTERFACE ${dxheaders_SOURCE_DIR}/include/directx)
    target_compile_definitions(${target_name} INTERFACE DIRECT3D_AGILITY_SDK_VERSION=0)
    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "WSL")
endfunction()

# -----------------------------------------------------------------------------
# Main function to add the target.
# -----------------------------------------------------------------------------
function(add_d3d12_target target_name)
    # Parse function args.
    set(params CACHE_PREFIX)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "${params}" "")

    # Parse cached args.
    init_d3d12_cache_variables(${ARG_CACHE_PREFIX})
    set(type "${${ARG_CACHE_PREFIX}_D3D12_TYPE}")
    set(nuget_id "${${ARG_CACHE_PREFIX}_D3D12_NUGET_ID}")
    set(nuget_version "${${ARG_CACHE_PREFIX}_D3D12_NUGET_VERSION}")
    set(nuget_hash "${${ARG_CACHE_PREFIX}_D3D12_NUGET_HASH}")
    set(dxheaders_tag "${${ARG_CACHE_PREFIX}_D3D12_HEADERS_TAG}")

    # Fetch open-source headers.
    if(TARGET_WINDOWS OR TARGET_WSL)
        FetchContent_Declare(
            dxheaders
            GIT_REPOSITORY https://github.com/microsoft/DirectX-Headers
            GIT_TAG ${dxheaders_tag}
        )
        FetchContent_MakeAvailable(dxheaders)
    endif()

    # Initialize target based on type.
    add_library(${target_name} INTERFACE)
    string(TOLOWER "${type}" type)
    if(type STREQUAL nuget)
        init_d3d12_target_nuget(${target_name} ${nuget_id} ${nuget_version} ${nuget_hash})
    elseif(type STREQUAL gdk)
        init_d3d12_target_gdk(${target_name})
    elseif(type STREQUAL winsdk)
        init_d3d12_target_winsdk(${target_name})
    elseif(type STREQUAL wsl)
        init_d3d12_target_wsl(${target_name})
    else()
        message(FATAL_ERROR "'${type}' is not a valid value for 'TYPE'.")
    endif()
endfunction()