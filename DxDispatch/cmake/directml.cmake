# =============================================================================
# Helper function to introduce a target representing a DirectML implementation.
#
# The main function is `add_directml_target`, which has the following parameters:
#
# - CACHE_PREFIX : string used to prefix cache variables associated with the function.
# 
# The following cache variables are defined after calling the main function 
# (all variable names are prefixed with the value of CACHE_PREFIX):
#
# - DIRECTML_TYPE
#       nuget  : Use NuGet distribution of DirectML. Adds a redistributable dependency.
#       winsdk : Use Windows SDK. Adds a system dependency.
#       local  : Use a local copy of DirectML (testing only).
#
# - DIRECTML_NUGET_ID : ID of the DirectML NuGet package (TYPE == nuget).
# - DIRECTML_NUGET_VERSION : Version of the DirectML NuGet package (TYPE == nuget).
# - DIRECTML_NUGET_HASH : SHA256 hash of the DirectML NuGet package (TYPE == nuget).
# - DIRECTML_LOCAL_PATH : Path to a local build of DirectML (TYPE == local).
# =============================================================================

include_guard()
include(FetchContent)
include(${CMAKE_CURRENT_LIST_DIR}/helper_platform.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/helper_redist.cmake)

# -----------------------------------------------------------------------------
# Add cache variables.
# -----------------------------------------------------------------------------
function(init_directml_cache_variables prefix)
    if(NOT prefix)
        message(FATAL_ERROR "The prefix must not be an empty string.")
    endif()

    # <PREFIX>_DIRECTML_TYPE
    set(default_type nuget)
    set(${prefix}_DIRECTML_TYPE 
        "${default_type}" 
        CACHE STRING "DirectML dependency type"
    )
    set_property(CACHE ${prefix}_DIRECTML_TYPE PROPERTY STRINGS nuget winsdk wsl)

    # <PREFIX>_DIRECTML_NUGET_ID
    set(${prefix}_DIRECTML_NUGET_ID
        Microsoft.AI.DirectML
        CACHE STRING "ID of the DirectML NuGet package (TYPE == nuget)."
    )

    # <PREFIX>_DIRECTML_NUGET_VERSION
    set(${prefix}_DIRECTML_NUGET_VERSION
        1.15.0
        CACHE STRING "Version of the DirectML NuGet package (TYPE == nuget)."
    )

    # <PREFIX>_DIRECTML_NUGET_HASH
    set(${prefix}_DIRECTML_NUGET_HASH 
        10d175f8e97447712b3680e3ac020bbb8eafdf651332b48f09ffee2eec801c23
        CACHE STRING "SHA256 hash of the DirectML NuGet package (TYPE == nuget)."
    )

    # <PREFIX>_DIRECTML_LOCAL_PATH
    set(${prefix}_DIRECTML_LOCAL_PATH
        ""
        CACHE STRING "Path to a local build of DirectML (TYPE == local)."
    )
endfunction()

# -----------------------------------------------------------------------------
# Init using a NuGet distribution.
# -----------------------------------------------------------------------------
function(init_directml_target_nuget target_name pkg_id pkg_version pkg_hash)
    if(TARGET_XBOX)
        message(FATAL_ERROR "The DirectML NuGet doesn't support Xbox")
    endif()

    set(content ${target_name}_content)
    FetchContent_Declare(
        ${content}
        URL "https://www.nuget.org/api/v2/package/${pkg_id}/${pkg_version}"
        URL_HASH SHA256=${pkg_hash}
    )
    FetchContent_MakeAvailable(${content})

    if(TARGET_WINDOWS)
        if (TARGET_ARCH STREQUAL X86)
            set(bin_path ${${content}_SOURCE_DIR}/bin/x86-win)
        elseif (TARGET_ARCH STREQUAL X64)
            set(bin_path ${${content}_SOURCE_DIR}/bin/x64-win)
        elseif (TARGET_ARCH STREQUAL ARM)
            set(bin_path ${${content}_SOURCE_DIR}/bin/arm-win)
        elseif (TARGET_ARCH STREQUAL ARM64)
            set(bin_path ${${content}_SOURCE_DIR}/bin/arm64-win)
        endif()
        target_append_redist_file(${target_name} "${bin_path}/DirectML.dll")
        target_append_redist_file(${target_name} "${bin_path}/DirectML.Debug.dll")
    else()
        set(bin_path ${${content}_SOURCE_DIR}/bin/x64-linux)
        target_append_redist_file(${target_name} "${bin_path}/libdirectml.so")
    endif()

    target_include_directories(${target_name} INTERFACE "${${content}_SOURCE_DIR}/include")
    target_compile_definitions(${target_name} INTERFACE DML_TARGET_VERSION_USE_LATEST)
    target_compile_definitions(${target_name} INTERFACE DML_NUGET_VERSION)

    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "NuGet (${pkg_id}.${pkg_version})")
endfunction()

# -----------------------------------------------------------------------------
# Init using Windows SDK.
# -----------------------------------------------------------------------------
function(init_directml_target_winsdk target_name)
    if(NOT TARGET_WINDOWS)
        message(FATAL_ERROR "The SDK version of DirectML only works on Windows")
    endif()

    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "Windows SDK")
endfunction()

# -----------------------------------------------------------------------------
# Init using a local build.
# -----------------------------------------------------------------------------
function(init_directml_target_local target_name local_path)
    if(NOT IS_DIRECTORY "${local_path}")
        message(FATAL_ERROR "'${local_path}' is not a directory. You must set DXD_DIRECTML_LOCAL_PATH to a directory containing pre-built DirectML.")
    endif()

    # DirectML.dll is required.
    set(directml_dll_path "${local_path}/bin/DirectML.dll")
    if(NOT EXISTS ${directml_dll_path})
        message(FATAL_ERROR "Could not find '${directml_dll_path}'")
    endif()
    target_append_redist_file(${target_name} ${directml_dll_path})

    # DirectML.Debug.dll is optional.
    set(directml_debug_dll_path "${local_path}/bin/DirectML.Debug.dll")
    if(EXISTS ${directml_debug_dll_path})
        target_append_redist_file(${target_name} ${directml_debug_dll_path})
    endif()

    # DirectML.lib is required.
    set(directml_lib_path "${local_path}/lib/DirectML.lib")
    if(NOT EXISTS ${directml_lib_path})
        message(FATAL_ERROR "Could not find '${directml_lib_path}'")
    endif()
    target_link_libraries(${target_name} INTERFACE ${directml_lib_path})

    # Include dir must exist with DirectML.h in it.
    set(directml_h_path "${local_path}/include/DirectML.h")
    if(NOT EXISTS ${directml_h_path})
        message(FATAL_ERROR "Could not find '${directml_h_path}'")
    endif()
    target_include_directories(${target_name} INTERFACE "${local_path}/include")
    target_compile_definitions(${target_name} INTERFACE DML_TARGET_VERSION_USE_LATEST)
    
    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "Local")
endfunction()

# -----------------------------------------------------------------------------
# Init using a local build.
# -----------------------------------------------------------------------------
function(add_directml_guids target_name)
    # The IIDs for DirectML interfaces need to be defined for platforms not using
    # MSVC extensions (mainly WSL). This block of logic is clunky and unnecessarily ties 
    # IID initialization to platform instead of compiler, but the various IUnknown
    # helpers (from DirectX-Headers) used to init the IIDs assume a non-Windows target.
    if(NOT TARGET_WSL)
        return()
    endif()

    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/directml_guids/directml_guids.h [[
#pragma once
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
#include <directx/dxcore.h>
#include <dxguids/dxguids.h>
#include <DirectML.h>
WINADAPTER_IID(IDMLObject, 0xc8263aac, 0x9e0c, 0x4a2d, 0x9b, 0x8e, 0x00, 0x75, 0x21, 0xa3, 0x31, 0x7c);
WINADAPTER_IID(IDMLDevice, 0x6dbd6437, 0x96fd, 0x423f, 0xa9, 0x8c, 0xae, 0x5e, 0x7c, 0x2a, 0x57, 0x3f);
WINADAPTER_IID(IDMLDeviceChild, 0x27e83142, 0x8165, 0x49e3, 0x97, 0x4e, 0x2f, 0xd6, 0x6e, 0x4c, 0xb6, 0x9d);
WINADAPTER_IID(IDMLPageable, 0xb1ab0825, 0x4542, 0x4a4b, 0x86, 0x17, 0x6d, 0xde, 0x6e, 0x8f, 0x62, 0x01);
WINADAPTER_IID(IDMLOperator, 0x26caae7a, 0x3081, 0x4633, 0x95, 0x81, 0x22, 0x6f, 0xbe, 0x57, 0x69, 0x5d);
WINADAPTER_IID(IDMLDispatchable, 0xdcb821a8, 0x1039, 0x441e, 0x9f, 0x1c, 0xb1, 0x75, 0x9c, 0x2f, 0x3c, 0xec);
WINADAPTER_IID(IDMLCompiledOperator, 0x6b15e56a, 0xbf5c, 0x4902, 0x92, 0xd8, 0xda, 0x3a, 0x65, 0x0a, 0xfe, 0xa4);
WINADAPTER_IID(IDMLOperatorInitializer, 0x427c1113, 0x435c, 0x469c, 0x86, 0x76, 0x4d, 0x5d, 0xd0, 0x72, 0xf8, 0x13);
WINADAPTER_IID(IDMLBindingTable, 0x29c687dc, 0xde74, 0x4e3b, 0xab, 0x00, 0x11, 0x68, 0xf2, 0xfc, 0x3c, 0xfc);
WINADAPTER_IID(IDMLCommandRecorder, 0xe6857a76, 0x2e3e, 0x4fdd, 0xbf, 0xf4, 0x5d, 0x2b, 0xa1, 0x0f, 0xb4, 0x53);
WINADAPTER_IID(IDMLDebugDevice, 0x7d6f3ac9, 0x394a, 0x4ac3, 0x92, 0xa7, 0x39, 0x0c, 0xc5, 0x7a, 0x82, 0x17);
WINADAPTER_IID(IDMLDevice1, 0xa0884f9a, 0xd2be, 0x4355, 0xaa, 0x5d, 0x59, 0x01, 0x28, 0x1a, 0xd1, 0xd2);]]
    )

    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/directml_guids/directml_guids.cpp [[
#define INITGUID
#include "directml_guids.h"]]
    )

    target_sources(${target_name} INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/directml_guids/directml_guids.cpp)
    target_include_directories(${target_name} INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/directml_guids)
endfunction()

# -----------------------------------------------------------------------------
# Main function to add the target.
# -----------------------------------------------------------------------------
function(add_directml_target target_name)
    # Parse function args.
    set(params CACHE_PREFIX)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "${params}" "")

    # Parse cached args.
    init_directml_cache_variables(${ARG_CACHE_PREFIX})
    set(type "${${ARG_CACHE_PREFIX}_DIRECTML_TYPE}")
    set(nuget_id "${${ARG_CACHE_PREFIX}_DIRECTML_NUGET_ID}")
    set(nuget_version "${${ARG_CACHE_PREFIX}_DIRECTML_NUGET_VERSION}")
    set(nuget_hash "${${ARG_CACHE_PREFIX}_DIRECTML_NUGET_HASH}")
    set(local_path "${${ARG_CACHE_PREFIX}_DIRECTML_LOCAL_PATH}")

    add_library(${target_name} INTERFACE)

    # Always make DirectMLX header available.
    FetchContent_Declare(
        directmlx
        URL https://raw.githubusercontent.com/microsoft/DirectML/91cc5e5e823d582938c3407ec65e8e4a96e020a1/Libraries/DirectMLX.h
        DOWNLOAD_NO_EXTRACT 1
    )
    FetchContent_MakeAvailable(directmlx)
    target_include_directories(${target_name} INTERFACE ${directmlx_SOURCE_DIR})
    add_directml_guids(${target_name})

    # Initialize target based on type.
    string(TOLOWER "${type}" type)
    if(type STREQUAL nuget)
        init_directml_target_nuget(${target_name} ${nuget_id} ${nuget_version} ${nuget_hash})
    elseif(type STREQUAL winsdk)
        init_directml_target_winsdk(${target_name})
    elseif(type STREQUAL local)
        init_directml_target_local(${target_name} "${local_path}")
    else()
        message(FATAL_ERROR "'${type}' is not a valid value for 'TYPE'.")
    endif()
endfunction()