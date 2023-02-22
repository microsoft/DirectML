# =============================================================================
# Helper function to introduce a target representing a DXCompiler.
#
# The main function is `add_dxcompiler_target`, which has the following parameters:
#
# - CACHE_PREFIX : string used to prefix cache variables associated with the function.
# - GDK_DXCOMPILER_PATH : path to dxcompiler DLL (TYPE == gdk).
# 
# The following cache variables are defined after calling the main function 
# (all variable names are prefixed with the value of CACHE_PREFIX):
#
# - DXCOMPILER_TYPE
#       archive : Use a GitHub release archive.
#       gdk     : Use a GDK distribution of DXCompiler.
#       source  : Build DXCompiler from source.
#       none    : No dependency on DXCompiler (stub target).
#
# - DXCOMPILER_ARCHIVE_URL : URL of the GitHub release archive (TYPE == archive).
# - DXCOMPILER_ARCHIVE_HASH : SHA256 hash of the GitHub release archive (TYPE == archive).
# - DXCOMPILER_SOURCE_TAG : Git commit/tag hash of the DXCompiler repo  (TYPE == source).
# =============================================================================

include_guard()
include(FetchContent)
include(${CMAKE_CURRENT_LIST_DIR}/helper_platform.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/helper_redist.cmake)

# -----------------------------------------------------------------------------
# Add cache variables.
# -----------------------------------------------------------------------------
function(init_dxcompiler_cache_variables prefix)
    if(NOT prefix)
        message(FATAL_ERROR "The prefix must not be an empty string.")
    endif()

    # <PREFIX>_DXCOMPILER_TYPE
    if(TARGET_XBOX)
        set(default_type gdk)
    elseif(TARGET_WINDOWS AND TARGET_ARCH MATCHES "^X64|ARM64$")
        set(default_type archive)
    else()
        set(default_type none) # "source" type isn't implemented yet
    endif()
    set(${prefix}_DXCOMPILER_TYPE 
        "${default_type}" 
        CACHE STRING "DXCompiler dependency type"
    )
    set_property(CACHE ${prefix}_DXCOMPILER_TYPE PROPERTY STRINGS archive gdk source none)

    # <PREFIX>_DXCOMPILER_ARCHIVE_URL
    set(${prefix}_DXCOMPILER_ARCHIVE_URL 
        https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.7.2212/dxc_2022_12_16.zip
        CACHE STRING "URL of the GitHub release archive (TYPE == archive)."
    )

    # <PREFIX>_DXCOMPILER_ARCHIVE_HASH
    set(${prefix}_DXCOMPILER_ARCHIVE_HASH 
        ed77c7775fcf1e117bec8b5bb4de6735af101b733d3920dda083496dceef130f 
        CACHE STRING "SHA256 hash of the GitHub release archive (TYPE == archive)."
    )
    
    # <PREFIX>_DXCOMPILER_SOURCE_TAG
    set(${prefix}_DXCOMPILER_SOURCE_TAG 
        v1.7.2212
        CACHE STRING "Git commit/tag hash of the GitHub repo (TYPE == source)."
    )
endfunction()

# -----------------------------------------------------------------------------
# Init using GitHub release archive.
# -----------------------------------------------------------------------------
function(init_dxcompiler_target_archive target_name url hash)
    if(NOT(TARGET_WINDOWS AND TARGET_ARCH MATCHES "^X64|ARM64$"))
        message(FATAL_ERROR "DXCompiler release archives aren't supported on this platform.")
    endif()

    set(content ${target_name}_content)
    FetchContent_Declare(
        ${content}
        URL ${url}
        URL_HASH SHA256=${hash}
    )
    FetchContent_MakeAvailable(${content})

    target_include_directories(${target_name} INTERFACE "${${content}_SOURCE_DIR}/inc")
    target_link_libraries(${target_name} INTERFACE "${${content}_SOURCE_DIR}/lib/${TARGET_ARCH}/dxcompiler.lib")
    target_append_redist_file(${target_name} "${${content}_SOURCE_DIR}/bin/${TARGET_ARCH}/dxcompiler.dll")
    target_append_redist_file(${target_name} "${${content}_SOURCE_DIR}/bin/${TARGET_ARCH}/dxil.dll")

    cmake_path(GET url PARENT_PATH release_tag)
    cmake_path(GET release_tag FILENAME release_tag)
    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "Release (${release_tag})")
endfunction()

# -----------------------------------------------------------------------------
# Init using a GDK distribution.
# -----------------------------------------------------------------------------
function(init_dxcompiler_target_gdk target_name dxcompiler_path)
    if(NOT TARGET_XBOX)
        message(FATAL_ERROR "The GDK version of DXCompiler only works on Xbox")
    endif()

    target_append_redist_file(${target_name} "${dxcompiler_path}")
    target_link_libraries(${target_name} INTERFACE dxcompiler_${TARGET_XBOX_FILE_SUFFIX}.lib)
    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "GDK")
endfunction()

# -----------------------------------------------------------------------------
# Init using a version built from source.
# -----------------------------------------------------------------------------
function(init_dxcompiler_target_source target_name tag)
    message(FATAL_ERROR "Not yet implemented")
    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "Source (${commit})")
endfunction()

# -----------------------------------------------------------------------------
# Init as a stub dependency.
# -----------------------------------------------------------------------------
function(init_dxcompiler_target_stub target_name)
    set(stub_root ${CMAKE_CURRENT_BINARY_DIR}/stubs/dxcompiler)
    file(WRITE ${stub_root}/dxcapi.h [[
#pragma once
    ]])

    target_include_directories(${target_name} INTERFACE ${stub_root})
    target_compile_definitions(${target_name} INTERFACE DXCOMPILER_NONE)
    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "None")
endfunction()

# -----------------------------------------------------------------------------
# Main function to add the target.
# -----------------------------------------------------------------------------
function(add_dxcompiler_target target_name)
    # Parse function args.
    set(params CACHE_PREFIX GDK_DXCOMPILER_PATH)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "${params}" "")
    set(param_group DXCOMPILER)

    # Parse cached args.
    init_dxcompiler_cache_variables(${ARG_CACHE_PREFIX})
    set(type "${${ARG_CACHE_PREFIX}_DXCOMPILER_TYPE}")
    set(archive_url "${${ARG_CACHE_PREFIX}_DXCOMPILER_ARCHIVE_URL}")
    set(archive_hash "${${ARG_CACHE_PREFIX}_DXCOMPILER_ARCHIVE_HASH}")
    set(source_tag "${${ARG_CACHE_PREFIX}_DXCOMPILER_SOURCE_TAG}")

    # Initialize target based on type.
    add_library(${target_name} INTERFACE)
    string(TOLOWER ${type} type)
    if(type STREQUAL archive)
        init_dxcompiler_target_archive(${target_name} ${archive_url} ${archive_hash})
    elseif(type STREQUAL gdk)
        init_dxcompiler_target_gdk(${target_name} ${ARG_GDK_DXCOMPILER_PATH})
    elseif(type STREQUAL source)
        init_dxcompiler_target_source(${target_name} ${source_tag})
    elseif(type STREQUAL none)
        init_dxcompiler_target_stub(${target_name})
    else()
        message(FATAL_ERROR "'${type}' is not a valid value for 'TYPE'.")
    endif()
endfunction()