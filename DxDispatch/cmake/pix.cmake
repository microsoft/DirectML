# =============================================================================
# Helper function to introduce a target representing a PIX event runtime.
#
# The main function is `add_pix_target`, which has the following parameters:
#
# - CACHE_PREFIX : string used to prefix cache variables associated with the function.
#
# The following cache variables are defined after calling the main function 
# (all variable names are prefixed with the value of CACHE_PREFIX):
#
# - PIX_TYPE
#       nuget  : Use NuGet distribution of PIX event runtime.
#       gdk    : Use GDK distribution of PIX event runtime.
#       none   : No dependency on PIX event runtime (stub target).
#
# - PIX_NUGET_ID : ID of the PIX event runtime NuGet package (TYPE == nuget).
# - PIX_NUGET_VERSION : Version of the PIX event runtime NuGet package (TYPE == nuget).
# - PIX_NUGET_HASH : SHA256 hash of the PIX event runtime NuGet package (TYPE == nuget).
# =============================================================================

include_guard()
include(FetchContent)
include(${CMAKE_CURRENT_LIST_DIR}/helper_platform.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/helper_redist.cmake)

# -----------------------------------------------------------------------------
# Add cache variables.
# -----------------------------------------------------------------------------
function(init_pix_cache_variables prefix)
    if(NOT prefix)
        message(FATAL_ERROR "The prefix must not be an empty string.")
    endif()

    # <PREFIX>_PIX_TYPE
    if(TARGET_XBOX)
        set(default_type gdk)
    elseif(TARGET_WINDOWS AND TARGET_ARCH MATCHES "^X64|ARM64|ARM64EC$")
        set(default_type nuget)
    else()
        set(default_type none)
    endif()
    set(${prefix}_PIX_TYPE 
        "${default_type}" 
        CACHE STRING "PIX dependency type"
    )
    set_property(CACHE ${prefix}_PIX_TYPE PROPERTY STRINGS nuget gdk none)

    # <PREFIX>_PIX_NUGET_ID
    set(${prefix}_PIX_NUGET_ID
        WinPixEventRuntime
        CACHE STRING "ID of the PIX event runtime NuGet package (TYPE == nuget)."
    )

    # <PREFIX>_PIX_NUGET_VERSION
    set(${prefix}_PIX_NUGET_VERSION
        1.0.230302001
        CACHE STRING "Version of the PIX event runtime NuGet package (TYPE == nuget)."
    )

    # <PREFIX>_PIX_NUGET_HASH
    set(${prefix}_PIX_NUGET_HASH 
        1CC9C6618A00F26375A8D98ADBA60620904FBF6A8E71007E14439CA01436589D
        CACHE STRING "SHA256 hash of the PIX event runtime NuGet package (TYPE == nuget)."
    )
endfunction()

# -----------------------------------------------------------------------------
# Init using a NuGet distribution.
# -----------------------------------------------------------------------------
function(init_pix_target_nuget target_name pkg_id pkg_version pkg_hash)
    if(NOT (TARGET_WINDOWS AND TARGET_ARCH MATCHES "^X64|ARM64|ARM64EC$"))
        message(FATAL_ERROR "PIX NuGet isn't available on this platform")
    endif()

    set(content ${target_name}_content)
    FetchContent_Declare(
        ${content}
        URL "https://www.nuget.org/api/v2/package/${pkg_id}/${pkg_version}"
        URL_HASH SHA256=${pkg_hash}
    )
    FetchContent_MakeAvailable(${content})

    target_include_directories(${target_name} INTERFACE "${${content}_SOURCE_DIR}/include")

    set(bin_path "${${content}_SOURCE_DIR}/bin/${TARGET_ARCH}")
    target_link_libraries(${target_name} INTERFACE "${bin_path}/WinPixEventRuntime.lib")
    target_compile_definitions(${target_name} INTERFACE USE_PIX)
    target_append_redist_file(${target_name} "${bin_path}/WinPixEventRuntime.dll")

    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "NuGet (${pkg_id}.${pkg_version})")
endfunction()

# -----------------------------------------------------------------------------
# Init using a GDK distribution.
# -----------------------------------------------------------------------------
function(init_pix_target_gdk target_name)
    if(NOT TARGET_XBOX)
        message(FATAL_ERROR "The GDK version of PIX only works on Xbox")
    endif()

    target_link_libraries(${target_name} INTERFACE pixevt.lib)
    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "GDK")
endfunction()

# -----------------------------------------------------------------------------
# Init as a stub dependency.
# -----------------------------------------------------------------------------
function(init_pix_target_stub target_name)
    set(stub_root ${CMAKE_CURRENT_BINARY_DIR}/stubs/pix)    
    file(WRITE ${stub_root}/WinPixEventRuntime/pix3.h [[
#pragma once
#define PIXBeginEvent(...)
#define PIXEndEvent(...)
#define PIX_COLOR(...)
#define PIXScopedEvent(...)]]
    )

    target_include_directories(${target_name} INTERFACE ${stub_root})
    target_compile_definitions(${target_name} INTERFACE PIX_NONE)
    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "None")
endfunction()

# -----------------------------------------------------------------------------
# Main function to add the target.
# -----------------------------------------------------------------------------
function(add_pix_target target_name)
    # Parse function args.
    set(params CACHE_PREFIX)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "${params}" "")

    # Parse cached args.
    init_pix_cache_variables(${ARG_CACHE_PREFIX})
    set(type "${${ARG_CACHE_PREFIX}_PIX_TYPE}")
    set(nuget_id "${${ARG_CACHE_PREFIX}_PIX_NUGET_ID}")
    set(nuget_version "${${ARG_CACHE_PREFIX}_PIX_NUGET_VERSION}")
    set(nuget_hash "${${ARG_CACHE_PREFIX}_PIX_NUGET_HASH}")

    # Initialize target based on type.
    add_library(${target_name} INTERFACE)
    string(TOLOWER "${type}" type)
    if(type STREQUAL nuget)
        init_pix_target_nuget(${target_name} ${nuget_id} ${nuget_version} ${nuget_hash})
    elseif(type STREQUAL gdk)
        init_pix_target_gdk(${target_name})
    elseif(type STREQUAL none)
        init_pix_target_stub(${target_name})
    else()
        message(FATAL_ERROR "'${type}' is not a valid value for 'TYPE'.")
    endif()
endfunction()