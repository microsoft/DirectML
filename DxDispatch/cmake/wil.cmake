# =============================================================================
# Helper function to introduce a target representing a D3D12 implementation.
# This target will implicitly depend on a system version of DXCore (Windows/WSL)
# or DXGI (GDK).
#
# The main function is `add_wil_target`, which has the following parameters:
#
# - CACHE_PREFIX     : string used to prefix cache variables associated with the function.
# 
# The following cache variables are defined after calling the main function 
# (all variable names are prefixed with the value of CACHE_PREFIX):
#
# - WIL_TYPE
#       source  : Build against WIL source repository.
#       none    : No dependency on WIL (stub target).
#
# - WIL_SOURCE_TAG : Git commit/tag hash of the WIL repo  (TYPE == source).
# =============================================================================

include_guard()
include(FetchContent)
include(${CMAKE_CURRENT_LIST_DIR}/helper_platform.cmake)

# -----------------------------------------------------------------------------
# Add cache variables.
# -----------------------------------------------------------------------------
function(init_wil_cache_variables prefix)
    if(NOT prefix)
        message(FATAL_ERROR "The prefix must not be an empty string.")
    endif()

    # <PREFIX>_WIL_TYPE
    if(TARGET_WSL)
        set(default_type none)
    else()
        set(default_type source)
    endif()
    set(${prefix}_WIL_TYPE 
        "${default_type}" 
        CACHE STRING "WIL dependency type"
    )
    set_property(CACHE ${prefix}_WIL_TYPE PROPERTY STRINGS source none)

    # <PREFIX>_WIL_SOURCE_TAG
    set(${prefix}_WIL_SOURCE_TAG
        2e225973d6c2ecf17fb4d376ddbeedb6db7dd82f
        CACHE STRING "Git commit/tag hash of the WIL repo  (TYPE == source)."
    )
endfunction()

# -----------------------------------------------------------------------------
# Init using a version built from source.
# -----------------------------------------------------------------------------
function(init_wil_target_source target_name git_tag)
    if(TARGET_WSL)
        message(FATAL_ERROR "WIL does not build for Linux.")
    endif()

    FetchContent_Declare(
        wil
        GIT_REPOSITORY https://github.com/Microsoft/wil
        GIT_TAG ${git_tag}
    )

    FetchContent_GetProperties(wil)
    if(NOT wil_POPULATED)
        set(WIL_BUILD_PACKAGING OFF CACHE INTERNAL "Sets option to build the packaging, default on")
        set(WIL_BUILD_TESTS OFF CACHE INTERNAL "Sets option to build the unit tests, default on")
        FetchContent_Populate(wil)
        add_subdirectory(${wil_SOURCE_DIR} ${wil_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()

    target_link_libraries(${target_name} INTERFACE WIL)
    target_compile_definitions(${target_name} INTERFACE DMLX_USE_WIL)
    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "Source (${git_tag})")
endfunction()

# -----------------------------------------------------------------------------
# Init as a stub dependency.
# -----------------------------------------------------------------------------
function(init_wil_target_none target_name)
    set(stub_root ${CMAKE_CURRENT_BINARY_DIR}/stubs/wil)
    file(WRITE ${stub_root}/wil/result.h [[
#pragma once

#define CATCH_RETURN() \
catch (HRESULT hr) \
{ \
    if(FAILED(hr)) \
        return hr; \
    else \
        return hr + 0x80000000; \
} \
catch (std::invalid_argument const& ex) \
{ \
    return E_INVALIDARG; \
} \
catch (std::out_of_range const& ex) \
{ \
    return E_INVALIDARG; \
} \
catch (std::bad_alloc const& ex) \
{ \
    return E_OUTOFMEMORY; \
} \
catch (std::exception const& ex) \
{ \
    return E_UNEXPECTED; \
} \
catch (...) \
{ \
    return E_FAIL; \
}

#define THROW_IF_FAILED(hr) \
do \
{ \
    if (FAILED(hr)) \
    { \
        throw hr; \
    } \
} while (0)

#define THROW_HR(hr) \
do \
{ \
    throw hr; \
} while (0)

#define RETURN_IF_FAILED(hr) \
do \
{ \
    if (FAILED(hr)) \
    { \
        return hr; \
    } \
} while (0)

#define RETURN_HR_IF_NULL(hr, ptr) \
do \
{ \
    if (!ptr) \
    { \
        return hr; \
    } \
} while (0)
    ]])

    target_include_directories(${target_name} INTERFACE ${stub_root})
    target_compile_definitions(${target_name} INTERFACE WIL_NONE)
    set_property(TARGET ${target_name} PROPERTY DX_COMPONENT_CONFIG "None")
endfunction()

# -----------------------------------------------------------------------------
# Main function to add the target.
# -----------------------------------------------------------------------------
function(add_wil_target target_name)
    # Parse function args.
    set(params CACHE_PREFIX)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "${params}" "")

    # Parse cached args.
    init_wil_cache_variables(${ARG_CACHE_PREFIX})
    set(type "${${ARG_CACHE_PREFIX}_WIL_TYPE}")
    set(source_tag "${${ARG_CACHE_PREFIX}_WIL_SOURCE_TAG}")

    # Initialize target based on type.
    add_library(${target_name} INTERFACE)
    string(TOLOWER ${type} type)
    if(type STREQUAL source)
        init_wil_target_source(${target_name} ${source_tag})
    elseif(type STREQUAL none)
        init_wil_target_none(${target_name})
    else()
        message(FATAL_ERROR "'${type}' is not a valid value for 'TYPE'.")
    endif()

endfunction()